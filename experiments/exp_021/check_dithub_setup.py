#!/usr/bin/env python3
"""exp_021: DitHub 実装の学習前検証 (implementation_plan.md 6 節の 7 項目).

  1. パラメータ監査: 学習可能 = lora_* のみ、対象層が規則通り
  2. attention 分解の等価性: DecomposedMHA と nn.MultiheadAttention の一致
  3. θ0 等価性: B=0 の初期状態で検出出力が素の MM-GDINO と一致 (ckpt ロード込み)
  4. 勾配疎通: warmup / specialization で勾配の流れ先が正しい
  5. enable_per_class の単体テスト: コピーと式3融合の数値確認
  6. 評価合成の検証: 評価 forward = クラス別 ΔW 平均の手計算と一致
  7. ZCOCO 名前重複の事前調査: ドメインクラス × COCO 80 クラスの一致

使い方 (リポジトリルートで):
    python experiments/exp_021/check_dithub_setup.py
"""
import copy
import random
import sys

import torch
from torch import nn

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

ZIRA_DIR = 'experiments/exp_021'
CFG = f'{ZIRA_DIR}/configs/dithub_underwater.py'
BASE_CFG = ('configs/mm_grounding_dino/'
            'grounding_dino_swin-t_finetune_8xb4_20e_underwater.py')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
results = []


def report(name, ok, detail):
    results.append((name, ok))
    print(f"[{'OK' if ok else 'NG'}] {name}: {detail}")


def build(cfg_path):
    cfg = Config.fromfile(cfg_path)
    init_default_scope('mmdet')
    from mmdet.registry import MODELS
    model = MODELS.build(cfg.model)
    model.init_weights()
    return cfg, model


def main():
    torch.manual_seed(0)
    random.seed(0)
    from mmdet.models.layers.dithub_layers import (STATE, DecomposedMHA,
                                                   DitHubLinear,
                                                   canonical_key)

    # ---- 1. パラメータ監査 -------------------------------------------------
    cfg, model = build(CFG)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    bad_train = [n for n in trainable if 'lora_' not in n]
    layers = model.dithub_layer_names
    bad_layers = [l for l in layers
                  if l.startswith(('bbox_head', 'decoder', 'backbone',
                                   'language_model', 'neck', 'text_feat_map'))]
    must_have = [
        'memory_trans_fc',
        'encoder.layers.0.self_attn.sampling_offsets',
        'encoder.layers.0.self_attn.attention_weights',
        'encoder.text_layers.0.self_attn.attn.linear_q',
        'encoder.fusion_layers.0.attn.v_proj',
    ]
    missing = [m for m in must_have if m not in layers]
    # 期待数: 画像側 6層×(MSDA 4 + FFN 2) + テキスト側 6層×(attn 4 + FFN 2)
    #         + 融合 6層×6 + memory_trans_fc 1 = 36+36+36+1 = 109
    report('1 パラメータ監査',
           not bad_train and not bad_layers and not missing
           and len(layers) == 109,
           f'対象層={len(layers)} (期待109), 非LoRAの学習対象={bad_train or "なし"}, '
           f'除外漏れ={bad_layers or "なし"}, 必須層の欠落={missing or "なし"}')

    # ---- 2. attention 分解の等価性 -----------------------------------------
    torch.manual_seed(1)
    mha = nn.MultiheadAttention(256, 4, dropout=0.0).eval()
    dec = DecomposedMHA(256, 4, dropout=0.0).eval()
    dec.load_from_multihead(mha)
    q = torch.randn(20, 3, 256)
    mask = torch.rand(3 * 4, 20, 20) > 0.8
    mask[:, :, 0] = False  # 全遮蔽行を避ける
    with torch.no_grad():
        ref, _ = mha(q, q, q, attn_mask=mask)
        out, _ = dec(q, q, q, attn_mask=mask)
    d = (ref - out).abs().max().item()
    report('2 attention 分解の等価性', d < 1e-5, f'最大差={d:.3e}')

    # ---- 3. θ0 等価性 ------------------------------------------------------
    load_checkpoint(model, cfg.load_from, map_location='cpu')
    base_cfg = Config.fromfile(BASE_CFG)
    init_default_scope('mmdet')
    from mmdet.registry import MODELS
    base_model = MODELS.build(base_cfg.model)
    base_model.init_weights()
    load_checkpoint(base_model, cfg.load_from, map_location='cpu')
    model = model.to(DEVICE).eval()
    base_model = base_model.to(DEVICE).eval()
    val_loader = Runner.build_dataloader(base_cfg.val_dataloader)
    batch = next(iter(val_loader))
    with torch.no_grad():
        pred_d = model.test_step(copy.deepcopy(batch))[0].pred_instances
        pred_b = base_model.test_step(copy.deepcopy(batch))[0].pred_instances
    d_box = (pred_d.bboxes - pred_b.bboxes).abs().max().item()
    d_score = (pred_d.scores - pred_b.scores).abs().max().item()
    report('3 θ0 等価性', d_box < 1e-3 and d_score < 1e-4,
           f'bbox 最大差={d_box:.3e}, score 最大差={d_score:.3e} '
           f'(B=0 のため attention 分解の丸め以外は厳密一致のはず)')
    del base_model
    torch.cuda.empty_cache()

    # ---- 4. 勾配疎通 (warmup / specialization) -----------------------------
    train_loader = Runner.build_dataloader(cfg.train_dataloader)
    train_batch = next(iter(train_loader))

    def one_step():
        data = model.data_preprocessor(train_batch, True)
        losses = model.loss(data['inputs'], data['data_samples'])
        parsed, _ = model.parse_losses(losses)
        model.zero_grad(set_to_none=True)
        parsed.backward()

    # 恒等ゼロ項 (loss_lora_tether) により全 LoRA パラメータに勾配テンソルは
    # 存在する。実際に学習信号が流れているかは勾配ノルムの非ゼロで判定する。
    def grad_nonzero(name_frag):
        return [n for n, p in model.named_parameters()
                if p.requires_grad and name_frag in n
                and p.grad is not None and p.grad.abs().max() > 0]

    # B=0 の初期状態では dL/dA が B に比例して厳密にゼロになる (LoRA ゼロ初期化
    # の力学。公式も同じ)。勾配の「流れ先」を見るため B を微小乱数化しておく。
    with torch.no_grad():
        for n, p in model.named_parameters():
            if 'shared_lora_b' in n:
                p.normal_(0, 0.01)
    model.train()
    one_step()  # warmup フェーズ
    warm_ok = (bool(grad_nonzero('warmup_lora_a'))
               and bool(grad_nonzero('shared_lora_b'))
               and not grad_nonzero('per_class_lora_A'))

    model.enable_per_class()
    one_step()  # specialization フェーズ
    chosen = set(STATE.train_class_keys)
    g_cls = grad_nonzero('per_class_lora_A')
    spec_ok = (not grad_nonzero('warmup_lora_a') and bool(g_cls))
    wrong_class = [n for n in g_cls if not any(k in n for k in chosen)]
    report('4 勾配疎通', warm_ok and spec_ok and not wrong_class,
           f'warmup 期の流れ先 OK={warm_ok}, specialization 期 OK={spec_ok}, '
           f'選択外クラスへの勾配={wrong_class or "なし"} (選択={sorted(chosen)})')
    model.zero_grad(set_to_none=True)
    model.eval()

    # ---- 5. enable_per_class 単体テスト ------------------------------------
    lay = DitHubLinear(8, 130, class_keys=['class_a', 'class_b'], r=4)
    with torch.no_grad():
        lay.warmup_lora_a.normal_()
        lay.per_class_lora_A['class_b'].normal_()
    old_b = lay.per_class_lora_A['class_b'].data.clone()
    lay.enable_per_class(lambda_a=0.3, trained_keys={'class_b'})
    copy_ok = torch.equal(lay.per_class_lora_A['class_a'].data,
                          lay.warmup_lora_a.data)
    merge_expect = 0.3 * lay.warmup_lora_a.data + 0.7 * old_b
    merge_ok = torch.allclose(lay.per_class_lora_A['class_b'].data,
                              merge_expect, atol=1e-7)
    report('5 enable_per_class 単体', copy_ok and merge_ok,
           f'新規クラス=warmupコピー: {copy_ok}, 学習済みクラス=式3融合: {merge_ok}')

    # ---- 6. 評価合成の検証 --------------------------------------------------
    with torch.no_grad():
        lay.shared_lora_b.normal_(0, 0.1)
        for k in lay.per_class_lora_A:
            lay.per_class_lora_A[k].normal_()
    lay.eval()
    STATE.eval_class_keys = ['class_a', 'class_b', 'class_absent']
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        out = lay(x)
        delta_w = torch.stack([
            lay.shared_lora_b @ lay.per_class_lora_A['class_a'],
            lay.shared_lora_b @ lay.per_class_lora_A['class_b'],
        ]).mean(0) * lay.scaling
        ref = torch.nn.functional.linear(x, lay.weight + delta_w, lay.bias)
    d = (out - ref).abs().max().item()
    report('6 評価合成', d < 1e-5, f'手計算との最大差={d:.3e}')

    # ---- 7. ZCOCO 名前重複の事前調査 ----------------------------------------
    from mmdet.datasets import CocoDataset
    coco_keys = {canonical_key(c) for c in CocoDataset.METAINFO['classes']}
    print('--- ドメインクラス × COCO 80 の重複 ---')
    overlaps = {}
    for d_name in ['underwater', 'aerial', 'videogames', 'microscopic',
                   'documents', 'electromagnetic']:
        dcfg = Config.fromfile(
            f'configs/mm_grounding_dino/'
            f'grounding_dino_swin-t_finetune_8xb4_20e_{d_name}.py')
        keys = {canonical_key(c) for c in dcfg['class_name']}
        ov = sorted(keys & coco_keys)
        overlaps[d_name] = ov
        print(f'  {d_name}: {len(ov)} 件 {ov}')
    report('7 ZCOCO 名前重複調査', True,
           '重複ゼロのドメインは ZCOCO が θ0=0.504 と一致するはず（手法の性質）')


    # ---- 8. specialization のサンプル別経路の数値検証 -----------------------
    lay8 = DitHubLinear(8, 130, class_keys=['class_a', 'class_b', 'class_c'],
                        r=4)
    with torch.no_grad():
        lay8.shared_lora_b.normal_(0, 0.1)
        for k in lay8.per_class_lora_A:
            lay8.per_class_lora_A[k].normal_()
    lay8.dithub_per_class_enabled.fill_(1)
    lay8.train()
    STATE.train_class_keys = ['class_a', 'class_c']
    x8 = torch.randn(2, 5, 8)
    out8 = lay8(x8)
    refs = []
    for i, key in enumerate(STATE.train_class_keys):
        dw = (lay8.shared_lora_b @ lay8.per_class_lora_A[key]) * lay8.scaling
        refs.append(torch.nn.functional.linear(x8[i], lay8.weight + dw,
                                               lay8.bias))
    d8 = (out8 - torch.stack(refs)).abs().max().item()
    report('8 サンプル別bmm経路', d8 < 1e-5, f'サンプル毎の手計算との最大差={d8:.3e}')

    # ---- 9. 推論時のモジュール選択 (検出器レベル) ----------------------------
    # LoRA を乱数化し per_class を有効化した状態で:
    #   (a) ドメインのプロンプト -> モジュールが選択され出力が変わる
    #   (b) 未知クラスのプロンプト -> 何も適用されず B=0 と厳密一致
    with torch.no_grad():
        for n, p in model.named_parameters():
            if 'lora_' in n:
                p.normal_(0, 0.02)
    lora_backup = {n: p.detach().clone() for n, p in model.named_parameters()
                   if 'lora_' in n}

    def zero_b():
        with torch.no_grad():
            for n, p in model.named_parameters():
                if 'shared_lora_b' in n:
                    p.zero_()

    def restore():
        with torch.no_grad():
            for n, p in model.named_parameters():
                if 'lora_' in n:
                    p.copy_(lora_backup[n])

    model.eval()
    with torch.no_grad():
        pred_on = model.test_step(copy.deepcopy(batch))[0].pred_instances
        zero_b()
        pred_off = model.test_step(copy.deepcopy(batch))[0].pred_instances
        restore()
    d_engage = (pred_on.scores - pred_off.scores).abs().max().item()
    probe = model.memory_trans_fc
    n_hit = sum(1 for k in STATE.eval_class_keys
                if k in probe.per_class_lora_A)

    fake = copy.deepcopy(batch)
    for ds in fake['data_samples']:
        ds.set_metainfo(dict(text=('zzzunknownclass', 'anotherfakething')))
    with torch.no_grad():
        pred_f_on = model.test_step(copy.deepcopy(fake))[0].pred_instances
        zero_b()
        pred_f_off = model.test_step(copy.deepcopy(fake))[0].pred_instances
        restore()
    d_fake = (pred_f_on.scores - pred_f_off.scores).abs().max().item()
    report('9 推論時モジュール選択',
           d_engage > 1e-3 and n_hit == 28 and d_fake < 1e-6,
           f'ドメインプロンプトで選択={n_hit}/28 クラス・出力変化={d_engage:.3e} '
           f'(>1e-3 期待), 未知プロンプトでの変化={d_fake:.3e} (=0 期待)')

    # ---- 10. フックの配線 ---------------------------------------------------
    import types
    from mmdet.engine.hooks.dithub_phase_hook import DitHubPhaseHook
    for m in model.modules():
        if isinstance(m, DitHubLinear):
            m.dithub_per_class_enabled.fill_(0)
    logger = types.SimpleNamespace(info=lambda *a, **k: None)
    hook = DitHubPhaseHook(warmup_epochs=10)
    runner = types.SimpleNamespace(epoch=9, model=model, logger=logger)
    hook.before_train_epoch(runner)
    fired_early = any(bool(m.dithub_per_class_enabled) for m in model.modules()
                      if isinstance(m, DitHubLinear))
    runner.epoch = 10
    hook.before_train_epoch(runner)
    fired = all(bool(m.dithub_per_class_enabled) for m in model.modules()
                if isinstance(m, DitHubLinear))
    report('10 フック配線', (not fired_early) and fired,
           f'epoch9では発火せず={not fired_early}, epoch10で全層有効={fired}')

    print('\n===== 結果 =====')
    n_ok = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'OK' if ok else 'NG'}  {name}")
    print(f'{n_ok}/{len(results)} passed')
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == '__main__':
    main()
