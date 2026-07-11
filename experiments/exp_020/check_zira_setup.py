#!/usr/bin/env python3
"""exp_020: ZiRa 実装の学習前検証 (implementation_plan.md 6 節の 6 項目).

  1. パラメータ監査: 学習可能 = RDB (hlrb/llrb/scaling) のみ、本体は全凍結
  2. θ0 等価性: RDB 初期状態の ZiRa モデルと素の MM-GDINO の検出出力一致
  3. 勾配疎通: 1 step で RDB 全パラメータに勾配、本体には勾配なし
  4. loss_zil: 損失辞書に loss_zil が現れ、初期値がほぼ 0
  5. 推論等価性: RDB 乱数状態で「全和 forward」=「merge 後の forward」
  6. 逐次開始状態: merge 後の state が LLRB=融合値 / HLRB=1e-8 / s=0.1

使い方 (リポジトリルートで):
    python experiments/exp_020/check_zira_setup.py
"""
import copy
import sys

import torch

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

sys.path.insert(0, 'experiments/exp_020')
from merge_hlrb import HLRB_INIT, SCALING_INIT, merge_state_dict  # noqa: E402

ZIRA_CFG = 'experiments/exp_020/configs/zira_underwater.py'
BASE_CFG = ('configs/mm_grounding_dino/'
            'grounding_dino_swin-t_finetune_8xb4_20e_underwater.py')

RDB_TOKENS = ('hlrb', 'llrb', 'scaling')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
results = []


def report(name, ok, detail):
    results.append((name, ok))
    print(f"[{'OK' if ok else 'NG'}] {name}: {detail}")


def build_model(cfg_path):
    cfg = Config.fromfile(cfg_path)
    init_default_scope('mmdet')
    from mmdet.registry import MODELS
    model = MODELS.build(cfg.model)
    model.init_weights()
    return cfg, model


def main():
    torch.manual_seed(0)

    # ---- 1. パラメータ監査 -------------------------------------------------
    cfg, model = build_model(ZIRA_CFG)
    trainable = [(n, tuple(p.shape)) for n, p in model.named_parameters()
                 if p.requires_grad]
    frozen_n = sum(1 for _, p in model.named_parameters()
                   if not p.requires_grad)
    bad = [n for n, _ in trainable
           if not any(t in n for t in RDB_TOKENS)]
    # 期待: text 5 (hlrb w/b, llrb w/b, scaling) + neck 4 RDB x 5 = 25
    report('1 パラメータ監査',
           len(trainable) == 25 and not bad and frozen_n == 896,
           f'trainable={len(trainable)} (期待25), 凍結={frozen_n} (期待896), '
           f'非RDBの学習対象={bad or "なし"}')
    for n, s in trainable:
        print(f'    trainable: {n} {s}')

    # ---- checkpoint ロード -------------------------------------------------
    ckpt_path = cfg.load_from
    load_checkpoint(model, ckpt_path, map_location='cpu')
    # 事前学習 ckpt に RDB キーは存在しないので、ロード後も RDB は初期状態の
    # まま (HLRB=1e-8, LLRB=0, s=0.1) であることを直接確認する
    sd = model.state_dict()
    ok_load = True
    for k, v in sd.items():
        if k.endswith('.hlrb.weight') or k.endswith('.hlrb.bias'):
            ok_load &= torch.allclose(v, torch.full_like(v, HLRB_INIT))
        elif k.endswith('.llrb.weight') or k.endswith('.llrb.bias'):
            ok_load &= torch.allclose(v, torch.zeros_like(v))
        elif k.endswith('.scaling'):
            ok_load &= torch.allclose(v, torch.full_like(v, SCALING_INIT))
    report('1b ckpt ロード後の RDB 初期状態', bool(ok_load),
           'HLRB=1e-8 / LLRB=0 / s=0.1 を維持' if ok_load else 'RDB が汚染された')

    _, base_model = build_model(BASE_CFG)
    load_checkpoint(base_model, ckpt_path, map_location='cpu')

    model = model.to(DEVICE).eval()
    base_model = base_model.to(DEVICE).eval()

    # ---- 2. θ0 等価性 (検出出力レベル) -------------------------------------
    base_cfg = Config.fromfile(BASE_CFG)
    val_loader = Runner.build_dataloader(base_cfg.val_dataloader)
    batch = next(iter(val_loader))
    with torch.no_grad():
        pred_z = model.test_step(copy.deepcopy(batch))[0].pred_instances
        pred_b = base_model.test_step(copy.deepcopy(batch))[0].pred_instances
    d_box = (pred_z.bboxes - pred_b.bboxes).abs().max().item()
    d_score = (pred_z.scores - pred_b.scores).abs().max().item()
    report('2 θ0 等価性', d_box < 1e-3 and d_score < 1e-4,
           f'bbox 最大差={d_box:.3e}, score 最大差={d_score:.3e} '
           f'(RDB 初期値 1e-8 由来の微小差のみ許容)')
    del base_model
    torch.cuda.empty_cache()

    # ---- 3+4. 勾配疎通と loss_zil ------------------------------------------
    train_loader = Runner.build_dataloader(cfg.train_dataloader)
    train_batch = next(iter(train_loader))
    model.train()
    data = model.data_preprocessor(train_batch, True)
    losses = model.loss(data['inputs'], data['data_samples'])
    assert 'loss_zil' in losses, 'loss_zil が損失辞書にない'
    zil_val = float(losses['loss_zil'])
    parsed, _ = model.parse_losses(losses)
    parsed.backward()
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or
                                       not torch.isfinite(p.grad).all())]
    base_grad = [n for n, p in model.named_parameters()
                 if not p.requires_grad and p.grad is not None]
    report('3 勾配疎通', not no_grad and not base_grad,
           f'勾配なしRDB={no_grad or "なし"}, 勾配ありの凍結param='
           f'{base_grad or "なし"}')
    report('4 loss_zil', zil_val < 1e-4,
           f'loss_zil={zil_val:.3e} (初期状態でほぼ 0 のはず)')
    model.zero_grad(set_to_none=True)
    model.eval()

    # ---- 5+6. 融合の等価性と融合後状態 --------------------------------------
    # RDB に乱数を入れ、全和 forward と merge 後 forward を比較する
    with torch.no_grad():
        for n, p in model.named_parameters():
            if 'hlrb' in n or 'llrb' in n:
                p.normal_(0, 0.02)
            elif 'scaling' in n:
                p.fill_(0.37)
    # 融合の等価性はモジュール単位の線形性なので、RDB を含む 2 モジュール
    # (text_feat_map, neck) の出力テンソルを直接比較する。最終検出リストの
    # 比較は、同点スコア候補の順位交替で別物体同士を比べてしまうため使わない。
    x_text = torch.randn(2, 20, 768, device=DEVICE)
    feats = [torch.randn(1, c, s, s, device=DEVICE)
             for c, s in ((192, 100), (384, 50), (768, 25))]
    with torch.no_grad():
        out_text_pre = model.text_feat_map(x_text)
        out_neck_pre = model.neck(tuple(feats))

    sd_pre = {k: v.detach().cpu().clone()
              for k, v in model.state_dict().items()}
    sd_post = copy.deepcopy(sd_pre)
    n_merged = merge_state_dict(sd_post)
    model.load_state_dict(sd_post)
    with torch.no_grad():
        out_text_post = model.text_feat_map(x_text)
        out_neck_post = model.neck(tuple(feats))
    # float32 の結合順序差による丸めが残るため、出力スケールに対する相対誤差で
    # 判定する (融合は W_llrb + s*W_hlrb の再結合なので厳密なビット一致はしない)
    d_text = ((out_text_pre - out_text_post).abs().max() /
              out_text_pre.abs().max()).item()
    d_neck = max(((a - b).abs().max() / a.abs().max()).item()
                 for a, b in zip(out_neck_pre, out_neck_post))
    report('5 融合等価性', n_merged == 5 and d_text < 1e-4 and d_neck < 1e-4,
           f'merged={n_merged} (期待5), text 出力最大相対差={d_text:.3e}, '
           f'neck 出力最大相対差={d_neck:.3e}')

    ok6, detail6 = True, []
    for k in sd_post:
        if k.endswith('.hlrb.weight') or k.endswith('.hlrb.bias'):
            if not torch.allclose(sd_post[k],
                                  torch.full_like(sd_post[k], HLRB_INIT)):
                ok6 = False
                detail6.append(f'{k} != {HLRB_INIT}')
        elif k.endswith('.scaling'):
            if not torch.allclose(sd_post[k],
                                  torch.full_like(sd_post[k], SCALING_INIT)):
                ok6 = False
                detail6.append(f'{k} != {SCALING_INIT}')
        elif k.endswith('.llrb.weight'):
            prefix = k[:-len('.llrb.weight')]
            expect = (sd_pre[k] +
                      sd_pre[prefix + '.scaling'] * sd_pre[prefix +
                                                           '.hlrb.weight'])
            if not torch.allclose(sd_post[k], expect, atol=1e-6):
                ok6 = False
                detail6.append(f'{k} 融合値不一致')
    report('6 融合後状態', ok6, '; '.join(detail6) or
           'LLRB=融合値 / HLRB=1e-8 / s=0.1 を確認')

    print('\n===== 結果 =====')
    n_ok = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'OK' if ok else 'NG'}  {name}")
    print(f'{n_ok}/{len(results)} passed')
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == '__main__':
    main()
