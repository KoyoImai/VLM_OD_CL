#!/usr/bin/env python3
"""exp_037: DitHub / ODinW-13 の実行前検証（design.md §6.3 の 7 項目）.

  1. 単一クラスタスクで warmup を使わないこと（forward と enable_per_class の両方）
  2. 多クラスタスクでは従来どおり warmup を使うこと
  3. 43 クラスの評価モデルがライブラリ ckpt のクラスだけを合成に使うこと
  4. dn 無効化（実データで model.loss() を通し dn_loss_* が出ないこと）
  5. θ0 ロードで形状不一致・想定外の欠落キーが無いこと
  6. 13 タスクの config が公式のデータ版・split を指し、実ファイルが存在すること
  7. 式3の trained_classes が逐次順から決まる重複と一致すること
  8. optimizer が LoRA だけを、base lr / wd で学習すること

使い方（リポジトリルートで）:
    python experiments/exp_037/check_dithub_odinw13_setup.py
"""
import glob
import os
import sys

import torch
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import _load_checkpoint, load_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'exp_034'))
from odinw_official_tasks import ODINW13, task_order  # noqa: E402

CFG_DIR = 'experiments/exp_037/configs'
SEED = 42
ORDER = task_order(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
results = []


def report(name, ok, detail):
    results.append((name, ok))
    print(f"[{'OK' if ok else 'NG'}] {name}: {detail}")


def build(path):
    cfg = Config.fromfile(path)
    init_default_scope('mmdet')
    from mmdet.registry import MODELS
    model = MODELS.build(cfg.model)
    model.init_weights()
    return cfg, model


def dithub_linears(model):
    from mmdet.models.layers.dithub_layers import DitHubLinear
    return [m for m in model.modules() if isinstance(m, DitHubLinear)]


class FakeRunner:
    def __init__(self, model, it=0, epoch=0):
        self.model, self.iter, self.epoch = model, it, epoch
        self.logger = MMLogger.get_instance('exp037_check')


def main():  # noqa: C901
    torch.manual_seed(0)
    from mmdet.models.layers.dithub_layers import STATE, canonical_key
    from mmdet.engine.hooks.dithub_seq_phase_hook import DitHubSeqPhaseHook

    # ---- 6. データ版と実ファイルの存在 ----
    missing, cfg_mismatch = [], []
    for name in ORDER:
        t = ODINW13[name]
        c = Config.fromfile(f'{CFG_DIR}/dithub_odinw13_{name}.py')
        ds = c.train_dataloader.dataset
        want_root = os.path.join('data/odinw', t['root'])
        if os.path.normpath(ds.data_root) != os.path.normpath(want_root) \
                or ds.ann_file != t['train_ann']:
            cfg_mismatch.append(name)
        for p in (os.path.join(want_root, t['train_ann']),
                  os.path.join(want_root, t['eval_ann'])):
            if not os.path.exists(p):
                missing.append(p)
    report('6 データ版と実ファイル', not missing and not cfg_mismatch,
           f'13タスクの train/eval アノテーション欠落={len(missing)}, '
           f'config とテーブルの不一致={cfg_mismatch or "なし"}')

    # ---- 7. 式3の trained_classes ----
    seen, bad = set(), []
    for i, name in enumerate(ORDER, 1):
        classes = ODINW13[name]['classes']
        want = [c for c in classes if c.lower() in seen]
        c = Config.fromfile(f'{CFG_DIR}/dithub_odinw13_{name}.py')
        got = list(c.custom_hooks[0].trained_classes)
        if got != want:
            bad.append(f't={i} {name}: got={got} want={want}')
        seen.update(x.lower() for x in classes)
    report('7 式3の trained_classes', not bad,
           f'不一致={bad or "なし"}（t=5 Bus/Car, t=7 dog/person, '
           f't=10 CoW, t=11 boat/car が発火する）')

    # ---- 1/2. 単一クラス・多クラスの warmup 条件 ----
    single = [n for n in ORDER if len(ODINW13[n]['classes']) == 1]
    multi = [n for n in ORDER if len(ODINW13[n]['classes']) > 1]
    cfg_s, model_s = build(f'{CFG_DIR}/dithub_odinw13_{single[0]}.py')
    lins = dithub_linears(model_s)
    lin = lins[0]
    key = list(lin.per_class_lora_A.keys())[0]
    with torch.no_grad():   # warmup A と per-class A を区別できる値にする
        lin.warmup_lora_a.fill_(0.01)
        lin.per_class_lora_A[key].fill_(0.02)
        lin.shared_lora_b.fill_(0.01)
    model_s.train()
    DitHubSeqPhaseHook(warmup_iters=1500).before_train(FakeRunner(model_s))
    STATE.train_class_keys = [key, key]
    x = torch.ones(2, 3, lin.in_features)
    base = torch.nn.functional.linear(x, lin.weight, lin.bias)
    d_wu = lin._delta_single(x, lin.warmup_lora_a)
    d_pc = lin._delta_per_sample(x, [key, key])
    out = lin(x)
    e_pc = (out - (base + d_pc)).abs().max().item()
    e_wu = (out - (base + d_wu)).abs().max().item()
    ok1 = e_pc < 1e-6 and e_wu > 1e-6
    report('1 単一クラスタスクの forward', ok1,
           f'{single[0]}（1クラス）warmup 期の出力と per-class 経路の差={e_pc:.3e}'
           f'（0 が期待）, warmup 経路との差={e_wu:.3e}（>0 が期待）'
           f'→ per-class を使用（公式 do_warmup_a の条件）')

    before = lin.per_class_lora_A[key].detach().clone()
    model_s.enable_per_class(trained_keys=set())
    ok1b = torch.equal(lin.per_class_lora_A[key].detach(), before)
    report('1b 単一クラスの enable_per_class', ok1b,
           f'未学習の単一クラスで A が warmup で上書きされない={ok1b}'
           f'（公式は counter>0 のときだけ式3、それ以外は触れない）')
    del model_s

    cfg_m, model_m = build(f'{CFG_DIR}/dithub_odinw13_{multi[0]}.py')
    lm = dithub_linears(model_m)[0]
    k2 = list(lm.per_class_lora_A.keys())[:2]
    with torch.no_grad():
        lm.warmup_lora_a.fill_(0.01)
        for k in lm.per_class_lora_A:
            lm.per_class_lora_A[k].fill_(0.02)
        lm.shared_lora_b.fill_(0.01)
    model_m.train()
    DitHubSeqPhaseHook(warmup_iters=1500).before_train(FakeRunner(model_m))
    STATE.train_class_keys = k2
    x2 = torch.ones(2, 3, lm.in_features)
    base2 = torch.nn.functional.linear(x2, lm.weight, lm.bias)
    out2 = lm(x2)
    e_wu2 = (out2 - (base2 + lm._delta_single(x2, lm.warmup_lora_a)))\
        .abs().max().item()
    e_pc2 = (out2 - (base2 + lm._delta_per_sample(x2, k2))).abs().max().item()
    ok2 = e_wu2 < 1e-6 and e_pc2 > 1e-6
    model_m.enable_per_class(trained_keys=set())
    copied = all(torch.equal(lm.per_class_lora_A[k].detach(),
                             lm.warmup_lora_a.detach())
                 for k in lm.per_class_lora_A)
    report('2 多クラスタスクの warmup', ok2 and copied,
           f'{multi[0]}（{len(cfg_m.class_name)}クラス）warmup 期の出力と '
           f'warmup 経路の差={e_wu2:.3e}（0 が期待）, per-class 経路との差='
           f'{e_pc2:.3e}（>0 が期待）, 切替で全クラスが warmup をコピー={copied}')
    del model_m

    # ---- 5/8. θ0 ロードと optimizer ----
    cfg, model = build(f'{CFG_DIR}/dithub_odinw13_{ORDER[0]}.py')
    ckpt = _load_checkpoint(cfg.load_from, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    msd = model.state_dict()
    shape_bad = [k for k in sd if k in msd and sd[k].shape != msd[k].shape]
    # DecomposedMHA は ckpt の in_proj_weight/bias・out_proj.* を linear_{q,k,v,o}
    # へ分解して読むので、素朴なキー比較では欠落に見える。実ロード後に数値で照合する。
    load_checkpoint(model, cfg.load_from, map_location='cpu', strict=False,
                    logger='current')
    mha_bad = []
    for name, mod in model.named_modules():
        if type(mod).__name__ != 'DecomposedMHA':
            continue
        w = sd[f'{name}.in_proj_weight']
        b = sd[f'{name}.in_proj_bias']
        qw, kw, vw = torch.chunk(w, 3, dim=0)
        qb, kb, vb = torch.chunk(b, 3, dim=0)
        pairs = [(mod.linear_q.weight, qw), (mod.linear_k.weight, kw),
                 (mod.linear_v.weight, vw), (mod.linear_q.bias, qb),
                 (mod.linear_k.bias, kb), (mod.linear_v.bias, vb),
                 (mod.linear_o.weight, sd[f'{name}.out_proj.weight']),
                 (mod.linear_o.bias, sd[f'{name}.out_proj.bias'])]
        if not all(torch.equal(a.detach().cpu(), e) for a, e in pairs):
            mha_bad.append(name)
    decomposed = {f'{n}.linear_{s}.{p}'
                  for n, m in model.named_modules()
                  if type(m).__name__ == 'DecomposedMHA'
                  for s in 'qkvo' for p in ('weight', 'bias')}
    missing_keys = [k for k in msd if k not in sd]
    unexpected = [k for k in sd if k not in msd]
    real_missing = [k for k in missing_keys
                    if 'lora_' not in k
                    and not k.endswith('dithub_per_class_enabled')
                    and k not in decomposed]
    report('5 θ0 ロード', not shape_bad and not real_missing and not mha_bad,
           f'形状不一致={len(shape_bad)}, 真に欠落したキー={real_missing or "なし"}, '
           f'attention 分解の復元に失敗した層={mha_bad or "なし"}（{len(decomposed)} '
           f'キーを ckpt の in_proj/out_proj から分解して復元）, '
           f'ckpt 側の余剰={len(unexpected)}, 欠落計={len(missing_keys)}'
           f'（内訳: LoRA と phase buffer {len(missing_keys) - len(decomposed)} '
           f'＋ 分解 {len(decomposed)}）')

    opt = Runner.build_optim_wrapper(
        type('R', (), {'model': model, 'logger': MMLogger.get_current_instance(),
                       'world_size': 1})(), cfg.optim_wrapper)
    groups = opt.optimizer.param_groups
    n_par = sum(len(g['params']) for g in groups)
    lrs = {round(g['lr'], 8) for g in groups}
    wds = {round(g['weight_decay'], 8) for g in groups}
    n_train = sum(1 for n, p in model.named_parameters() if p.requires_grad)
    n_lora = sum(1 for n, p in model.named_parameters() if 'lora_' in n)
    ok8 = (n_train == n_lora and lrs == {0.001} and wds == {0.01})
    report('8 optimizer の対象と lr / wd', ok8,
           f'学習対象={n_train}（lora_ を含むパラメータ={n_lora}）, '
           f'optimizer 登録={n_par}, lr={sorted(lrs)}, wd={sorted(wds)} '
           f'（期待 lr 0.001 / wd 0.01 の 1 グループ）')

    # ---- 4. dn 無効（実データで loss を通す）----
    model = model.to(DEVICE)
    loader = Runner.build_dataloader(cfg.train_dataloader)
    batch = next(iter(loader))
    data = model.data_preprocessor(batch, True)
    DitHubSeqPhaseHook(warmup_iters=1500).before_train(FakeRunner(model))
    model.train()
    losses = model.loss(data['inputs'], data['data_samples'])
    dn_keys = [k for k in losses if k.startswith('dn_')]
    report('4 dn 無効', not dn_keys,
           f'損失項={len(losses)} 個、dn_* の項={dn_keys or "なし"} '
           f'(dn 有効なら 18 項増えて 40 項になる)')
    del model

    # ---- 3. 評価モデルのライブラリ絞り込み ----
    cfg_e, model_e = build(f'{CFG_DIR}/dithub_odinw13_eval.py')
    le = dithub_linears(model_e)[0]
    all_keys = list(le.per_class_lora_A.keys())
    lib_keys = set(all_keys[:5])
    for m in dithub_linears(model_e):
        m.library_class_keys = set(lib_keys)
        m.dithub_per_class_enabled.fill_(1)
        with torch.no_grad():
            m.shared_lora_b.fill_(0.01)
    model_e.eval()
    xe = torch.ones(1, 3, le.in_features)
    basee = torch.nn.functional.linear(xe, le.weight, le.bias)
    STATE.eval_class_keys = all_keys[5:10]        # ライブラリ外だけ
    out_out = (le(xe) - basee).abs().max().item()
    STATE.eval_class_keys = all_keys[:5]          # ライブラリ内だけ
    out_in = (le(xe) - basee).abs().max().item()
    report('3 評価時のライブラリ絞り込み', out_out == 0.0 and out_in > 0.0,
           f'ライブラリ外のクラスのみのプロンプト → 出力変化={out_out:.3e}（0 が期待）, '
           f'ライブラリ内のクラス → {out_in:.3e}（>0 が期待）。'
           f'スロット総数={len(all_keys)}')

    print('\n===== 結果 =====')
    for name, ok in sorted(results):
        print(f"  {'OK' if ok else 'NG'}  {name}")
    n_ok = sum(ok for _, ok in results)
    print(f'{n_ok}/{len(results)} passed')
    return 0 if n_ok == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
