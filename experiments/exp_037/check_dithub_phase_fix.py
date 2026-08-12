#!/usr/bin/env python3
"""exp_037: DitHub のフェーズ引き継ぎバグの再現と修正の検証.

背景 (2026-08-10 に exp_023 のログ・ckpt から特定):
  `DitHubLinear.dithub_per_class_enabled` は persistent buffer なので
  checkpoint に保存される。逐次学習で前タスクのライブラリを load_from すると
  1 が読み込まれ、そのタスクは warmup を一度も実行しないまま specialization
  から始まる。warmup_lora_a は勾配を受けないまま、iter=max_iters/2 の切替で
  全クラスの A を上書きし、それまでの学習を破棄する。

  実測の裏づけ:
    - lib_after_underwater.pth の 109 層すべてで enabled=1
    - warmup_lora_a が t=1 ckpt と t=2 ckpt で 109 層ビット一致 (勾配ゼロ)
    - 切替直後の損失: t=1 10.60 -> 12.53 / t=2 10.12 -> 21.32 / t=3 6.94 -> 18.80

修正: DitHubPhaseHook.before_train で進捗からフェーズを決め直す
  (新規タスク -> WARMUP、途中再開 -> 進捗どおり)。

使い方 (リポジトリルートで):
    python experiments/exp_037/check_dithub_phase_fix.py
"""
import sys

import torch
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

CFG = 'experiments/exp_023/configs/dithub_replayfree_electromagnetic.py'
PREV_LIB = ('experiments/exp_023/dithub_replayfree_library/'
            'lib_after_underwater.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WARMUP_ITERS = 1500

results = []


def report(name, ok, detail):
    results.append((name, ok))
    print(f"[{'OK' if ok else 'NG'}] {name}: {detail}")


class FakeRunner:
    """フックが参照する属性だけを持つ最小の runner."""

    def __init__(self, model, it, epoch=0):
        self.model = model
        self.iter = it
        self.epoch = epoch
        self.logger = MMLogger.get_instance('dithub_phase_check')


def dithub_linears(model):
    from mmdet.models.layers.dithub_layers import DitHubLinear
    return [m for m in model.modules() if isinstance(m, DitHubLinear)]


def enabled_values(model):
    return {int(m.dithub_per_class_enabled) for m in dithub_linears(model)}


def build_model(cfg):
    init_default_scope('mmdet')
    from mmdet.registry import MODELS
    model = MODELS.build(cfg.model)
    model.init_weights()
    return model


def one_real_iter(model, data_batch, per_class_phase):
    """実データで 1 iteration 回し、勾配の流れ先を返す."""
    model.train()
    model.zero_grad(set_to_none=True)
    data = model.data_preprocessor(data_batch, True)
    losses = model.loss(data['inputs'], data['data_samples'])
    total = sum(v.sum() for v in losses.values()
                if isinstance(v, torch.Tensor))
    total.backward()
    warm = sum(1 for m in dithub_linears(model)
               if m.warmup_lora_a.grad is not None
               and m.warmup_lora_a.grad.abs().sum() > 0)
    b = sum(1 for m in dithub_linears(model)
            if m.shared_lora_b.grad is not None
            and m.shared_lora_b.grad.abs().sum() > 0)
    per = 0
    for m in dithub_linears(model):
        for p in m.per_class_lora_A.values():
            if p.grad is not None and p.grad.abs().sum() > 0:
                per += 1
    print(f'    [{per_class_phase}] grad!=0 -> warmup_lora_a: {warm} 層 / '
          f'shared_lora_b: {b} 層 / per_class_A: {per} テンソル')
    return warm, b, per


def main():
    torch.manual_seed(0)
    cfg = Config.fromfile(CFG)

    # ---- 1. 前タスクのライブラリが enabled=1 を持っていること (バグの原因) ----
    sd = torch.load(PREV_LIB, map_location='cpu')
    sd = sd.get('state_dict', sd)
    en_keys = [k for k in sd if k.endswith('dithub_per_class_enabled')]
    vals = {int(sd[k]) for k in en_keys}
    report('1 ライブラリ ckpt のフェーズフラグ', vals == {1},
           f'{len(en_keys)} 層すべて enabled={vals} '
           f'(=1 なら load_from で specialization が持ち込まれる)')

    # ---- 2. load_from 直後は enabled=1 になる (修正前の挙動を再現) ----
    model = build_model(cfg)
    before = enabled_values(model)
    load_checkpoint(model, PREV_LIB, map_location='cpu', strict=False,
                    logger='current')
    after = enabled_values(model)
    report('2 load_from によるフェーズ混入の再現',
           before == {0} and after == {1},
           f'build 直後={before} -> ライブラリ読込後={after} '
           f'(修正前はこの状態で warmup を飛ばして学習が始まっていた)')

    # ---- 3. 修正: before_train が新規タスクを WARMUP に戻す ----
    from mmdet.engine.hooks.dithub_phase_hook import DitHubPhaseHook
    hook = DitHubPhaseHook(warmup_iters=WARMUP_ITERS)
    hook.before_train(FakeRunner(model, it=0))
    fresh = enabled_values(model)
    report('3 修正: 新規タスク開始時に WARMUP へ戻る', fresh == {0},
           f'before_train(iter=0) 後 enabled={fresh} (期待 {{0}})')

    # ---- 4. 修正: 途中再開では specialization を引き継ぐ ----
    hook.before_train(FakeRunner(model, it=WARMUP_ITERS + 700))
    resumed = enabled_values(model)
    hook.before_train(FakeRunner(model, it=WARMUP_ITERS - 1))
    still_warm = enabled_values(model)
    report('4 修正: 途中再開時のフェーズ引き継ぎ',
           resumed == {1} and still_warm == {0},
           f'iter={WARMUP_ITERS + 700} -> {resumed} (期待 {{1}}), '
           f'iter={WARMUP_ITERS - 1} -> {still_warm} (期待 {{0}})')

    # ---- 4.5 warmup_lora_a がタスク開始時に引き直されること ----
    lib_w = {k: v for k, v in sd.items() if k.endswith('.warmup_lora_a')}
    load_checkpoint(model, PREV_LIB, map_location='cpu', strict=False,
                    logger='current')
    lins = dithub_linears(model)
    hook.before_train(FakeRunner(model, it=0))
    changed = 0
    for name, m in model.named_modules():
        key = f'{name}.warmup_lora_a'
        if key in lib_w and not torch.equal(
                m.warmup_lora_a.detach().cpu(), lib_w[key]):
            changed += 1
    # 途中再開 (iter>0) では引き直さない
    load_checkpoint(model, PREV_LIB, map_location='cpu', strict=False,
                    logger='current')
    hook.before_train(FakeRunner(model, it=WARMUP_ITERS + 700))
    kept = 0
    for name, m in model.named_modules():
        key = f'{name}.warmup_lora_a'
        if key in lib_w and torch.equal(
                m.warmup_lora_a.detach().cpu(), lib_w[key]):
            kept += 1
    report('4.5 warmup_lora_a のタスク毎の引き直し',
           changed == len(lins) and kept == len(lins),
           f'iter=0 で引き直された層={changed}/{len(lins)}, '
           f'途中再開で保持された層={kept}/{len(lins)} '
           f'(公式はタスク毎にモデルを作り直すため warmup_lora_a は復元されない)')

    # ---- 5. 実データで勾配の流れ先を確認 ----
    model = model.to(DEVICE)
    loader = Runner.build_dataloader(cfg.train_dataloader)
    data_batch = next(iter(loader))

    print('  --- 修正後 (warmup フェーズ) ---')
    hook.before_train(FakeRunner(model, it=0))
    w_warm, b_warm, p_warm = one_real_iter(model, data_batch, 'warmup')
    ok5 = w_warm == len(dithub_linears(model)) and p_warm == 0
    report('5 warmup 期の勾配疎通', ok5,
           f'warmup_lora_a に勾配が流れる層={w_warm}/'
           f'{len(dithub_linears(model))}, per_class_A へは {p_warm} '
           f'(期待: 全層 / 0)')

    # ---- 6. 修正前の経路 (enabled=1 のまま学習開始) の再現 ----
    print('  --- 修正前の再現 (ライブラリの enabled=1 のまま学習開始) ---')
    for m in dithub_linears(model):
        m.dithub_per_class_enabled.fill_(1)
    w_bug, b_bug, p_bug = one_real_iter(model, data_batch, 'buggy')
    report('6 バグの再現: warmup_lora_a が学習されない', w_bug == 0,
           f'warmup_lora_a に勾配が流れる層={w_bug} (期待 0)。'
           f'この状態が 1500 iter 続いた後、切替で全 A が未学習の '
           f'warmup_lora_a に置き換えられていた')

    # ---- 7. 切替の連続性: A <- warmup_lora_a のコピーで出力が変わらないこと ----
    hook.before_train(FakeRunner(model, it=0))
    with torch.no_grad():
        for m in dithub_linears(model):     # warmup が進んだ状態を模擬
            m.warmup_lora_a.normal_(0, 0.02)
            m.shared_lora_b.normal_(0, 0.02)
    from mmdet.models.layers.dithub_layers import STATE
    lin = dithub_linears(model)[0]
    probe = torch.randn(2, 5, lin.in_features, device=DEVICE)
    STATE.train_class_keys = list(lin.per_class_lora_A.keys())[:2]
    model.train()
    with torch.no_grad():
        out_warm = lin(probe)
        model.enable_per_class(trained_keys=set())
        out_spec = lin(probe)
    diff = (out_warm - out_spec).abs().max().item()
    report('7 フェーズ切替の連続性', diff < 1e-5,
           f'切替前後の同一入力に対する最大差={diff:.3e} '
           f'(A <- warmup コピーなので 0 のはず)')

    print('\n===== 結果 =====')
    for name, ok in results:
        print(f"  {'OK' if ok else 'NG'}  {name}")
    n_ok = sum(ok for _, ok in results)
    print(f'{n_ok}/{len(results)} passed')
    return 0 if n_ok == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
