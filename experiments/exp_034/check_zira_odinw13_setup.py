#!/usr/bin/env python3
"""exp_034: 学習開始前の実装検証（design.md §6.4 の5項目）.

1. 学習対象が RDB のみ（hlrb / llrb / scaling）
2. dn が無効（dn_loss_* が出ない = 損失22項）
3. LLRB の実効 lr が 2e-4、HLRB / scaling が 1e-3
4. loss_zil が計算され、optimizer step まで通る
5. Rep+（merge_hlrb.py）が forward を保存する

実データ（逐次順の先頭タスク）を1バッチ使って確認する。

使い方: python experiments/exp_034/check_zira_odinw13_setup.py [task]
"""
import os
import subprocess
import sys
import tempfile

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from odinw_official_tasks import task_order  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
OK, NG = [], []


def check(name, cond, detail=''):
    (OK if cond else NG).append(name)
    print(f'[{"OK" if cond else "NG"}] {name}' + (f'  {detail}' if detail else ''))


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else task_order(42)[0]
    print(f'=== exp_034 実装検証 / task = {task} ===')
    init_default_scope('mmdet')

    cfg = Config.fromfile(
        os.path.join(HERE, 'configs', f'zira_odinw13_{task}.py'))
    cfg.work_dir = tempfile.mkdtemp(prefix='exp034_check_')

    runner = Runner.from_cfg(cfg)
    model = runner.model
    model.init_weights()
    runner.load_or_resume()   # θ0 を読み込む
    model.train()

    # --- 1. 学習対象 ---
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    bad = [n for n in trainable
           if not any(k in n for k in ('hlrb', 'llrb', 'scaling'))]
    check('1. 学習対象が RDB のみ', not bad,
          f'{len(trainable)} params, 想定外={bad[:3]}')

    # --- 3. optimizer の param group ---
    # Runner は optim_wrapper を遅延構築するため、ここで明示的に組み立てる。
    runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
    optim = runner.optim_wrapper.optimizer
    lrs = {}
    name_of = {id(p): n for n, p in model.named_parameters()}
    for g in optim.param_groups:
        for p in g['params']:
            lrs[name_of[id(p)]] = g['lr']
    llrb = {v for k, v in lrs.items() if 'llrb' in k}
    other = {v for k, v in lrs.items() if 'llrb' not in k}
    check('3. LLRB の実効 lr が 2e-4', llrb == {2e-4}, f'llrb lr={llrb}')
    check('3. HLRB / scaling の lr が 1e-3', other == {1e-3}, f'other lr={other}')

    # --- 2 & 4. 実データ1バッチで損失と backward ---
    # Runner は dataloader も遅延構築する。ここで明示的に組み立てる。
    train_dl = Runner.build_dataloader(cfg.train_dataloader)
    print(f'  train dataset: {len(train_dl.dataset)} 枚')
    batch = next(iter(train_dl))
    data = model.data_preprocessor(batch, True)
    losses = model.loss(data['inputs'], data['data_samples'])
    keys = sorted(losses)
    dn_keys = [k for k in keys if 'dn_loss' in k]
    check('2. dn が無効（dn_loss_* が無い）', not dn_keys, f'損失 {len(keys)} 項')
    check('2. 損失が 22 項（検出21 + loss_zil）', len(keys) == 22, str(keys))
    check('4. loss_zil が計算される', 'loss_zil' in losses,
          f"loss_zil={float(losses.get('loss_zil', float('nan'))):.3e}")

    # update_params は step 後に zero_grad するため、backward と step を分けて確認する。
    before = {n: p.detach().clone()
              for n, p in model.named_parameters() if p.requires_grad}
    runner.optim_wrapper.backward(sum(v.sum() for v in losses.values()))
    grads = [n for n, p in model.named_parameters()
             if p.requires_grad and p.grad is not None]
    check('4. 全 RDB に勾配がつく', len(grads) == len(trainable),
          f'grad あり {len(grads)}/{len(trainable)}')
    runner.optim_wrapper.step()
    runner.optim_wrapper.zero_grad()
    moved = [n for n, p in model.named_parameters()
             if p.requires_grad and not torch.equal(before[n], p.detach())]
    check('4. optimizer step で全 RDB が更新される', len(moved) == len(trainable),
          f'更新 {len(moved)}/{len(trainable)}')

    # --- 5. Rep+ が forward を保存する ---
    model.eval()
    with torch.no_grad():
        img = data['inputs'][:1]
        feats_before = model.extract_feat(img)
    ck_in = os.path.join(cfg.work_dir, 'before.pth')
    ck_out = os.path.join(cfg.work_dir, 'after.pth')
    torch.save({'state_dict': model.state_dict()}, ck_in)
    subprocess.run(
        [sys.executable, os.path.join(ROOT, 'experiments/exp_020/merge_hlrb.py'),
         ck_in, ck_out], check=True, capture_output=True)
    sd = torch.load(ck_out, map_location='cpu')['state_dict']
    model.load_state_dict(sd)
    with torch.no_grad():
        feats_after = model.extract_feat(img)
    diff = max(float((a - b).abs().max())
               for a, b in zip(feats_before, feats_after))
    scale = max(float(a.abs().max()) for a in feats_before)
    check('5. Rep+ が forward を保存する', diff / scale < 1e-4,
          f'最大差 {diff:.3e} / 特徴の最大値 {scale:.3e}')

    print(f'\n=== OK {len(OK)} 件 / NG {len(NG)} 件 ===')
    if NG:
        print('NG:', NG)
        sys.exit(1)


if __name__ == '__main__':
    main()
