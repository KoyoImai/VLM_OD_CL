#!/usr/bin/env python3
"""タスク終了時の対角・経験 Fisher を推定し、EWC の 2 バッファ状態を更新する。

exp_042 design.md §2.2 の仕様:
    - 対角・経験 Fisher: タスク学習終了時のパラメータ θ*_t で、そのタスクの訓練データに
      対する学習と同じ検出損失の勾配²を平均する（Avalanche / GEM と同方式）
    - batch=1（バッチ内の勾配相殺による過小推定を避ける。GMvandeVen 既定と同じ）
    - サンプル数は上限枚数を指定。訓練枚数が上限以下なら全数、超えるなら seed 固定の
      無作為抽出（2026-08-15 確定。上限の既定 1000）
    - dropout は無効化して推定する（model.eval()。Avalanche / GMvandeVen が eval で
      推定するのに対応。損失は model.loss() で明示的に計算するので eval でも流れる）
    - Fisher の勾配に EWC ペナルティは含めない（タスクの損失のみ。ewc.lam=0 で build）

使い方:
    python projects/ewc_cl/estimate_fisher.py CONFIG CKPT --out STATE_OUT \\
        [--prev-state STATE_IN] [--max-samples 1000] [--seed 0]

    CONFIG      : そのタスクの学習 config（EWCGroundingDINO・train_dataloader を含む）
    CKPT        : タスク終了時の checkpoint（θ*_t）
    --prev-state: 前タスクまでの 2 バッファ（t=1 は省略）
    --out       : 更新後の 2 バッファの保存先
"""
import argparse
import os
import random
import sys

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.utils import import_modules_from_strings

from mmdet.registry import DATASETS, MODELS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

from projects.ewc_cl.ewc_state import (new_empty_state, load_state,   # noqa: E402
                                       save_state, update_state,
                                       target_param_names)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('ckpt')
    p.add_argument('--out', required=True)
    p.add_argument('--prev-state', default=None)
    p.add_argument('--max-samples', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--task-name', default=None,
                   help='状態の履歴に記録するタスク名（省略時は config 名）')
    return p.parse_args()


def seed_everything(seed):
    """全 RNG を固定する（再現性）。

    サンプル選択に加え、データパイプラインの拡張（RandomFlip 等）がグローバル RNG を
    使うため、固定しないと同じ引数でも Fisher が実行ごとに変わる（2026-08-15 監査で
    追加）。CUDA カーネルの原子加算による ~1e-7 相対の非決定性は残る。
    """
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    seed_everything(args.seed)
    cfg = Config.fromfile(args.config)
    init_default_scope('mmdet')
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg['custom_imports'])

    # Fisher はタスク損失の勾配だけから作る。ペナルティ・状態は無効化して build する。
    assert cfg.model['type'] == 'EWCGroundingDINO', \
        f'EWCGroundingDINO の config を渡すこと: {cfg.model["type"]}'
    cfg.model['ewc']['lam'] = 0.0
    cfg.model['ewc']['state_path'] = None

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.ckpt, map_location='cpu')
    model = model.to(args.device)
    # dropout 無効で推定する（Avalanche / GMvandeVen が eval で推定するのに対応）。
    # model.eval() で全 dropout（モジュール型に加え、DecomposedMHA・融合層などの
    # F.dropout(training=self.training) を使う関数型も）を確実に切り、loss 経路の
    # 構造分岐（forward_transformer の学習時専用出力）に使われる検出器本体の
    # training フラグだけを立て直す。Dropout/DropPath の列挙では関数型を取りこぼし、
    # loss が呼び出しごとに 3〜5% 揺れることを実測した（2026-08-15）。
    # この状態で loss はビット単位で決定的になる（勾配計算は eval でも通る）。
    model.eval()
    model.training = True
    print('[Fisher] dropout 無効（eval + 検出器 training フラグのみ有効）')

    targets = target_param_names(model, model.ewc_target_components)
    target_set = set(targets)

    # 現在ドメインの訓練データ。リプレイ等は使わない（このタスクの Fisher なので、
    # ODinW-13 の各タスク config は現在タスクのデータのみを持つ）。
    dataset = DATASETS.build(cfg.train_dataloader['dataset'])
    n = len(dataset)
    if n > args.max_samples:
        idx = sorted(random.Random(args.seed).sample(range(n), args.max_samples))
    else:
        idx = list(range(n))
    print(f'[Fisher] データ {n} 枚中 {len(idx)} 枚で推定 '
          f'(max={args.max_samples}, seed={args.seed})')

    fisher = {name: torch.zeros_like(p, device=args.device)
              for name, p in model.named_parameters() if name in target_set}

    n_used = 0
    for j, i in enumerate(idx):
        batch = pseudo_collate([dataset[i]])
        data = model.data_preprocessor(batch, True)
        losses = model.loss(data['inputs'], data['data_samples'])
        total = sum(v.sum() for v in losses.values()
                    if isinstance(v, torch.Tensor) and v.requires_grad)
        model.zero_grad(set_to_none=True)
        total.backward()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in target_set and p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2
        n_used += 1
        if (j + 1) % 100 == 0:
            print(f'[Fisher] {j + 1}/{len(idx)}')
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        for name in fisher:
            fisher[name] /= float(n_used)

    # 健全性: 非負・有限
    bad = [k for k, v in fisher.items()
           if not torch.isfinite(v).all() or (v < 0).any()]
    assert not bad, f'Fisher が非負・有限でない: {bad[:3]}'

    theta_star = dict(model.named_parameters())
    state = load_state(args.prev_state) if args.prev_state else new_empty_state()
    update_state(
        state, fisher, theta_star,
        task_meta=dict(task=args.task_name or args.config,
                       ckpt=args.ckpt, n_samples=n_used, seed=args.seed,
                       max_samples=args.max_samples))
    save_state(state, args.out)

    nz = sum(int((v > 0).sum()) for v in fisher.values())
    tot = sum(v.numel() for v in fisher.values())
    print(f'[Fisher] 保存: {args.out} (num_tasks={state["meta"]["num_tasks"]}, '
          f'対象 {len(fisher)} params, 非零 {nz:,}/{tot:,})')


if __name__ == '__main__':
    main()
