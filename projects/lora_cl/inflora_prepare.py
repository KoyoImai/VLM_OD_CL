#!/usr/bin/env python3
"""InfLoRA: タスク学習前の A（次元削減行列）の設計（exp_042 design.md §2.3）。

θ_{t-1}（前タスクまでの融合済み plain ckpt）でタスク t の訓練データを流して各挿入層の
入力共分散を集め、DualGPM のメモリ（t=1 は無し）で射影してから SVD の上位 r 主成分で
A を設計する。出力の設計ファイルを学習 config の model.inflora.design_path に渡す。

このとき LoRA の B は 0 なので、収集時の forward は θ_{t-1} と厳密に一致する。

使い方:
    python projects/lora_cl/inflora_prepare.py CONFIG CKPT --out DESIGN_OUT \\
        [--memory MEM_IN] [--max-samples 1000] [--seed 0]
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

from projects.lora_cl.dual_gpm import design_A, load_memory          # noqa: E402
from projects.lora_cl.inflora_collect import build_and_collect       # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('ckpt')
    p.add_argument('--out', required=True)
    p.add_argument('--memory', default=None,
                   help='DualGPM メモリ（t=1 は省略 = 生の共分散で設計）')
    p.add_argument('--max-samples', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--task-name', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    model, covs, n_used = build_and_collect(
        args.config, args.ckpt, args.max_samples, args.seed, args.device,
        design_path=None)
    from mmengine.config import Config
    r = int(Config.fromfile(args.config).model['lora']['r'])

    mem = load_memory(args.memory) if args.memory else None
    designs = {}
    n_deficient = 0
    for name, cov in covs.items():
        feature = ptype = None
        if mem is not None and name in mem['feature']:
            feature, ptype = mem['feature'][name], mem['ptype'][name]
        a = design_A(cov, r, feature, ptype)
        n_zero = int((np.abs(a).sum(axis=1) == 0).sum())
        if n_zero:
            n_deficient += 1
            print(f'[InfLoRA] ランク不足: {name} 有効 {r - n_zero}/{r} 行'
                  f'（残りは 0 行 = その成分の ΔW は 0）')
        designs[name] = a
    if n_deficient:
        print(f'[InfLoRA] ランク不足の層: {n_deficient}/{len(covs)}'
              '（テキスト側はプロンプトの多様性で決まる。dual_gpm.design_A の注記参照）')
    torch.save(
        {'lora_A': designs,
         'meta': dict(config=args.config, ckpt=args.ckpt, rank=r,
                      memory=args.memory, n_samples=n_used, seed=args.seed,
                      max_samples=args.max_samples,
                      task=args.task_name or args.config)},
        args.out)
    print(f'[InfLoRA] 設計 A を保存: {args.out} ({len(designs)} 層, r={r}, '
          f'メモリ={"あり" if mem else "なし(t=1)"})')


if __name__ == '__main__':
    main()
