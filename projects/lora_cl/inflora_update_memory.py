#!/usr/bin/env python3
"""InfLoRA: タスク学習後の DualGPM メモリ更新（exp_042 design.md §2.3）。

学習済み ckpt（LoRA 分岐込みの work_dir ckpt。公式も学習後のネットワークで再収集する）で
入力共分散を集め、DualGPM のメモリを更新する。閾値は公式どおり
threshold = lamb + (lame − lamb)·task_index/total の線形増加（task_index は 0 始まり）。

使い方:
    python projects/lora_cl/inflora_update_memory.py CONFIG CKPT --out MEM_OUT \\
        --task-index 0 --total 13 [--memory MEM_IN] [--lamb 0.95] [--lame 1.0] \\
        [--max-samples 1000] [--seed 0]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

from projects.lora_cl.dual_gpm import (load_memory, new_empty_memory,   # noqa: E402
                                       save_memory, threshold_at,
                                       update_dual_gpm)
from projects.lora_cl.inflora_collect import build_and_collect          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('ckpt')
    p.add_argument('--out', required=True)
    p.add_argument('--memory', default=None, help='前タスクまでのメモリ（t=1 は省略）')
    p.add_argument('--task-index', type=int, required=True, help='0 始まりのタスク番号')
    p.add_argument('--total', type=int, default=13)
    p.add_argument('--lamb', type=float, default=0.95)
    p.add_argument('--lame', type=float, default=1.0)
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
    mem = load_memory(args.memory) if args.memory else new_empty_memory()
    thr = threshold_at(args.task_index, args.total, args.lamb, args.lame)
    print(f'[InfLoRA] DualGPM 更新: threshold={thr:.4f} '
          f'(t={args.task_index}/{args.total}, lamb={args.lamb}, lame={args.lame})')
    update_dual_gpm(covs, mem, thr)
    mem['meta']['num_tasks'] += 1
    mem['meta']['history'].append(
        dict(task=args.task_name or args.config, ckpt=args.ckpt,
             threshold=thr, n_samples=n_used, seed=args.seed))
    save_memory(mem, args.out)
    sizes = {n: (f.shape[1], mem['ptype'][n]) for n, f in mem['feature'].items()}
    n_rm = sum(1 for _, t in sizes.values() if t == 'remove')
    print(f'[InfLoRA] メモリ保存: {args.out} ({len(sizes)} 層, '
          f'remove {n_rm} / retain {len(sizes) - n_rm})')


if __name__ == '__main__':
    main()
