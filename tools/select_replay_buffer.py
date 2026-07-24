# Copyright (c) OpenMMLab. All rights reserved.
"""Randomly select a fixed subset of an ODVG (JSONL) file for a replay buffer.

汎用ツール（実験横断で再利用可）。ODVG 形式（1行=1画像）のファイルから、
--seed で固定したランダムサンプリングで --num 件を選び、
(1) 選ばれた分だけの ODVG サブセット（学習が読む実体。train_od.json と同形式）
(2) manifest（seed・件数・元ファイル・選択画像ファイル名一覧。確認/再現用）
を書き出す。同じ seed なら毎回同じ選択になる（再現性）。

Example:
  python tools/select_replay_buffer.py \
      --input /path/objects365_train_od.json \
      --num 1000 --seed 0 \
      --output experiments/exp_023/buffer/reference_o365v1_1000.odvg.json
"""
import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description='Select a random fixed subset of an ODVG (JSONL) file '
        'for a replay buffer.')
    parser.add_argument(
        '--input', required=True,
        help='source ODVG JSONL file (e.g. *_train_od.json)')
    parser.add_argument(
        '--num', type=int, required=True, help='number of entries to select')
    parser.add_argument(
        '--seed', type=int, required=True,
        help='random seed for reproducible selection')
    parser.add_argument(
        '--output', required=True, help='output ODVG JSONL subset path')
    parser.add_argument(
        '--manifest', default=None,
        help='manifest json path (default: <output>.manifest.json)')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1st pass: 有効行数を数える（大きいファイルでも全ロードしない）
    with open(args.input) as f:
        n = sum(1 for line in f if line.strip())
    if args.num > n:
        raise ValueError(
            f'--num ({args.num}) > entries in input ({n})')

    # seed 固定でインデックスをサンプリング（元順を保つため sorted）
    rng = random.Random(args.seed)
    idx = set(rng.sample(range(n), args.num))

    # 2nd pass: 選ばれた行だけを保持（メモリは k 件分のみ）
    selected = []
    with open(args.input) as f:
        i = 0
        for line in f:
            if not line.strip():
                continue
            if i in idx:
                selected.append(json.loads(line))
            i += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        for e in selected:
            f.write(json.dumps(e) + '\n')

    manifest_path = args.manifest or (args.output + '.manifest.json')
    manifest = {
        'source': os.path.abspath(args.input),
        'output': os.path.abspath(args.output),
        'num_source': n,
        'num_selected': len(selected),
        'seed': args.seed,
        'selected_filenames': [e['filename'] for e in selected],
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'selected {len(selected)}/{n} entries (seed={args.seed})')
    print(f'  ODVG subset -> {args.output}')
    print(f'  manifest    -> {manifest_path}')


if __name__ == '__main__':
    main()
