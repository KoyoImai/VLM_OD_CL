#!/usr/bin/env python3
"""exp_021: DitHub のタスク境界処理 (式4 の B 融合 + ライブラリ継承).

公式実装の end_task に対応する処理を checkpoint に対して行う:
    B_next = (1 - λ_B) * B_prev + λ_B * B_opt      (層ごと、λ_B = 0.7)
    ライブラリ = 前タスクまでの per_class_lora_A ∪ 現タスクの per_class_lora_A
                 (同名クラスは現タスク側で上書き)

B_prev は「現タスクの load_from に渡した checkpoint 内の B」であり、公式が
TaskMemory に持たせるスナップショットと同一。前 checkpoint に LoRA キーが
無い場合 (事前学習 θ0 = 最初のタスク) は公式同様に式4をスキップする。

exp_021 (単発ドメイン学習) では実行経路に乗らない。将来の逐次実験用。
使い方:
    python experiments/exp_021/merge_dithub.py <prev.pth> <current.pth> <out.pth>
"""
import argparse

import torch

LAMBDA_B = 0.7


def merge_task_boundary(prev_sd: dict, curr_sd: dict,
                        lambda_b: float = LAMBDA_B):
    """(融合後 state_dict, 融合した B の数, 継承したクラス A の数) を返す."""
    out = dict(curr_sd)
    prev_has_lora = any(k.endswith('.shared_lora_b') for k in prev_sd)
    n_b = 0
    if prev_has_lora:
        for k in curr_sd:
            if k.endswith('.shared_lora_b') and k in prev_sd:
                out[k] = (1.0 - lambda_b) * prev_sd[k] + lambda_b * curr_sd[k]
                n_b += 1
    n_inherit = 0
    for k in prev_sd:
        if '.per_class_lora_A.' in k and k not in out:
            out[k] = prev_sd[k]
            n_inherit += 1
    return out, n_b, n_inherit


def _load_sd(path):
    ckpt = torch.load(path, map_location='cpu')
    return ckpt, ckpt['state_dict'] if 'state_dict' in ckpt else ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prev', help='前タスクのライブラリ checkpoint (.pth)')
    parser.add_argument('current', help='現タスクの学習済み checkpoint (.pth)')
    parser.add_argument('output', help='次タスクの load_from 用の保存先 (.pth)')
    args = parser.parse_args()

    _, prev_sd = _load_sd(args.prev)
    curr_ckpt, curr_sd = _load_sd(args.current)
    merged, n_b, n_inherit = merge_task_boundary(prev_sd, curr_sd)
    if 'state_dict' in curr_ckpt:
        curr_ckpt['state_dict'] = merged
    else:
        curr_ckpt = merged
    torch.save(curr_ckpt, args.output)
    print(f'B merged: {n_b} layers (skip if 0 = first task), '
          f'inherited class-A tensors: {n_inherit} -> {args.output}')


if __name__ == '__main__':
    main()
