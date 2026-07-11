#!/usr/bin/env python3
"""exp_020: ZiRa の Rep+ (タスク間融合) を checkpoint に適用するスクリプト.

公式実装の __rep__ と同一の処理を state_dict に対して行う:
    W_llrb <- W_llrb + s * W_hlrb   (bias も同様)
    W_hlrb <- 1e-8,  s <- 0.1

用途: 逐次学習でドメイン t の checkpoint をドメイン t+1 へ渡す前に一度かける。
本プロジェクトの best checkpoint は融合前の状態で保存されるため、これを通さずに
次ドメインの load_from に渡すと公式実装と挙動がずれる (implementation_plan.md 3.5)。
単発ドメイン学習 (exp_020 本体) では評価が全和 forward のため本スクリプトは不要。

使い方:
    python experiments/exp_020/merge_hlrb.py <in.pth> <out.pth>
"""
import argparse

import torch

HLRB_INIT = 1e-8
SCALING_INIT = 0.1


def merge_state_dict(state_dict: dict) -> int:
    """RDB を LLRB へ融合し HLRB / scaling を再初期化する。融合した RDB 数を返す."""
    n_merged = 0
    for key in list(state_dict.keys()):
        if not key.endswith('.hlrb.weight'):
            continue
        prefix = key[:-len('.hlrb.weight')]
        scaling = state_dict[prefix + '.scaling']
        state_dict[prefix + '.llrb.weight'] = (
            state_dict[prefix + '.llrb.weight'] +
            scaling.to(state_dict[key].dtype) * state_dict[key])
        bias_key = prefix + '.hlrb.bias'
        if bias_key in state_dict:
            state_dict[prefix + '.llrb.bias'] = (
                state_dict[prefix + '.llrb.bias'] +
                scaling.squeeze().to(state_dict[bias_key].dtype) *
                state_dict[bias_key])
            state_dict[bias_key] = torch.full_like(state_dict[bias_key],
                                                   HLRB_INIT)
        state_dict[key] = torch.full_like(state_dict[key], HLRB_INIT)
        state_dict[prefix + '.scaling'] = torch.full_like(
            state_dict[prefix + '.scaling'], SCALING_INIT)
        n_merged += 1
    return n_merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='融合前の ZiRa checkpoint (.pth)')
    parser.add_argument('output', help='融合後の保存先 (.pth)')
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    n = merge_state_dict(state_dict)
    if n == 0:
        raise SystemExit('RDB (hlrb/llrb/scaling) キーが見つかりません。'
                         'ZiRa の checkpoint ではない可能性があります。')
    torch.save(ckpt, args.output)
    print(f'merged {n} RDB modules: {args.input} -> {args.output}')


if __name__ == '__main__':
    main()
