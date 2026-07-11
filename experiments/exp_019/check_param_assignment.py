#!/usr/bin/env python3
# =============================================================================
# exp_019: パラメータ帰属の検査（design.md §2 の相補性検査）
#   - モデルを config から構築し named_parameters を全列挙
#   - mmengine DefaultOptimWrapperConstructor と同じ規則
#     （custom_keys を (長さ, 辞書順) 降順で走査し最初に部分一致したキーを適用、
#       未一致は既定 lr_mult=1.0）で各条件の学習/凍結を判定
#   - 検査1: extfusion で学習されるパラメータ集合と downstream で学習される集合が
#            互いに素で、和が全パラメータに一致する（相補分割）
#   - 検査2: 意図した接頭辞リスト（design.md §2 の表）と実際の判定が一致する
#   - 出力: configs/param_assignment.md（帰属対応表）
# =============================================================================
import os
import sys

REPO = '/workspace/kouyou/mmdetection'
sys.path.insert(0, REPO)
os.chdir(REPO)

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.registry import MODELS

CFG_A = 'experiments/exp_019/configs/extfusion_underwater.py'
CFG_B = 'experiments/exp_019/configs/downstream_underwater.py'

# design.md §2 の意図した分割（接頭辞）
INTENDED_A = ('backbone.', 'language_model.', 'neck.', 'text_feat_map.',
              'encoder.', 'level_embed')                       # 抽出・融合部
INTENDED_B = ('decoder.', 'bbox_head.', 'memory_trans_fc.', 'memory_trans_norm.',
              'query_embedding.', 'dn_query_generator.')       # 下流


def lr_mult_of(name, custom_keys):
    """mmengine DefaultOptimWrapperConstructor の一致規則を再現。"""
    sorted_keys = sorted(custom_keys.keys(), key=lambda x: (len(x), x), reverse=True)
    for k in sorted_keys:
        if k in name:
            return float(custom_keys[k].get('lr_mult', 1.0))
    return 1.0


def main():
    init_default_scope('mmdet')
    cfg_a = Config.fromfile(CFG_A)
    cfg_b = Config.fromfile(CFG_B)
    model = MODELS.build(cfg_a.model)   # 構造は両条件で同一

    keys_a = cfg_a.optim_wrapper['paramwise_cfg']['custom_keys']
    keys_b = cfg_b.optim_wrapper['paramwise_cfg']['custom_keys']

    rows, errors = [], []
    n_train_a = n_train_b = 0
    for name, p in model.named_parameters():
        train_a = lr_mult_of(name, keys_a) > 0
        train_b = lr_mult_of(name, keys_b) > 0
        side_int = ('A' if any(name.startswith(x) for x in INTENDED_A) else
                    'B' if any(name.startswith(x) for x in INTENDED_B) else '?')
        # 検査1: 相補分割（どちらか一方でのみ学習される）
        if train_a == train_b:
            errors.append(f'相補性違反: {name} (extfusion={train_a}, downstream={train_b})')
        # 検査2: 意図と一致（A側は extfusion でのみ、B側は downstream でのみ学習）
        if side_int == '?':
            errors.append(f'意図分割の未割当: {name}')
        elif (side_int == 'A') != train_a:
            errors.append(f'意図と不一致: {name} 意図={side_int}, extfusion学習={train_a}')
        n_train_a += train_a
        n_train_b += train_b
        rows.append((name, side_int, tuple(p.shape)))

    out = ['# exp_019 パラメータ帰属対応表（自動生成: check_param_assignment.py）', '',
           f'- 総パラメータテンソル数: {len(rows)}',
           f'- 条件A(extfusion)で学習: {n_train_a} / 条件B(downstream)で学習: {n_train_b}',
           f'- 相補性・意図一致の検査: {"OK（違反なし）" if not errors else "NG"}', '']
    if errors:
        out += ['## 違反'] + [f'- {e}' for e in errors] + ['']
    out += ['## 帰属（A=抽出・融合部 / B=下流）', '',
            '| parameter | side | shape |', '|---|---|---|']
    out += [f'| `{n}` | {s} | {sh} |' for n, s, sh in rows]
    with open('experiments/exp_019/configs/param_assignment.md', 'w') as f:
        f.write('\n'.join(out) + '\n')

    print(f'[結果] tensors={len(rows)}  A学習={n_train_a}  B学習={n_train_b}  違反={len(errors)}')
    for e in errors[:20]:
        print(' -', e)
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
