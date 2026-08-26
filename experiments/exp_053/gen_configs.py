#!/usr/bin/env python3
# =============================================================================
# exp_053: 蒸留E × 学習可能モジュールのアブレーション・後半 3 ドメイン config 生成
#   （design.md §2）5 条件 × 3 ドメイン = 15 本を configs/ に生成する。
#   ベースは exp_040 の kdE_{domain}.py（旧バッチ [4,1,1]×6・λ=10・バッファ 2 枚・
#   教師 θ_{t-1}）。上書きは exp_052 と同一の 2 点だけ:
#     1. optim_wrapper.paramwise_cfg.custom_keys（凍結 = lr_mult 0.0）
#     2. ckpt 保存方針（last のみ・optimizer 状態なし）
#   条件表・キー照合の注意は experiments/exp_052/gen_configs.py と同一。
# =============================================================================
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'configs')
os.makedirs(OUT, exist_ok=True)

DOMAINS = ['aerial', 'microscopic', 'documents']   # t=4,5,6

ALL_KEYS = [
    'backbone', 'language_model', 'neck', 'text_feat_map', 'encoder',
    'decoder', 'bbox_head', 'memory_trans_fc', 'memory_trans_norm',
    'query_embedding', 'level_embed', 'dn_query_generator',
]

CONDS = {
    'swinNeck': {'backbone': 0.1, 'neck': 1.0},
    'bertTfm': {'language_model': 0.1, 'text_feat_map': 1.0},
    'enhancer': {'encoder': 1.0},
    'qsel': {'memory_trans_fc': 1.0, 'memory_trans_norm': 1.0,
             'level_embed': 1.0, 'query_embedding': 1.0},
    'decoder': {'decoder': 1.0},
}

LABELS = {
    'swinNeck': 'Swin-T + neck',
    'bertTfm': 'BERT + text_feat_map',
    'enhancer': 'Feature Enhancer（encoder）',
    'qsel': 'Language-guided Query Selection（memory_trans_fc/norm + level_embed + query_embedding）',
    'decoder': 'Cross-Modality Decoder（decoder.*）',
}

HEADER = '''\
# =============================================================================
# exp_053: 蒸留E × 学習可能モジュール = {label} / {domain}（逐次 t={t}）
#
# 【自動生成】experiments/exp_053/gen_configs.py。直接編集しない。
#
# ベース = exp_040 kdE（旧バッチ [4,1,1]×6・λ=10・バッファ由来 2 枚限定）。
# 上書きは paramwise の凍結（lr_mult 0.0）と ckpt 保存方針だけ（design.md §1.1）。
# t=4 の初期重み・教師は exp_052 の同一条件の videogames epoch_20（design.md §1.2）。
# 教師 θ_{{t-1}} はドライバが model.teacher_ckpt で渡す。
# =============================================================================
_base_ = '../../exp_040/configs/kdE_{domain}.py'

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
'''

FOOTER = '''\
        )))

# ckpt は last のみ・optimizer 状態なし（design.md §3）
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=-1,
                    save_optimizer=False, save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=50))
'''

for cond, train_keys in CONDS.items():
    for i, dom in enumerate(DOMAINS):
        t = i + 4
        lines = []
        for k in ALL_KEYS:
            mult = train_keys.get(k, 0.0)
            tag = '学習' if mult > 0 else '凍結'
            lines.append(f'            {k}=dict(lr_mult={mult}),  # {tag}')
        body = (HEADER.format(label=LABELS[cond], domain=dom, t=t)
                + '\n'.join(lines) + '\n' + FOOTER)
        path = os.path.join(OUT, f'{cond}_{dom}.py')
        with open(path, 'w') as f:
            f.write(body)
        print('wrote', os.path.relpath(path, HERE))
