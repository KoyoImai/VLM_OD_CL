#!/usr/bin/env python3
# =============================================================================
# exp_052: 蒸留E × 学習可能モジュールのアブレーション config 生成（design.md §4）
#   5 条件 × 前半 3 ドメイン = 15 本を configs/ に生成する。
#   ベースは exp_035 の kdE_condA_l2w100_{domain}.py（旧バッチ [4,1,1]×6・λ=10・
#   バッファ 2 枚限定・教師 θ_{t-1}）。本 config の上書きは
#     1. optim_wrapper.paramwise_cfg.custom_keys（凍結 = lr_mult 0.0。exp_017/018 方式）
#     2. ckpt 保存方針（last のみ・optimizer 状態なし。2026-08-22 決定）
#   の 2 点だけ。モジュール境界は exp_015 の区分（design.md §2.2）。
#
# custom_keys の照合は「パラメータ名の部分一致・キー長の降順」なので、
#   - BERT 内の 'encoder'/'backbone' は、より長い 'language_model' が先に一致して吸収される
#   （事前学習 config と exp_023 condA が既に依存している挙動）。
# 全 12 キーを毎条件で明示し、継承由来の lr_mult を残さない
# （absolute_pos_embed の decay_mult=0 だけはベースから継承される。lr は本表に従う）。
# =============================================================================
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'configs')
os.makedirs(OUT, exist_ok=True)

DOMAINS = ['underwater', 'electromagnetic', 'videogames']

# 全条件で明示する 12 キー（exp_018 の凍結スイープ＋text_feat_map）
ALL_KEYS = [
    'backbone', 'language_model', 'neck', 'text_feat_map', 'encoder',
    'decoder', 'bbox_head', 'memory_trans_fc', 'memory_trans_norm',
    'query_embedding', 'level_embed', 'dn_query_generator',
]

# 条件名 -> {学習するキー: lr_mult}（それ以外は 0.0）
CONDS = {
    'swinNeck': {'backbone': 0.1, 'neck': 1.0},
    'bertTfm': {'language_model': 0.1, 'text_feat_map': 1.0},
    'enhancer': {'encoder': 1.0},
    'qsel': {'memory_trans_fc': 1.0, 'memory_trans_norm': 1.0,
             'level_embed': 1.0, 'query_embedding': 1.0},
    'decoder': {'decoder': 1.0},
}

HEADER = '''\
# =============================================================================
# exp_052: 蒸留E × 学習可能モジュール = {label} / {domain}（逐次 t={t}）
#
# 【自動生成】experiments/exp_052/gen_configs.py。直接編集しない。
#
# ベース = exp_035 kdE（旧バッチ [4,1,1]×6・λ=10・バッファ由来 2 枚限定）。
# 上書きは paramwise の凍結（lr_mult 0.0）と ckpt 保存方針だけ（design.md §2.2）。
# 教師 θ_{{t-1}}（t=1 は θ0 の実パス）はドライバが model.teacher_ckpt で渡す。
# =============================================================================
_base_ = '../../exp_035/configs/kdE_condA_l2w100_{domain}.py'

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
'''

FOOTER = '''\
        )))

# ckpt は last のみ・optimizer 状態なし（design.md §5）
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=-1,
                    save_optimizer=False, save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=50))
'''

LABELS = {
    'swinNeck': 'Swin-T + neck',
    'bertTfm': 'BERT + text_feat_map',
    'enhancer': 'Feature Enhancer（encoder）',
    'qsel': 'Language-guided Query Selection（memory_trans_fc/norm + level_embed + query_embedding）',
    'decoder': 'Cross-Modality Decoder（decoder.*）',
}

for cond, train_keys in CONDS.items():
    for t, dom in enumerate(DOMAINS, 1):
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
