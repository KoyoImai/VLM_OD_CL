#!/usr/bin/env python3
"""exp_051: 蒸留の適用箇所アブレーション（kdA/kdB/kdD × 6 ドメイン）の config を生成する。

design.md §2 のとおり、データ・バッチ・スケジュールは exp_043 の replay config を
継承し、モデルだけを KDGroundingDINO に差し替える。exp_043 kdE との差分は
kd.targets（と ckpt 保存方針）のみ。

    kdA: targets=['img']   （neck 出力の多スケール画像特徴）
    kdB: targets=['txt']   （text_feat_map 出力のテキスト特徴）
    kdD: targets=['fus']   （enhancer 出力 memory / memory_text）

教師 θ_{t-1}（t=1 は θ0 の実パス）はドライバが --cfg-options model.teacher_ckpt= で渡す。
ckpt は last（epoch_20）のみ・optimizer 状態なし（2026-08-22 決定）。

生成するもの（18 本）: configs/kd{A,B,D}_{6ドメイン}.py

使い方: python experiments/exp_051/gen_configs.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, 'configs')

ORDER = ['underwater', 'electromagnetic', 'videogames',
         'aerial', 'microscopic', 'documents']
CONDS = {'kdA': ['img'], 'kdB': ['txt'], 'kdD': ['fus']}
COND_JA = {'kdA': '蒸留A（画像特徴のみ）', 'kdB': '蒸留B（テキスト特徴のみ）',
           'kdD': '蒸留D（融合後特徴のみ）'}

TMPL = '''\
# =============================================================================
# exp_051: {cond_ja}＋リプレイ（バッチ [4,2,2]×8）/ {domain}（逐次 t={t}）
#
# 【自動生成】experiments/exp_051/gen_configs.py。直接編集しない。
#
# データ・バッチ・スケジュールは exp_043 の replay config を継承し、
# 本 config が上書きするのはモデル（蒸留点）と ckpt 保存方針だけ（design.md §2）。
# exp_043 kdE との差分は kd.targets のみ。
# 教師 θ_{{t-1}}（t=1 は θ0 の実パス）はドライバが model.teacher_ckpt で渡す。
# =============================================================================
_base_ = '../../exp_043/configs/replay_{domain}.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.datasets.samplers.current_epoch_multi_source_sampler',
        'mmdet.models.losses.feature_distill_loss',
        'mmdet.models.detectors.kd_grounding_dino',
    ],
    allow_failed_imports=False)

model = dict(
    type='KDGroundingDINO',
    teacher_ckpt=None,
    kd=dict(
        targets={targets},
        loss=dict(type='FeatureDistillLoss', form='l2'),
        loss_weight=10.0,
        num_buffer_per_batch=4))

# ckpt は last のみ・optimizer 状態なし（2026-08-22 決定。exp_043 は全 epoch 保持）
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=-1,
                    save_optimizer=False, save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=50))
'''


def gen():
    os.makedirs(CFG_DIR, exist_ok=True)
    for cond, targets in CONDS.items():
        for i, d in enumerate(ORDER):
            body = TMPL.format(cond_ja=COND_JA[cond], domain=d, t=i + 1,
                               targets=targets)
            path = os.path.join(CFG_DIR, f'{cond}_{d}.py')
            open(path, 'w').write(body)
            print('generated', os.path.basename(path))


if __name__ == '__main__':
    gen()
