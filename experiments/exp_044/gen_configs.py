#!/usr/bin/env python3
"""exp_044: 蒸留E+（projector 蒸留・全データ対象）の config を生成する。

design.md §2 のとおり、データ・スケジュール・バッチ構成は既存 RF100 系列と同一にする。
そのため継承元をリプレイ側のドメイン config に取り、モデルだけを蒸留E+ に差し替える。

    experiments/exp_023/configs/fullft_replay_<domain>.py
    （t=1 [現在4, 汎用2] / t≥2 [現在4, 汎用1, 過去1]、batch 6/GPU）

教師 θ_{t-1}（t=1 は θ0）はドライバが --cfg-options model.teacher_ckpt= で渡す。
蒸留対象の全データ化と projector は KDProjGroundingDINO 側の実装（config 指定不要）。

生成するもの（3 本）: configs/kdEp_{underwater,electromagnetic,videogames}.py

使い方: python experiments/exp_044/gen_configs.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, 'configs')

# 前半 3 ドメインのみ学習する（2026-08-16 変更。design.md §1）
FULL_ORDER = ['underwater', 'electromagnetic', 'videogames']
BASE = {
    'underwater': '../../exp_023/configs/fullft_replay_underwater.py',
    'electromagnetic': '../../exp_023/configs/fullft_replay_electromagnetic.py',
    'videogames': '../../exp_023/configs/fullft_replay_videogames.py',
}

BODY = '''\
# =============================================================================
# exp_044: 蒸留E+（projector・全データ）＋リプレイ / {domain}（逐次 t={t}）
#
# 【自動生成】experiments/exp_044/gen_configs.py。直接編集しない。
#
# データ・スケジュール・バッチ構成（既存系列 [現在4, 汎用1, 過去1]×6、t=1 は
# [現在4, 汎用2]×6）は {base} を継承し、
# 本 config が上書きするのはモデル（蒸留E+）だけ（design.md §2.1-2.2）。
#
# 逐次学習では前タスクの epoch_20 を load_from に、教師 θ_{{t-1}}（t=1 は θ0）を
# model.teacher_ckpt に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '{base}'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.datasets.samplers.current_epoch_multi_source_sampler',
        'mmdet.models.losses.feature_distill_loss',
        'mmdet.models.detectors.kd_grounding_dino',
        'mmdet.models.detectors.kd_proj_grounding_dino',
    ],
    allow_failed_imports=False)

# 蒸留E+: projector（4 点独立・点ごと 2 層 MLP）と蒸留対象の全データ化は
# KDProjGroundingDINO の実装。損失は従来の蒸留E と同じ集計（L2・3 項・λ=10）。
model = dict(
    type='KDProjGroundingDINO',
    teacher_ckpt=None,
    kd=dict(
        targets=['img', 'txt', 'fus'],
        loss=dict(type='FeatureDistillLoss', form='l2'),
        loss_weight=10.0))
'''


def gen():
    os.makedirs(CFG_DIR, exist_ok=True)
    for i, d in enumerate(FULL_ORDER):
        path = os.path.join(CFG_DIR, f'kdEp_{d}.py')
        with open(path, 'w') as f:
            f.write(BODY.format(domain=d, t=i + 1, base=BASE[d]))
        print(f'generated {path}  (t={i + 1})')


if __name__ == '__main__':
    gen()
