# =============================================================================
# exp_044: 蒸留E+（projector・全データ）＋リプレイ / underwater（逐次 t=1）
#
# 【自動生成】experiments/exp_044/gen_configs.py。直接編集しない。
#
# データ・スケジュール・バッチ構成（既存系列 [現在4, 汎用1, 過去1]×6、t=1 は
# [現在4, 汎用2]×6）は ../../exp_023/configs/fullft_replay_underwater.py を継承し、
# 本 config が上書きするのはモデル（蒸留E+）だけ（design.md §2.1-2.2）。
#
# 逐次学習では前タスクの epoch_20 を load_from に、教師 θ_{t-1}（t=1 は θ0）を
# model.teacher_ckpt に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '../../exp_023/configs/fullft_replay_underwater.py'

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
