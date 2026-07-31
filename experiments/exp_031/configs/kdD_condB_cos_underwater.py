# =============================================================================
# exp_031-D: 条件B（特徴抽出+融合）+ 蒸留D（融合後（feature enhancer 出力）・コサイン類似度）+ リプレイ / underwater
#   exp_029 の対応 config との差は model.kd.loss.form が 'cosine' になる1点のみ。
#   paramwise_cfg（下流6モジュールの凍結）は継承元のまま変更しない。
#   design: experiments/exp_028/design.md（共通仕様）／experiments/exp_031/design.md（差分）
# =============================================================================
_base_ = '../../exp_027/configs/condB_replay_underwater.py'

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
    # 教師 = θ_{t-1}（t=1 は θ0）。ドライバが --cfg-options で実パスに差し替える。
    teacher_ckpt=None,
    kd=dict(
        targets=['fus'],
        loss=dict(type='FeatureDistillLoss', form='cosine'),
        # λ = 1.0 で全条件固定（2026-07-27 決定。design.md §4.5）。
        loss_weight=1.0,
        num_buffer_per_batch=2))
