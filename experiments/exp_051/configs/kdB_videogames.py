# =============================================================================
# exp_051: 蒸留B（テキスト特徴のみ）＋リプレイ（バッチ [4,2,2]×8）/ videogames（逐次 t=3）
#
# 【自動生成】experiments/exp_051/gen_configs.py。直接編集しない。
#
# データ・バッチ・スケジュールは exp_043 の replay config を継承し、
# 本 config が上書きするのはモデル（蒸留点）と ckpt 保存方針だけ（design.md §2）。
# exp_043 kdE との差分は kd.targets のみ。
# 教師 θ_{t-1}（t=1 は θ0 の実パス）はドライバが model.teacher_ckpt で渡す。
# =============================================================================
_base_ = '../../exp_043/configs/replay_videogames.py'

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
        targets=['txt'],
        loss=dict(type='FeatureDistillLoss', form='l2'),
        loss_weight=10.0,
        num_buffer_per_batch=4))

# ckpt は last のみ・optimizer 状態なし（2026-08-22 決定。exp_043 は全 epoch 保持）
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=-1,
                    save_optimizer=False, save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=50))
