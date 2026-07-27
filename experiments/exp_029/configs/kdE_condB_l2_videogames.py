# =============================================================================
# exp_029-E: 条件B（特徴抽出+融合） + 蒸留E（画像・テキスト・融合後すべて（3項平均）・L2ノルム）+ リプレイ / videogames
#   リプレイ・データ・スケジュールは exp_027 と完全に同一（継承のみ）。
#   差分は detector の差し替えと蒸留の設定だけ。
#   design: experiments/exp_028/design.md（共通仕様） ／ experiments/exp_029/design.md（差分）
# =============================================================================
_base_ = '../../exp_027/configs/condB_replay_videogames.py'

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
        targets=['img', 'txt', 'fus'],
        loss=dict(type='FeatureDistillLoss', form='l2'),
        # λ = 1.0 で全条件固定（2026-07-27 決定。design.md §4.5）。
        loss_weight=1.0,
        num_buffer_per_batch=2))
