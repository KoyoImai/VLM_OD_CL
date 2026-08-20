# =============================================================================
# exp_048: 条件2 kdEall（リプレイ[4,1,1]×6・全6枚に蒸留E） / aerial（逐次 t=4）
#
# 【自動生成】experiments/exp_048/gen_configs.py。直接編集しない。
#
# データ・スケジュールは ../../exp_040/configs/replay_aerial.py
# を継承し、本 config が上書きするのはモデル（蒸留E）だけ（design.md §2）。
# 逐次学習では前タスクの epoch_20 を load_from に、教師 θ_{t-1}（t=1 は θ0）を
# model.teacher_ckpt に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '../../exp_040/configs/replay_aerial.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.datasets.samplers.current_epoch_multi_source_sampler',
        'mmdet.models.losses.feature_distill_loss',
        'mmdet.models.detectors.kd_grounding_dino',
    ],
    allow_failed_imports=False)

# 条件2（design.md §2.4）: リプレイあり・全サンプル蒸留。
# バッチは [現在4, 汎用1, 過去1]×6/GPU（t=1 は [現在4, 汎用2]。継承元のまま）。
# num_buffer_per_batch=6 = バッチサイズ → 蒸留E（L2・λ=10・教師 θ_{t-1}）が
# 現在ドメイン 4 枚とバッファ 2 枚の全 6 枚に掛かる（exp_035 はバッファ 2 枚のみ）。
model = dict(
    type='KDGroundingDINO',
    teacher_ckpt=None,
    kd=dict(
        targets=['img', 'txt', 'fus'],
        loss=dict(type='FeatureDistillLoss', form='l2'),
        loss_weight=10.0,
        num_buffer_per_batch=6))
