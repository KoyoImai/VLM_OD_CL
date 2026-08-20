# =============================================================================
# exp_048: 条件1 kdEonly（リプレイ無し・現在ドメイン全4枚に蒸留E） / microscopic（逐次 t=5）
#
# 【自動生成】experiments/exp_048/gen_configs.py。直接編集しない。
#
# データ・スケジュールは ../../exp_040/configs/replayfree_microscopic.py
# を継承し、本 config が上書きするのはモデル（蒸留E）だけ（design.md §2）。
# 逐次学習では前タスクの epoch_20 を load_from に、教師 θ_{t-1}（t=1 は θ0）を
# model.teacher_ckpt に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '../../exp_040/configs/replayfree_microscopic.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.losses.feature_distill_loss',
        'mmdet.models.detectors.kd_grounding_dino',
    ],
    allow_failed_imports=False)

# 条件1（design.md §2.3）: リプレイ無し・蒸留のみ。
# バッチは現在ドメイン 4 枚/GPU（DefaultSampler。継承元のまま）。
# num_buffer_per_batch=4 = バッチサイズ → 末尾スライスが全体になり、
# 蒸留E（L2・λ=10・教師 θ_{t-1}）が現在ドメインの全 4 枚に掛かる。
model = dict(
    type='KDGroundingDINO',
    teacher_ckpt=None,
    kd=dict(
        targets=['img', 'txt', 'fus'],
        loss=dict(type='FeatureDistillLoss', form='l2'),
        loss_weight=10.0,
        num_buffer_per_batch=4))
