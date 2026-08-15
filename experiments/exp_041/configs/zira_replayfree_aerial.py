# =============================================================================
# exp_041: ZiRa / リプレイ無し / aerial（逐次 t=4）
#
# 【自動生成】experiments/exp_041/gen_configs.py。直接編集しない。
#
# データ・スケジュール・optimizer は exp_040 と同一にするため、継承元を
# ../../exp_040/configs/replayfree_aerial.py に取る（design.md §2.2）。
#   20 epoch / MultiStepLR milestones=[15] / AdamW lr 1e-4 wd 1e-4 /
#   grad clip 0.1(L2) / batch 4 / dn 有効 / num_classes 256 / seed 0
# 本 config が上書きするのは**手法固有の設定だけ**（design.md §2.3）。
#
# 逐次学習では前タスクの融合済み ckpt を load_from に上書きする（ドライバが実施。
# t=4 は exp_039 の merged_after_t3_videogames.pth。design.md §2.1）。
# =============================================================================
_base_ = '../../exp_040/configs/replayfree_aerial.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.datasets.samplers.current_epoch_multi_source_sampler',
        'mmdet.models.layers.zira_layers',
        'mmdet.models.necks.zira_channel_mapper',
        'mmdet.models.detectors.zira_grounding_dino',
    ],
    allow_failed_imports=False)

# 手法固有: ZiL の λ=0.1、neck を RDB 付きに差し替え
model = dict(
    type='ZiRaGroundingDINO',
    zil_loss_weight=0.1,
    neck=dict(type='ZiRaChannelMapper'))

# 手法固有: LLRB のみ lr×η（η=0.2）。他の custom_keys は継承元のものを保つ。
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1),
            language_model=dict(lr_mult=0.1),
            llrb=dict(lr_mult=0.2))))
