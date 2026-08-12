# =============================================================================
# exp_039: ZiRa / リプレイ無し / videogames
#
# 【自動生成】experiments/exp_039/gen_configs.py。直接編集しない。
#
# データ・スケジュール・optimizer はリプレイ・蒸留と同一にするため、継承元を
# ../../exp_024/configs/fullft_replayfree_videogames.py に取る（design.md §2.1）。
#   20 epoch / MultiStepLR milestones=[15] / AdamW lr 1e-4 wd 1e-4 /
#   grad clip 0.1(L2) / batch 4 / dn 有効 / num_classes 256 / seed 0 /
#   ODVGDataset + RandomSamplingNegPos
# 本 config が上書きするのは**手法固有の設定だけ**（design.md §2.2）。
#
# 逐次学習では前ドメインの融合済み ckpt を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '../../exp_024/configs/fullft_replayfree_videogames.py'

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
