# =============================================================================
# exp_024: リプレイ無し逐次FT（条件A＝全モジュール）/ electromagnetic（逐次 t=2）
#   逐次順 underwater -> electromagnetic -> videogames の2番目。
#   重みは前ドメイン（underwater）の last をドライバが --cfg-options load_from で渡す。
#   LR スケジュール・optimizer 状態はリセットされる（load_from は重みのみ読む）。
#   リプレイ無しのため学習データは現在ドメインのみ。batch_size=4/GPU。
# =============================================================================
_base_ = './fullft_replayfree_base.py'

train_pipeline = _base_.train_pipeline

_em_root = '/workspace/kouyou/datasets/rf100_domain/electromagnetic/'

# 現在ドメイン（electromagnetic、OD-ODVG）
current_electromagnetic = dict(
    type='ODVGDataset',
    data_root=_em_root,
    ann_file='electromagnetic_train_od.json',
    label_map_file='electromagnetic_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=current_electromagnetic)
