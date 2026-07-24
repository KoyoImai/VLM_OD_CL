# =============================================================================
# exp_023 比較手法: ZiRa（リプレイフリー・論文準拠）/ electromagnetic（逐次 t=2）
#   zira_replayfree_underwater.py と同一（データのみ electromagnetic）。既存コード無変更。
# =============================================================================
_base_ = './zira_replayfree_base.py'

train_pipeline = _base_.train_pipeline

_em_root = '/workspace/kouyou/datasets/rf100_domain/electromagnetic/'
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
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=current_electromagnetic)
