# =============================================================================
# exp_023 比較手法: ZiRa（リプレイフリー・論文準拠）/ videogames（逐次 t=3）
#   zira_replayfree_underwater.py と同一（データのみ videogames）。
#   87 クラスだが学習は RandomSamplingNegPos（max_tokens=256）で収まる。
#   全クラス一括評価は eval_videogames.py（chunked_size）で対処。既存コード無変更。
# =============================================================================
_base_ = './zira_replayfree_base.py'

train_pipeline = _base_.train_pipeline

_vg_root = '/workspace/kouyou/datasets/rf100_domain/videogames/'
current_videogames = dict(
    type='ODVGDataset',
    data_root=_vg_root,
    ann_file='videogames_train_od.json',
    label_map_file='videogames_label_map.json',
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
    dataset=current_videogames)
