# =============================================================================
# exp_023 比較手法: ZiRa（リプレイ有り・論文準拠）/ electromagnetic（逐次 t=2）
#   バッファ=参照(Objects365v1 1,000)＋過去プール(underwater 500)。
#   [現在, 参照, 過去(underwater)] の3ソース、source_ratio [4,1,1]、batch_size=6
#   -> 現在4・参照1・過去1（参照・過去とも毎バッチ必ず1件以上＝解釈B）。
#   load_from はドライバが前ドメイン(underwater)の last に差し替える。既存コード無変更。
# =============================================================================
_base_ = './zira_replayfree_base.py'

train_pipeline = _base_.train_pipeline

_em_root = '/workspace/kouyou/datasets/rf100_domain/electromagnetic/'
_uw_root = '/workspace/kouyou/datasets/rf100_domain/underwater/'
_o365_root = '/workspace/kouyou/datasets/o365v1_stage/Objects365_v1/2019-08-02/'
_buf = '/workspace/kouyou/mmdetection/experiments/exp_023/buffer/'

current_electromagnetic = dict(
    type='ODVGDataset',
    data_root=_em_root,
    ann_file='electromagnetic_train_od.json',
    label_map_file='electromagnetic_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)

reference_buffer = dict(
    type='ODVGDataset',
    data_root=_o365_root,
    ann_file=_buf + 'reference_o365v1_1000.odvg.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)

past_underwater = dict(
    type='ODVGDataset',
    data_root=_uw_root,
    ann_file=_buf + 'underwater_500.odvg.json',
    label_map_file='underwater_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)

train_dataloader = dict(
    _delete_=True,
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='MultiSourceSampler', batch_size=6, source_ratio=[4, 1, 1]),
    dataset=dict(
        type='ConcatDataset',
        datasets=[current_electromagnetic, reference_buffer, past_underwater]))
