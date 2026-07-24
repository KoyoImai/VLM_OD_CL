# =============================================================================
# exp_023 比較手法: ZiRa（リプレイ有り・論文準拠）/ underwater（逐次 t=1）
#   ZiRa 手法＋論文準拠スケジュール（zira_replayfree_base）に、リプレイ混合を載せる。
#   混合は fullft_replay_underwater と同一構造（ConcatDataset＋MultiSourceSampler）。
#   t=1 は過去が無いのでバッファ=参照(Objects365v1 1,000)のみ。source_ratio [2,1]、
#   batch_size=6 -> 現在4・参照2（replay-free の現在4/GPU と現在ドメインを一致させ、
#   差分をリプレイ=バッファ2追加だけにする）。既存コード無変更。
# =============================================================================
_base_ = './zira_replayfree_base.py'

train_pipeline = _base_.train_pipeline

_uw_root = '/workspace/kouyou/datasets/rf100_domain/underwater/'
_o365_root = '/workspace/kouyou/datasets/o365v1_stage/Objects365_v1/2019-08-02/'
_buf = '/workspace/kouyou/mmdetection/experiments/exp_023/buffer/'

current_underwater = dict(
    type='ODVGDataset',
    data_root=_uw_root,
    ann_file='underwater_train_od.json',
    label_map_file='underwater_label_map.json',
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

train_dataloader = dict(
    _delete_=True,
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='MultiSourceSampler', batch_size=6, source_ratio=[2, 1]),
    dataset=dict(
        type='ConcatDataset',
        datasets=[current_underwater, reference_buffer]))
