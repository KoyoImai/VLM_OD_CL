# =============================================================================
# exp_023: full finetuning + リプレイ / underwater（逐次 t=1）
#   逐次順 underwater -> electromagnetic -> videogames の1番目。
#   t=1 は過去ドメインが無いのでバッファ = 参照(Objects365v1 1,000) のみ。
#   現在:バッファ = 2:1（source_ratio [2,1]、batch_size=6 -> 現在4・参照2）。
#   batch_size は全ドメイン 6 に統一（現在ドメインのバッチを 4 に揃え、過去実験の batch4 と整合）。
#   すべて OD-ODVG ＋ RandomSamplingNegPos（同一ドメイン負例）。既存ファイルは無変更。
# =============================================================================
_base_ = './fullft_replay_base.py'

train_pipeline = _base_.train_pipeline  # ベース（事前学習）の ODVG＋負例サンプリング pipeline

# パス
_uw_root = '/workspace/kouyou/datasets/rf100_domain/underwater/'
_o365_root = '/workspace/kouyou/datasets/o365v1_stage/Objects365_v1/2019-08-02/'
_buf = '/workspace/kouyou/mmdetection/experiments/exp_023/buffer/'

# 現在ドメイン（underwater、OD-ODVG）
current_underwater = dict(
    type='ODVGDataset',
    data_root=_uw_root,
    ann_file='underwater_train_od.json',
    label_map_file='underwater_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)

# 参照バッファ（Objects365v1 1,000。画像・label_map は Objects365 側、ann_file はバッファサブセット）
reference_buffer = dict(
    type='ODVGDataset',
    data_root=_o365_root,
    ann_file=_buf + 'reference_o365v1_1000.odvg.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)

# 混合: 現在:バッファ = 2:1（batch_size=6 -> 現在4・参照2）
train_dataloader = dict(
    _delete_=True,
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='CurrentEpochMultiSourceSampler', batch_size=6, source_ratio=[2, 1]),
    dataset=dict(
        type='ConcatDataset',
        datasets=[current_underwater, reference_buffer]))
