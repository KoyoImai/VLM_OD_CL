# =============================================================================
# exp_023: full finetuning + リプレイ / videogames（逐次 t=3・最終）
#   逐次順 underwater -> electromagnetic -> videogames の3番目。
#   バッファ = 参照(Objects365v1 1,000) ＋ 過去プール(underwater 500 ＋ electromagnetic 500)。
#   過去2ドメインを入れ子 ConcatDataset で1つの「過去プール」source に束ねる（解釈B）。
#   [現在, 参照, 過去プール] の3ソース、source_ratio [4,1,1]、batch_size=6。
#   現在:バッファ = 4:(1+1) = 2:1。参照・過去とも毎バッチ 1件以上（過去はプールから一様）。
#   load_from はドライバが「前ドメイン(electromagnetic)の last」に差し替える。
# =============================================================================
_base_ = './fullft_replay_base.py'

train_pipeline = _base_.train_pipeline

_vg_root = '/workspace/kouyou/datasets/rf100_domain/videogames/'
_uw_root = '/workspace/kouyou/datasets/rf100_domain/underwater/'
_em_root = '/workspace/kouyou/datasets/rf100_domain/electromagnetic/'
_o365_root = '/workspace/kouyou/datasets/o365v1_stage/Objects365_v1/2019-08-02/'
_buf = '/workspace/kouyou/mmdetection/experiments/exp_023/buffer/'

current_videogames = dict(
    type='ODVGDataset',
    data_root=_vg_root,
    ann_file='videogames_train_od.json',
    label_map_file='videogames_label_map.json',
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

# 過去プール（underwater 500 ＋ electromagnetic 500 を入れ子 ConcatDataset で1 source に）
past_pool = dict(
    type='ConcatDataset',
    datasets=[
        dict(
            type='ODVGDataset',
            data_root=_uw_root,
            ann_file=_buf + 'underwater_500.odvg.json',
            label_map_file='underwater_label_map.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=False),
            return_classes=True,
            pipeline=train_pipeline),
        dict(
            type='ODVGDataset',
            data_root=_em_root,
            ann_file=_buf + 'electromagnetic_500.odvg.json',
            label_map_file='electromagnetic_label_map.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=False),
            return_classes=True,
            pipeline=train_pipeline),
    ])

train_dataloader = dict(
    _delete_=True,
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='CurrentEpochMultiSourceSampler', batch_size=6, source_ratio=[4, 1, 1]),
    dataset=dict(
        type='ConcatDataset',
        # 過去プールが入れ子 ConcatDataset で metainfo に cumulative_sizes を持つため、
        # トップの整合チェックで無視する（mmdet ConcatDataset の既存パラメータ）。
        ignore_keys=['cumulative_sizes'],
        datasets=[current_videogames, reference_buffer, past_pool]))
