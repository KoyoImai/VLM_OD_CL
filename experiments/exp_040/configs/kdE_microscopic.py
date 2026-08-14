# =============================================================================
# exp_040: 蒸留E（3項・L2・λ=10）＋ リプレイ（条件A） / microscopic（逐次 t=5）
#
# 【自動生成】experiments/exp_040/gen_configs.py。直接編集しない。
#
# 6ドメイン逐次 underwater -> electromagnetic -> videogames -> aerial -> microscopic -> documents の 5 番目。
# 重みは前ドメインの last をドライバが load_from で渡す（t=4 は前半3タスク目の
# epoch_20.pth。design.md §2.1）。
#   画像 9,576 枚 / box 74,208 / クラス 28
#
# 学習設定は前半（exp_024 / exp_027 / exp_035）と完全に同一で、本 config が
# 定義するのはデータだけである（design.md §2.3）。
# =============================================================================
_base_ = '../../exp_023/configs/fullft_replay_base.py'

train_pipeline = _base_.train_pipeline

_root = '/workspace/kouyou/datasets/rf100_domain/microscopic/'

# 現在ドメイン（OD-ODVG）
current_microscopic = dict(
    type='ODVGDataset',
    data_root=_root,
    ann_file='microscopic_train_od.json',
    label_map_file='microscopic_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)

# 参照バッファ（Objects365v1 1,000。全タスク共通・固定。過去プールには統合しない）
reference_buffer = dict(
    type='ODVGDataset',
    data_root='/workspace/kouyou/datasets/o365v1_stage/Objects365_v1/2019-08-02/',
    ann_file='/workspace/kouyou/mmdetection/experiments/exp_023/buffer/reference_o365v1_1000.odvg.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)

# 過去プール（学習済み 4 ドメイン × 500 を入れ子 ConcatDataset で1 source に）
#   underwater 500 + electromagnetic 500 + videogames 500 + aerial 500
past_pool = dict(
    type='ConcatDataset',
    datasets=[
        dict(
            type='ODVGDataset',
            data_root='/workspace/kouyou/datasets/rf100_domain/underwater/',
            ann_file='/workspace/kouyou/mmdetection/experiments/exp_023/buffer/underwater_500.odvg.json',
            label_map_file='underwater_label_map.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=False),
            return_classes=True,
            pipeline=train_pipeline),
        dict(
            type='ODVGDataset',
            data_root='/workspace/kouyou/datasets/rf100_domain/electromagnetic/',
            ann_file='/workspace/kouyou/mmdetection/experiments/exp_023/buffer/electromagnetic_500.odvg.json',
            label_map_file='electromagnetic_label_map.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=False),
            return_classes=True,
            pipeline=train_pipeline),
        dict(
            type='ODVGDataset',
            data_root='/workspace/kouyou/datasets/rf100_domain/videogames/',
            ann_file='/workspace/kouyou/mmdetection/experiments/exp_023/buffer/videogames_500.odvg.json',
            label_map_file='videogames_label_map.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=False),
            return_classes=True,
            pipeline=train_pipeline),
        dict(
            type='ODVGDataset',
            data_root='/workspace/kouyou/datasets/rf100_domain/aerial/',
            ann_file='/workspace/kouyou/mmdetection/experiments/exp_023/buffer/aerial_500.odvg.json',
            label_map_file='aerial_label_map.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=False),
            return_classes=True,
            pipeline=train_pipeline),
    ])

# 混合: [現在, 参照, 過去プール] の3ソース、source_ratio [4,1,1]、batch_size=6
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
        datasets=[current_microscopic, reference_buffer, past_pool]))

# 手法固有: 蒸留E（画像・テキスト・融合の3特徴、L2）、λ=10.0（exp_035 と同一）
custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.datasets.samplers.current_epoch_multi_source_sampler',
        'mmdet.models.losses.feature_distill_loss',
        'mmdet.models.detectors.kd_grounding_dino',
    ],
    allow_failed_imports=False)

model = dict(
    type='KDGroundingDINO',
    # 教師 = θ_{{t-1}}。ドライバが --cfg-options で実パスに差し替える。
    teacher_ckpt=None,
    kd=dict(
        targets=['img', 'txt', 'fus'],
        loss=dict(type='FeatureDistillLoss', form='l2'),
        loss_weight=10.0,
        num_buffer_per_batch=2))
