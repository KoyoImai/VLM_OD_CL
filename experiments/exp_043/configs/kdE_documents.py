# =============================================================================
# exp_043: 蒸留E（λ=10・バッファ4枚限定）＋ リプレイ（バッチ [4,2,2]×8） / documents（逐次 t=6）
#
# 【自動生成】experiments/exp_043/gen_configs.py。直接編集しない。
#
# 6ドメイン逐次 underwater -> electromagnetic -> videogames -> aerial -> microscopic -> documents の 6 番目。
# 学習枠は既存 RF100 系列と同一（20 epoch / lr 1e-4 / wd 1e-4。design.md §2.1）。
# バッチ構成が本実験の核（design.md §2.2）:
#   t=6: [現在4, 汎用2, 過去2]（3 ソース、過去 5 ドメイン累積）
# 重みは前ドメインの last をドライバが load_from で渡す（t=1 は config の θ0）。
#   画像 17,866 枚 / クラス 59
# =============================================================================
_base_ = '../../exp_023/configs/fullft_replay_base.py'

train_pipeline = _base_.train_pipeline

_root = '/workspace/kouyou/datasets/rf100_domain/documents/'

# 現在ドメイン（OD-ODVG）
current_documents = dict(
    type='ODVGDataset',
    data_root=_root,
    ann_file='documents_train_od.json',
    label_map_file='documents_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)

# 汎用ドメイン（Objects365v1 1,000。seed 0 ランダム選択・全タスク共通）
reference_buffer = dict(
    type='ODVGDataset',
    data_root='/workspace/kouyou/datasets/o365v1_stage/Objects365_v1/2019-08-02/',
    ann_file='/workspace/kouyou/mmdetection/experiments/exp_023/buffer/reference_o365v1_1000.odvg.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)

# 過去プール（学習済み 5 ドメイン × 500 を入れ子 ConcatDataset で 1 ソースに。
# バッチにはここからランダムに 2 枚入る）
#   underwater 500 + electromagnetic 500 + videogames 500 + aerial 500 + microscopic 500
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
        dict(
            type='ODVGDataset',
            data_root='/workspace/kouyou/datasets/rf100_domain/microscopic/',
            ann_file='/workspace/kouyou/mmdetection/experiments/exp_023/buffer/microscopic_500.odvg.json',
            label_map_file='microscopic_label_map.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=False),
            return_classes=True,
            pipeline=train_pipeline),
    ])

# t>=2: [現在, 汎用, 過去プール] の 3 ソース、source_ratio [4, 2, 2]、batch 8/GPU（合計 32）
train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='CurrentEpochMultiSourceSampler', batch_size=8, source_ratio=[4, 2, 2]),
    dataset=dict(
        type='ConcatDataset',
        # 過去プールが入れ子 ConcatDataset で metainfo に cumulative_sizes を持つため無視する
        ignore_keys=['cumulative_sizes'],
        datasets=[current_documents, reference_buffer, past_pool]))

# 手法固有: 蒸留E（画像・テキスト・融合の 3 特徴、L2）、λ=10（exp_035/040 と同一）。
# 蒸留はバッチ末尾のバッファ由来 4 枚（t=1: 汎用4 / t>=2: 汎用2+過去2）に限定する
# （num_buffer_per_batch=4。design.md §2.4。従来系列は 2 枚）。
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
    # 教師 = θ_{{t-1}}（t=1 は θ0）。ドライバが --cfg-options で実パスに差し替える。
    teacher_ckpt=None,
    kd=dict(
        targets=['img', 'txt', 'fus'],
        loss=dict(type='FeatureDistillLoss', form='l2'),
        loss_weight=10.0,
        num_buffer_per_batch=4))
