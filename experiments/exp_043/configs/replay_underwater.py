# =============================================================================
# exp_043: リプレイのみ（バッチ [4,2,2]×8） / underwater（逐次 t=1）
#
# 【自動生成】experiments/exp_043/gen_configs.py。直接編集しない。
#
# 6ドメイン逐次 underwater -> electromagnetic -> videogames -> aerial -> microscopic -> documents の 1 番目。
# 学習枠は既存 RF100 系列と同一（20 epoch / lr 1e-4 / wd 1e-4。design.md §2.1）。
# バッチ構成が本実験の核（design.md §2.2）:
#   t=1: [現在4, 汎用4]（2 ソース）
# 重みは前ドメインの last をドライバが load_from で渡す（t=1 は config の θ0）。
#   画像 12,633 枚 / クラス 28
# =============================================================================
_base_ = '../../exp_023/configs/fullft_replay_base.py'

train_pipeline = _base_.train_pipeline

_root = '/workspace/kouyou/datasets/rf100_domain/underwater/'

# 現在ドメイン（OD-ODVG）
current_underwater = dict(
    type='ODVGDataset',
    data_root=_root,
    ann_file='underwater_train_od.json',
    label_map_file='underwater_label_map.json',
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

# t=1: [現在, 汎用] の 2 ソース、source_ratio [4, 4]、batch 8/GPU（合計 32）
train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='CurrentEpochMultiSourceSampler', batch_size=8, source_ratio=[4, 4]),
    dataset=dict(
        type='ConcatDataset',
        datasets=[current_underwater, reference_buffer]))
