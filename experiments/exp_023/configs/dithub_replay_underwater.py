# =============================================================================
# exp_023 比較手法: DitHub（リプレイ有り・論文準拠）/ underwater（逐次 t=1）
#   DitHub 手法＋論文準拠スケジュール（dithub_replayfree_base）に、リプレイ混合を載せる。
#   検出器は DitHubReplayGroundingDINO（新規サブクラス、既存 dithub コード無変更）。
#   参照/過去ドメイン画像（dithub_classes に無いクラス）は specialization 中 warmup_lora_a
#   経路を通る（クラッシュ回避、ユーザー選択 2026-07-22。§1 特則は評価時のフィルタで成立）。
#   混合は fullft_replay_underwater と同一構造。t=1 はバッファ=参照のみ、source_ratio [2,1]、
#   batch_size=6 -> 現在4・参照2。
# =============================================================================
_base_ = './dithub_replayfree_base.py'

# base の dithub imports に、+replay 用サブクラス exp023_dithub_replay を追加
custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.layers.dithub_layers',
        'mmdet.models.detectors.dithub_grounding_dino',
        'mmdet.engine.hooks.dithub_phase_hook',
        'mmdet.engine.hooks.dithub_seq_phase_hook',
        'mmdet.models.detectors.dithub_replay_grounding_dino',
    ],
    allow_failed_imports=False)

# 検出器を +replay 用サブクラスへ。dithub_classes は現在ドメイン（underwater）。
model = dict(
    type='DitHubReplayGroundingDINO',
    dithub_classes=(
        'pipe', 'fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish',
        'stingray', 'peix', 'taca', 'echinus', 'holothurian', 'scallop',
        'waterweeds', 'Arborescent', 'Caespitose-a', 'Caespitose-b',
        'Columnar', 'Corymbose', 'Digitate', 'Encrusting', 'Foliose',
        'Massive-Faviidae', 'Massive-Merulinidae', 'Massive-Mussidae',
        'Massive-Poritidae', 'Solitary', 'Tabular'))

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
