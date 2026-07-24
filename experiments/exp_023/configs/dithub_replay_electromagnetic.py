# =============================================================================
# exp_023 比較手法: DitHub（リプレイ有り・論文準拠）/ electromagnetic（逐次 t=2）
#   dithub_replay_underwater.py と同一方針（データと dithub_classes を差し替え）。
#   [現在, 参照, 過去(underwater)] の3ソース、source_ratio [4,1,1]、batch_size=6。
#   load_from はドライバが前ドメイン(underwater)の last に差し替える。既存コード無変更。
# =============================================================================
_base_ = './dithub_replayfree_base.py'

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

model = dict(
    type='DitHubReplayGroundingDINO',
    dithub_classes=(
        'dog', 'person', 'Cell', 'Cell-Multi', 'No-Anomaly', 'Shadowing',
        'Unclassified', 'stray', 'target', 'cheetah', 'human', 'artefact',
        'distal phalanges', 'fifth metacarpal bone', 'first metacarpal bone',
        'fourth metacarpal bone', 'intermediate phalanges',
        'proximal phalanges', 'radius', 'second metacarpal bone',
        'soft tissue calcination', 'third metacarpal bone', 'ulna', 'acl', '0',
        'negative', 'positive', '6W', '7W', 'EH', 'label0', 'label1', 'label2',
        'angle', 'fracture', 'line', 'messed_up_angle', 'bicycle', 'car'))

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
