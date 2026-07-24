# =============================================================================
# exp_023 比較手法: DitHub（リプレイフリー・論文準拠）/ electromagnetic（逐次 t=2）
#   dithub_replayfree_underwater.py と同一（dithub_classes とデータのみ差し替え）。
#   既存コード無変更。
# =============================================================================
_base_ = './dithub_replayfree_base.py'

# クラス別 LoRA 用のクラス名（electromagnetic_label_map.json の index 順）
model = dict(
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
current_electromagnetic = dict(
    type='ODVGDataset',
    data_root=_em_root,
    ann_file='electromagnetic_train_od.json',
    label_map_file='electromagnetic_label_map.json',
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
    dataset=current_electromagnetic)
