# =============================================================================
# exp_023 比較手法: DitHub（リプレイフリー・論文準拠）/ underwater（逐次 t=1）
#   スケジュール/lr/optimizer/phase hook/ckpt は dithub_replayfree_base（論文準拠）を継承。
#   本 config は dithub_classes（クラス別 LoRA 用、label_map の index 順）と
#   データ（ODVG 単一ソース = underwater、同一ドメイン負例）、バッチ（batch_size=4、
#   iter ベースなので InfiniteSampler）を定義する。
#   逐次時は前ドメイン last を load_from に上書き（ドライバが実施）。既存コード無変更。
# =============================================================================
_base_ = './dithub_replayfree_base.py'

# クラス別 LoRA 用のクラス名（underwater_label_map.json の index 順）
model = dict(
    dithub_classes=(
        'pipe', 'fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish',
        'stingray', 'peix', 'taca', 'echinus', 'holothurian', 'scallop',
        'waterweeds', 'Arborescent', 'Caespitose-a', 'Caespitose-b',
        'Columnar', 'Corymbose', 'Digitate', 'Encrusting', 'Foliose',
        'Massive-Faviidae', 'Massive-Merulinidae', 'Massive-Mussidae',
        'Massive-Poritidae', 'Solitary', 'Tabular'))

train_pipeline = _base_.train_pipeline

_uw_root = '/workspace/kouyou/datasets/rf100_domain/underwater/'
current_underwater = dict(
    type='ODVGDataset',
    data_root=_uw_root,
    ann_file='underwater_train_od.json',
    label_map_file='underwater_label_map.json',
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
    dataset=current_underwater)
