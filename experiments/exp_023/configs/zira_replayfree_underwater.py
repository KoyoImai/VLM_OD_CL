# =============================================================================
# exp_023 比較手法: ZiRa（リプレイフリー・論文準拠）/ underwater（逐次 t=1）
#   スケジュール/lr/optimizer/ckpt は zira_replayfree_base（論文準拠）を継承。
#   本 config はデータ（ODVG 単一ソース = 現在ドメイン underwater、同一ドメイン負例）と
#   バッチ（batch_size=4、iter ベースなので InfiniteSampler）だけを定義する。
#   逐次時は前ドメイン last を load_from に上書き（ドライバが実施）。既存コード無変更。
# =============================================================================
_base_ = './zira_replayfree_base.py'

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
