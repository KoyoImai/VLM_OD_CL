# =============================================================================
# exp_024: リプレイ無し逐次FT（条件A＝全モジュール）/ underwater（逐次 t=1）
#   逐次順 underwater -> electromagnetic -> videogames の1番目。
#   リプレイ無しのため学習データは現在ドメインのみ（バッファを混ぜない）。
#   batch_size=4/GPU（note04「バッチの内訳（リプレイフリー）：現在ドメイン4」）。
#   exp_023 の +replay 版は同じ現在4にバッファ2を足して batch=6 としており、
#   現在ドメインの露出は両者で一致する（差分はリプレイの有無のみ）。
#   OD-ODVG ＋ RandomSamplingNegPos（同一ドメイン負例）。既存ファイルは無変更。
# =============================================================================
_base_ = './fullft_replayfree_base.py'

train_pipeline = _base_.train_pipeline  # ベース（事前学習）の ODVG＋負例サンプリング pipeline

_uw_root = '/workspace/kouyou/datasets/rf100_domain/underwater/'

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

# リプレイ無し: 現在ドメイン単独（ConcatDataset / MultiSourceSampler は使わない）
train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=current_underwater)
