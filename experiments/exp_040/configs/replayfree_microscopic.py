# =============================================================================
# exp_040: リプレイ無し逐次FT（条件A＝全モジュール） / microscopic（逐次 t=5）
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
_base_ = '../../exp_024/configs/fullft_replayfree_base.py'

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

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=current_microscopic)
