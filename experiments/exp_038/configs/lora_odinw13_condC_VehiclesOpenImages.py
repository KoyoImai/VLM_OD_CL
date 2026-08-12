# =============================================================================
# exp_038: 素の LoRA 学習 config / t=05 VehiclesOpenImages
#
# 【自動生成】experiments/exp_038/gen_configs.py が
# experiments/exp_034/odinw_official_tasks.py から生成する。直接編集しない。
#
# スケジュール・lr・optimizer・LoRA の挿入箇所は lora_odinw13_base.py を継承し、
# 本 config はデータとクラスだけを定義する。学習データは公式実装と同じ版の train split。
#   画像 878 枚 / box 1,676 / クラス 5（3000 iter × batch 2 = 6000 枚で 6.8 周）
#
# 逐次学習では前タスクのマージ済み ckpt（θ_{t-1}）を load_from に上書きする
# （ドライバが実施）。LoRA は毎タスク B=0 から引き直すので、学習開始時点は θ_{t-1} と
# 厳密に等価である（design.md §2.3）。
# 逐次順は seed 42（exp_034 / exp_037 と同一）。
# =============================================================================
_base_ = './lora_odinw13_condC_base.py'

class_name = ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck')
metainfo = dict(classes=class_name)
_data_root = 'data/odinw/' + 'VehiclesOpenImages/416x416/'

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=_data_root,
        metainfo=metainfo,
        ann_file='train/annotations_without_background.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        return_classes=True,
        pipeline=_base_.train_pipeline))
