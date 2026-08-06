# =============================================================================
# exp_034: ZiRa 学習 config / Aquarium
#
# 【自動生成】experiments/exp_034/gen_configs.py が
# experiments/exp_034/odinw_official_tasks.py から生成する。直接編集しない。
#
# スケジュール・lr・optimizer・ZiRa の設定は zira_odinw13_base.py（公式準拠）を継承し、
# 本 config はデータだけを定義する。学習データは公式実装と同じ版の train split。
#   画像 448 枚 / box 3,324 / クラス 7（2000 iter × batch 2 = 4000 枚で 8.9 周）
# 逐次学習では前タスクの Rep+ 融合済み ckpt を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = './zira_odinw13_base.py'

class_name = ('fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish',
              'stingray')
metainfo = dict(classes=class_name)
_data_root = 'data/odinw/' + 'Aquarium/Aquarium Combined.v2-raw-1024.coco/'

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
