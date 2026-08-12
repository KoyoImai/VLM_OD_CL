# =============================================================================
# exp_037: DitHub 学習 config / t=01 pistols
#
# 【自動生成】experiments/exp_037/gen_configs.py が
# experiments/exp_034/odinw_official_tasks.py から生成する。直接編集しない。
#
# スケジュール・lr・optimizer・DitHub の設定は dithub_odinw13_base.py（公式準拠）を
# 継承し、本 config はデータとクラスだけを定義する。学習データは公式実装と同じ版の
# train split。
#   画像 2,377 枚 / box 2,728 / クラス 1（3000 iter × batch 2 = 6000 枚で 2.5 周）
#
# dithub_classes は当該タスクのクラスのみ。公式は全タスク分のスロットを最初から確保するが、
# B=0 と enable_per_class の対象が現タスク限定なので学習は等価（design.md §1.7）。
# クラス数が 1 のタスクでは warmup を行わず iter 0 から per-class A を学習する
# （公式 do_warmup_a の条件。design.md §1.2）。本タスクは 単一クラス（warmup なし） である。
#
# trained_classes は「本タスクのクラスのうち、逐次順で先行するタスクに出現したもの」で、
# specialization 切替時に式3（A ← λ_A·A_warmup + (1-λ_A)·A_old）が発火する対象。
# 逐次順は seed 42（exp_034 と同一）。
#
# 逐次学習では前タスクの成長ライブラリ ckpt を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = './dithub_odinw13_base.py'

class_name = ('pistol', )
metainfo = dict(classes=class_name)
_data_root = 'data/odinw/' + 'pistols/export/'

model = dict(dithub_classes=class_name)

custom_hooks = [
    dict(
        type='DitHubSeqPhaseHook',
        warmup_iters=1500,
        trained_classes=[])
]

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
        ann_file='train_annotations_without_background.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        return_classes=True,
        pipeline=_base_.train_pipeline))
