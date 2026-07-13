# exp_020 追加実験: ZiRa を公式実装のハイパラで学習する (electromagnetic)
# 公式に合わせる項目: 2000 iter 固定 / total batch 2 (1 GPU) / AdamW lr 1e-3,
#   wd 1e-4 / iter 800 で lr x0.1 / grad clip 0.1 / η=0.2 / λ=0.1
# 公式に合わせない項目: dn 有効 (全基準線と共通) / seed=0 (プロジェクト規約)
# 詳細: design.md 追記 (2026-07-12)、underwater 版と同一の設定
_base_ = './zira_electromagnetic.py'

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=2000,
    val_interval=2000)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=2000,
        by_epoch=False,
        milestones=[800],
        gamma=0.1)
]

optim_wrapper = dict(optimizer=dict(lr=0.001))

train_dataloader = dict(
    batch_size=2,
    sampler=dict(_delete_=True, type='InfiniteSampler', shuffle=True))

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=2000, max_keep_ckpts=1, save_best=None))

log_processor = dict(by_epoch=False)
