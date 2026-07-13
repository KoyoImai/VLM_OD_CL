# exp_021: DitHub 再現・公式準拠版 (videogames)
# 3000 iter (warmup 1500 + specialization 1500) / batch 2 / lr 1e-3 / wd 1e-2 /
# iter 1200 で lr x0.1 / 1 GPU / dn 有効 / seed 0
_base_ = './dithub_videogames.py'

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=3000,
    val_interval=3000)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=3000,
        by_epoch=False,
        milestones=[1200],
        gamma=0.1)
]

optim_wrapper = dict(optimizer=dict(lr=0.001, weight_decay=0.01))

train_dataloader = dict(
    batch_size=2,
    sampler=dict(_delete_=True, type='InfiniteSampler', shuffle=True))

custom_hooks = [dict(type='DitHubPhaseHook', warmup_iters=1500)]

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=3000, max_keep_ckpts=1, save_best=None))

log_processor = dict(by_epoch=False)
