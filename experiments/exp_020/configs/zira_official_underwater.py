# exp_020 対照 run: ZiRa を公式実装のハイパラで学習する (underwater のみ)
#
# 目的: 実装のエンドツーエンド検証。公式スケジュールで ZCOCO 低下が論文並み
# (~1 ポイント級) に収まれば「実装は忠実、20 epoch 版との差は学習量の体制差」
# が確定する (design.md 追記 2026-07-11)。
#
# 公式に合わせる項目 (papers/ZiRa_implementation_notes.md 6 節):
#   2000 iter 固定 / total batch 2 (1 GPU) / AdamW lr 1e-3, wd 1e-4 /
#   iter 800 で lr x0.1 / grad clip max_norm 0.1 / η=0.2 / λ=0.1
# 公式に合わせない項目: dn は有効のまま (20 epoch 版・全基準線と共通の条件、
#   本対照の目的はスケジュール変数の分離)。seed=0 (プロジェクト規約)。
_base_ = './zira_underwater.py'

# --- 公式スケジュール: iter ベース 2000 iter ---
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
