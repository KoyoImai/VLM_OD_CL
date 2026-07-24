# =============================================================================
# exp_023 比較手法: ZiRa（リプレイフリー）共通ベース（新規・既存無変更）
#   fullft_replay_base（pretrain 継承・ODVG・num_classes=256・θ0・dn 有効）を土台に、
#   ZiRa 検出器へ差し替え、学習スケジュール/lr を「論文（公式コード）準拠」に設定する。
#
#   論文準拠値（papers/ZiRa_implementation_notes.md 6節 / exp_020 zira_official）:
#     - IterBasedTrainLoop, タスクあたり 2000 iter
#     - AdamW lr 1e-3, weight_decay 1e-4
#     - iter 800 で lr ×0.1（MultiStepLR, by_epoch=False）
#     - LLRB（名前に llrb を含む param）のみ lr×0.2（η=0.2）、λ=0.1
#     - grad clip は base（max_norm 0.1）を継承
#   dn は有効のまま（他基準線と条件を揃え、スケジュール変数のみ論文準拠にする方針。
#     exp_020 zira_official と同じ判断。dn 有効なので num_classes=256 による
#     label_embedding の正しいロードが引き続き効く）。
#   ckpt は last（by_epoch=False, keep 1, save_best なし）。in-training val なし。
#   バッチ/データは各ドメイン差分 config で設定（batch_size=4：+replay の現在4/GPU と
#     現在ドメインの投入枚数を一致させ、差分をリプレイ（バッファ追加）だけにする）。
# =============================================================================
_base_ = './fullft_replay_base.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.layers.zira_layers',
        'mmdet.models.necks.zira_channel_mapper',
        'mmdet.models.detectors.zira_grounding_dino',
    ],
    allow_failed_imports=False)

model = dict(
    type='ZiRaGroundingDINO',
    zil_loss_weight=0.1,
    neck=dict(type='ZiRaChannelMapper'))

# 論文準拠: AdamW lr 1e-3 / wd 1e-4、LLRB のみ lr×0.2
optim_wrapper = dict(
    optimizer=dict(lr=0.001, weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys={
        'llrb': dict(lr_mult=0.2),
    }))

# 論文準拠スケジュール: 2000 iter 固定、iter 800 で lr×0.1
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

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=1,
        save_best=None),
    logger=dict(type='LoggerHook', interval=50))
log_processor = dict(by_epoch=False)
