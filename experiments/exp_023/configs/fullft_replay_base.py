# =============================================================================
# exp_023: full finetuning + リプレイ の共通ベース（新規ファイル・既存無変更）
#   - 事前学習 config を継承（ODVG 対応モデル num_classes=256、RandomSamplingNegPos
#     入りの train_pipeline、tokenizer 名を引き継ぐ）。
#   - full FT: 事前学習の backbone/language lr_mult=0.1（＝全モジュール学習）をそのまま使い、
#     lr のみ 1e-4、schedule を 20 epoch / milestone[15] に。
#   - 学習データ（混合）と val は各ドメインの差分 config で定義する。
#   - seed=0。適応 mAP は last を採用（in-training val は使わず、評価は逐次ドライバが実施）。
# 参照: [[../design]] / [[../implementation_plan]]
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'  # noqa

# numpy 互換シム（既存コード無変更）: 既存の RandomSamplingNegPos が使う `np.long` は
# numpy>=1.24 で削除された（現環境 1.26）。別モジュール `exp023_np_compat`（リポジトリ
# ルート）で削除された別名を復元し、custom_imports で読み込む。main プロセスで適用され、
# fork される dataloader worker にも継承される。既存の mmdet コードは変更しない。
custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.datasets.samplers.current_epoch_multi_source_sampler',
    ],
    allow_failed_imports=False)

# θ0（事前学習チェックポイント）から開始（t=1 用。t>=2 はドライバが前ドメイン last に差し替え）
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa

# full finetuning: 事前学習の paramwise_cfg（backbone/language lr_mult=0.1）を継承し、lr のみ 1e-4。
optim_wrapper = dict(optimizer=dict(lr=0.0001))

# スケジュール（全ドメイン統一）
max_epochs = 20
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=max_epochs + 1)  # in-training val は行わない（評価はドライバ）

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=-1),
    logger=dict(type='LoggerHook', interval=50))

randomness = dict(deterministic=False, seed=0)
