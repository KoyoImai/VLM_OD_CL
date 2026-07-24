# =============================================================================
# exp_024: リプレイ無し逐次ファインチューニング（条件A＝全モジュール）の共通ベース
#   （新規ファイル・既存無変更）
#
#   - 事前学習 config を継承（ODVG 対応モデル num_classes=256、RandomSamplingNegPos
#     入りの train_pipeline、tokenizer 名を引き継ぐ）。
#   - 条件A（全モジュール学習）: 事前学習の paramwise_cfg（backbone/language lr_mult=0.1）
#     をそのまま使い、lr のみ 1e-4、schedule を 20 epoch / milestone[15] に。
#   - **リプレイ無し**: バッファを混ぜない。学習データは現在ドメインのみ（各ドメイン config で定義）。
#   - seed=0。適応 mAP は last を採用（in-training val は使わず、評価は逐次ドライバが実施）。
#
#   exp_023 の fullft_replay_base.py との差分は次の3点のみ:
#     (1) リプレイ用の CurrentEpochMultiSourceSampler を import しない（混合を使わないため）
#     (2) backbone.init_cfg=None（下記）
#     (3) 学習データが現在ドメイン単独（各ドメイン config 側）
#
# 参照: [[../design]]
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'  # noqa

# numpy 互換シム（既存コード無変更）: 既存の RandomSamplingNegPos が使う `np.long` は
# numpy>=1.24 で削除された（現環境 1.26）。リポジトリルートの `exp023_np_compat` で
# 削除された別名を復元し、custom_imports で読み込む。main プロセスで適用され、
# fork される dataloader worker にも継承される。既存の mmdet コードは変更しない。
custom_imports = dict(
    imports=['exp023_np_compat'],
    allow_failed_imports=False)

# θ0（事前学習チェックポイント）から開始（t=1 用。t>=2 はドライバが前ドメイン last に差し替え）
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa

# backbone の init_cfg を無効化（design.md §7 の決定・2026-07-24）。
#   事前学習 config は ImageNet 事前学習 Swin-T（swin_tiny_patch4_window7_224.pth）を
#   init_weights() で読むが、その直後の load_from（θ0 or 前ドメイン last）が backbone の
#   187 パラメータを全て上書きするため、最終的な重みには影響しない（検証: init_cfg 有無で
#   全 908 テンソル完全一致）。無効化により毎ジョブの不要なダウンロード・読み込みを省く。
model = dict(backbone=dict(init_cfg=None))

# 条件A（全モジュール学習）: 事前学習の paramwise_cfg（backbone/language lr_mult=0.1）を
# 継承し、lr のみ 1e-4。
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
