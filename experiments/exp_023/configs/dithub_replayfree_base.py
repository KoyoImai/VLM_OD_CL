# =============================================================================
# exp_023 比較手法: DitHub（リプレイフリー）共通ベース（新規・既存無変更）
#   fullft_replay_base（pretrain 継承・ODVG・num_classes=256・θ0・dn 有効）を土台に、
#   DitHub 検出器へ差し替え、学習スケジュール/lr を「論文（公式コード）準拠」に設定する。
#
#   論文準拠値（papers/DitHub_implementation_notes.md 5節 / exp_021 dithub_official）:
#     - IterBasedTrainLoop, タスクあたり 3000 iter
#     - AdamW lr 1e-3, weight_decay 1e-2（LoRA パラメータのみ学習）
#     - iter 1200 で lr ×0.1（MultiStepLR, by_epoch=False）
#     - warmup 1500 iter + specialization 1500 iter（iter 1500 で切替）
#     - grad clip は base（max_norm 0.1）を継承、追加損失なし
#   dn は有効のまま（他基準線と条件を揃える方針。exp_021 dithub_official と同じ判断。
#     num_classes=256 による label_embedding の正しいロードが引き続き効く）。
#   ckpt は last（by_epoch=False, keep 1, save_best なし）。in-training val なし。
#   dithub_classes（クラス別 LoRA 用）とデータ/バッチは各ドメイン差分 config で設定。
# =============================================================================
_base_ = './fullft_replay_base.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.layers.dithub_layers',
        'mmdet.models.detectors.dithub_grounding_dino',
        'mmdet.engine.hooks.dithub_phase_hook',
        'mmdet.engine.hooks.dithub_seq_phase_hook',  # 逐次学習の式3対応フェーズフック（新規）
        'mmdet.models.detectors.dithub_replay_grounding_dino',  # ODVG 対応 loss を持つ DitHubODVGGroundingDINO（新規）
    ],
    allow_failed_imports=False)

# dithub_classes は各ドメイン差分 config で与える（label_map の index 順）。
# type は ODVG のキャプション文字列 text に対応した loss を持つ版（既存無変更・新規サブクラス）。
model = dict(
    type='DitHubODVGGroundingDINO',
    encoder=dict(num_cp=0),
    encoder_cp=6)

# 論文準拠: AdamW lr 1e-3 / wd 1e-2
optim_wrapper = dict(optimizer=dict(lr=0.001, weight_decay=0.01))

# 論文準拠スケジュール: 3000 iter 固定、iter 1200 で lr×0.1
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

# フェーズ切替（iter 1500 = 学習量の半分で specialization へ）。
# 逐次の式3対応版フック。trained_classes（現ドメイン∩過去ドメインのクラス）を渡すと
# specialization 切替時に式3 fetch+merge を発火。既定は空（t=1,2 は重複なし）。
# t=3(videogames) は trained_classes=['person','car'] を各ドメイン config で上書きする。
custom_hooks = [
    dict(type='DitHubSeqPhaseHook', warmup_iters=1500, trained_classes=[])
]

# specialization 中は未選択クラスの A が不使用になるため必須
find_unused_parameters = True

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=3000,
        max_keep_ckpts=1,
        save_best=None),
    logger=dict(type='LoggerHook', interval=50))
log_processor = dict(by_epoch=False)
