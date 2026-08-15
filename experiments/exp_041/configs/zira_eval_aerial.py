# =============================================================================
# exp_041 ZiRa 評価用 / aerial valid
#
# 【自動生成】experiments/exp_041/gen_configs.py。直接編集しない。
#
#   ZiRa は RDB を全入力に一律適用（クラス選択なし）＋ Rep+ で LLRB に過去累積するため、
#   単一の ZiRa モデル（全和 forward = base + s*HLRB + LLRB）で現在・過去ドメインを評価できる。
#   データ・num_classes は exp_026 の eval_aerial.py を継承。既存コード無変更。
#   （前半3ドメインと ZCOCO は exp_023 の zira_eval_*.py をそのまま使う。design.md §4）
# =============================================================================
_base_ = '../../exp_026/configs/eval_aerial.py'

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
