# =============================================================================
# exp_034: ZCOCO 評価（COCO2017-val ゼロショット）/ ZiRa モデル
#   COCO の評価プロトコルは eval_base_coco.py（プロジェクト共通・exp_002.5 確定）を継承する。
#   モデル設定は exp_034 の学習と同一（ZiRa 全和 forward、dn 無効）。
#   θ0 の基準値は 0.504（exp_001 実測）。
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/eval_base_coco.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.layers.zira_layers',
        'mmdet.models.necks.zira_channel_mapper',
        'mmdet.models.detectors.zira_grounding_dino',
        'mmdet.models.dense_heads.nodn_grounding_dino_head',
    ],
    allow_failed_imports=False)

model = dict(
    type='ZiRaGroundingDINO',
    zil_loss_weight=0.1,
    use_dn=False,
    neck=dict(type='ZiRaChannelMapper'),
    bbox_head=dict(type='NoDNGroundingDINOHead'))
