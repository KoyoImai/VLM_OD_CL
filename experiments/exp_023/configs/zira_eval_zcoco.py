# =============================================================================
# exp_023 ZiRa 評価用 / ZCOCO（COCO2017-val ゼロショット）
#   ZiRa モデル（全和 forward）で COCO2017-val を評価。RDB は全入力に一律適用され、
#   ZiL と Rep+ でゼロショット劣化を抑える設計。COCO データは eval_base_coco.py を継承。
#   既存コード無変更。
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/eval_base_coco.py'

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
