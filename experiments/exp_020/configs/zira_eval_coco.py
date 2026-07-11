# exp_020: ZiRa checkpoint 用の ZCOCO 評価 config
# eval_base_coco.py と同一の評価設定で、モデル type だけ ZiRa 版に差し替える
# (RDB キーを含む checkpoint を読むため)。評価 forward は全和 = 融合後と等価。
_base_ = '../../../configs/mm_grounding_dino/eval_base_coco.py'

custom_imports = dict(
    imports=[
        'mmdet.models.layers.zira_layers',
        'mmdet.models.necks.zira_channel_mapper',
        'mmdet.models.detectors.zira_grounding_dino',
    ],
    allow_failed_imports=False)

model = dict(
    type='ZiRaGroundingDINO',
    neck=dict(type='ZiRaChannelMapper'))
