# exp_021: DitHub checkpoint (aerial) 用の ZCOCO 評価 config
# per_class_lora_A のキー集合を学習時と一致させるため、ドメインごとに
# クラス一覧を静的に埋め込む (生成元: 学習 config の class_name)。
_base_ = '../../../configs/mm_grounding_dino/eval_base_coco.py'

custom_imports = dict(
    imports=[
        'mmdet.models.layers.dithub_layers',
        'mmdet.models.detectors.dithub_grounding_dino',
        'mmdet.engine.hooks.dithub_phase_hook',
    ],
    allow_failed_imports=False)

model = dict(
    type='DitHubGroundingDINO',
    dithub_classes=('black-hat', 'bodysurface', 'bodyunder', 'umpire', 'white-hat', 'chain', 'green_sphero', 'orange-sphero', 'orange_sphero', 'purple_sphero', 'red_sphero', 'yellow_sphero', 'football', 'player', 'referee', 'crop', 'weed', 'cow', 'Fish', 'Flower', 'Gravel', 'Sugar'))
