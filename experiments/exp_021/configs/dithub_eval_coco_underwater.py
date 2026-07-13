# exp_021: DitHub checkpoint (underwater) 用の ZCOCO 評価 config
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
    dithub_classes=('pipe', 'fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray', 'peix', 'taca', 'echinus', 'holothurian', 'scallop', 'waterweeds', 'Arborescent', 'Caespitose-a', 'Caespitose-b', 'Columnar', 'Corymbose', 'Digitate', 'Encrusting', 'Foliose', 'Massive-Faviidae', 'Massive-Merulinidae', 'Massive-Mussidae', 'Massive-Poritidae', 'Solitary', 'Tabular'))
