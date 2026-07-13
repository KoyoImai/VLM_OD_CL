# exp_021: DitHub checkpoint (electromagnetic) 用の ZCOCO 評価 config
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
    dithub_classes=('dog', 'person', 'Cell', 'Cell-Multi', 'No-Anomaly', 'Shadowing', 'Unclassified', 'stray', 'target', 'cheetah', 'human', 'artefact', 'distal phalanges', 'fifth metacarpal bone', 'first metacarpal bone', 'fourth metacarpal bone', 'intermediate phalanges', 'proximal phalanges', 'radius', 'second metacarpal bone', 'soft tissue calcination', 'third metacarpal bone', 'ulna', 'acl', '0', 'negative', 'positive', '6W', '7W', 'EH', 'label0', 'label1', 'label2', 'angle', 'fracture', 'line', 'messed_up_angle', 'bicycle', 'car'))
