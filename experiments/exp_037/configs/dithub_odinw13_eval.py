# =============================================================================
# exp_037: ODinW-13 評価 config（DitHub モデル / 公式と同一の版・split）
#
# 【自動生成】experiments/exp_037/gen_configs.py が
# experiments/exp_034/odinw_official_tasks.py から生成する。直接編集しない。
#
# データの版・split・クラス名は exp_034（ZiRa）と完全に同一。差分の一覧は
# exp_034/design.md §3 と exp_037/design.md §5.2 を参照。
#
# dithub_classes は全13タスクのクラスの和集合（43 クラス）。ライブラリ ckpt の
# per_class_lora_A を漏れなく読み込むためにスロットを全確保する。評価時に合成へ
# 使われるのは ckpt に含まれるクラス（= 学習済み）だけである（design.md §1.3）。
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'  # noqa

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.layers.dithub_layers',
        'mmdet.models.detectors.dithub_grounding_dino',
        'mmdet.models.dense_heads.nodn_grounding_dino_head',
    ],
    allow_failed_imports=False)

model = dict(
    type='DitHubGroundingDINO',
    use_dn=False,
    encoder_cp=0,
    dithub_classes=('aeroplane', 'Ambulance', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'chanterelle', 'Cottontail-Rabbit',
        'CoW', 'Crab', 'diningtable', 'dock', 'dog', 'fish', 'hand',
        'horse', 'jellyfish', 'jetski', 'lift', 'Lobster', 'motorbike',
        'Motorcycle', 'package', 'penguin', 'person', 'pistol', 'pothole',
        'pottedplant', 'puffin', 'raccoon', 'shark', 'sheep', 'Shrimp',
        'sofa', 'starfish', 'stingray', 'train', 'Truck', 'tvmonitor'),
    bbox_head=dict(type='NoDNGroundingDINOHead'))

dataset_type = 'CocoDataset'
data_root = 'data/odinw/'

base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'caption_prompt')

# --------------------- 1 AerialMaritimeDrone ---------------------#
class_name = ('boat', 'car', 'dock', 'jetski', 'lift')
metainfo = dict(classes=class_name)
_data_root = data_root + 'AerialMaritimeDrone/tiled/'
dataset_AerialMaritimeDrone = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='test/annotations_without_background.json',
    data_prefix=dict(img='test/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_AerialMaritimeDrone = dict(
    type='CocoMetric',
    ann_file=_data_root + 'test/annotations_without_background.json',
    metric='bbox')

# --------------------- 2 Aquarium ---------------------#
class_name = ('fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish',
              'stingray')
metainfo = dict(classes=class_name)
_data_root = data_root + 'Aquarium/Aquarium Combined.v2-raw-1024.coco/'
dataset_Aquarium = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='test/annotations_without_background.json',
    data_prefix=dict(img='test/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_Aquarium = dict(
    type='CocoMetric',
    ann_file=_data_root + 'test/annotations_without_background.json',
    metric='bbox')

# --------------------- 3 CottontailRabbits ---------------------#
class_name = ('Cottontail-Rabbit', )
metainfo = dict(classes=class_name)
_data_root = data_root + 'CottontailRabbits/'
dataset_CottontailRabbits = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='test/annotations_without_background.json',
    data_prefix=dict(img='test/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_CottontailRabbits = dict(
    type='CocoMetric',
    ann_file=_data_root + 'test/annotations_without_background.json',
    metric='bbox')

# --------------------- 4 EgoHands ---------------------#
class_name = ('hand', )
metainfo = dict(classes=class_name)
_data_root = data_root + 'EgoHands/generic/'
dataset_EgoHands = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='test/annotations_without_background.json',
    data_prefix=dict(img='test/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_EgoHands = dict(
    type='CocoMetric',
    ann_file=_data_root + 'test/annotations_without_background.json',
    metric='bbox')

# --------------------- 5 NorthAmericaMushrooms ---------------------#
class_name = ('CoW', 'chanterelle')
metainfo = dict(classes=class_name)
_data_root = data_root + 'NorthAmericaMushrooms/'
dataset_NorthAmericaMushrooms = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='North American Mushrooms.v2-416x416augmented.coco/train/annotations_without_background.json',
    data_prefix=dict(img='North American Mushrooms.v2-416x416augmented.coco/train/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_NorthAmericaMushrooms = dict(
    type='CocoMetric',
    ann_file=_data_root + 'North American Mushrooms.v2-416x416augmented.coco/train/annotations_without_background.json',
    metric='bbox')

# --------------------- 6 Packages ---------------------#
class_name = ('package', )
metainfo = dict(classes=class_name)
_data_root = data_root + 'Packages/augmented-v1/'
dataset_Packages = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='test/annotations_without_background.json',
    data_prefix=dict(img='test/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_Packages = dict(
    type='CocoMetric',
    ann_file=_data_root + 'test/annotations_without_background.json',
    metric='bbox')

# --------------------- 7 PascalVOC ---------------------#
class_name = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
              'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
              'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
metainfo = dict(classes=class_name)
_data_root = data_root + 'PascalVOC/'
dataset_PascalVOC = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='valid/annotations_without_background.json',
    data_prefix=dict(img='valid/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_PascalVOC = dict(
    type='CocoMetric',
    ann_file=_data_root + 'valid/annotations_without_background.json',
    metric='bbox')

# --------------------- 8 pistols ---------------------#
class_name = ('pistol', )
metainfo = dict(classes=class_name)
_data_root = data_root + 'pistols/export/'
dataset_pistols = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='test_annotations_without_background.json',
    data_prefix=dict(img=''),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_pistols = dict(
    type='CocoMetric',
    ann_file=_data_root + 'test_annotations_without_background.json',
    metric='bbox')

# --------------------- 9 pothole ---------------------#
class_name = ('pothole', )
metainfo = dict(classes=class_name)
_data_root = data_root + 'pothole/'
dataset_pothole = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='test/annotations_without_background.json',
    data_prefix=dict(img='test/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_pothole = dict(
    type='CocoMetric',
    ann_file=_data_root + 'test/annotations_without_background.json',
    metric='bbox')

# --------------------- 10 Raccoon ---------------------#
class_name = ('raccoon', )
metainfo = dict(classes=class_name)
_data_root = data_root + 'Raccoon/Raccoon.v38-416x416-resize.coco/'
dataset_Raccoon = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='test/annotations_without_background.json',
    data_prefix=dict(img='test/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_Raccoon = dict(
    type='CocoMetric',
    ann_file=_data_root + 'test/annotations_without_background.json',
    metric='bbox')

# --------------------- 11 ShellfishOpenImages ---------------------#
class_name = ('Crab', 'Lobster', 'Shrimp')
metainfo = dict(classes=class_name)
_data_root = data_root + 'ShellfishOpenImages/416x416/'
dataset_ShellfishOpenImages = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='test/annotations_without_background.json',
    data_prefix=dict(img='test/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_ShellfishOpenImages = dict(
    type='CocoMetric',
    ann_file=_data_root + 'test/annotations_without_background.json',
    metric='bbox')

# --------------------- 12 thermalDogsAndPeople ---------------------#
class_name = ('dog', 'person')
metainfo = dict(classes=class_name)
_data_root = data_root + 'thermalDogsAndPeople/'
dataset_thermalDogsAndPeople = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='test/annotations_without_background.json',
    data_prefix=dict(img='test/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_thermalDogsAndPeople = dict(
    type='CocoMetric',
    ann_file=_data_root + 'test/annotations_without_background.json',
    metric='bbox')

# --------------------- 13 VehiclesOpenImages ---------------------#
class_name = ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck')
metainfo = dict(classes=class_name)
_data_root = data_root + 'VehiclesOpenImages/416x416/'
dataset_VehiclesOpenImages = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='test/annotations_without_background.json',
    data_prefix=dict(img='test/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_VehiclesOpenImages = dict(
    type='CocoMetric',
    ann_file=_data_root + 'test/annotations_without_background.json',
    metric='bbox')

# --------------------- Config ---------------------#
dataset_prefixes = [
    'AerialMaritimeDrone',
    'Aquarium',
    'CottontailRabbits',
    'EgoHands',
    'NorthAmericaMushrooms',
    'Packages',
    'PascalVOC',
    'pistols',
    'pothole',
    'Raccoon',
    'ShellfishOpenImages',
    'thermalDogsAndPeople',
    'VehiclesOpenImages',
]
datasets = [
    dataset_AerialMaritimeDrone,
    dataset_Aquarium,
    dataset_CottontailRabbits,
    dataset_EgoHands,
    dataset_NorthAmericaMushrooms,
    dataset_Packages,
    dataset_PascalVOC,
    dataset_pistols,
    dataset_pothole,
    dataset_Raccoon,
    dataset_ShellfishOpenImages,
    dataset_thermalDogsAndPeople,
    dataset_VehiclesOpenImages,
]
metrics = [
    val_evaluator_AerialMaritimeDrone,
    val_evaluator_Aquarium,
    val_evaluator_CottontailRabbits,
    val_evaluator_EgoHands,
    val_evaluator_NorthAmericaMushrooms,
    val_evaluator_Packages,
    val_evaluator_PascalVOC,
    val_evaluator_pistols,
    val_evaluator_pothole,
    val_evaluator_Raccoon,
    val_evaluator_ShellfishOpenImages,
    val_evaluator_thermalDogsAndPeople,
    val_evaluator_VehiclesOpenImages,
]

val_dataloader = dict(
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=datasets))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)
test_evaluator = val_evaluator
