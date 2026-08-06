# =============================================================================
# exp_034: ODinW-13 のタスク定義（公式 ZiRa 実装と同一の版・split）
#
# 公式 https://github.com/JarintotionDin/ZiRaGroundingDINO の
# groundingdino/config/configs/common/data/odinw/*.py で登録されているパスに合わせる。
# 本リポジトリの configs/mm_grounding_dino/odinw/…odinw13.py とは、
#   - 評価が valid ではなく test（PascalVOC のみ valid）
#   - AerialMaritimeDrone は large ではなく tiled
#   - NorthAmericaMushrooms は train=v1 / eval=v2 の train
#   - Packages は Raw ではなく augmented-v1
#   - Raccoon は v2-raw ではなく v38-416x416-resize
#   - ShellfishOpenImages は raw ではなく 416x416
# の点で異なる（design.md §3, §6.5(3)。2026-08-05 決定）。
#
# クラス名は odinw13 config のものと全タスクで一致することを実測確認済みなので流用する。
# 本ファイルは config ではなく、config から import される定数モジュールである。
# =============================================================================

DATA_ROOT = 'data/odinw/'

# name -> dict(root, train_ann, train_img, eval_ann, eval_img, classes)
ODINW13 = {
    'AerialMaritimeDrone': dict(
        root='AerialMaritimeDrone/tiled/',
        train_ann='train/annotations_without_background.json',
        train_img='train/',
        eval_ann='test/annotations_without_background.json',
        eval_img='test/',
        classes=('boat', 'car', 'dock', 'jetski', 'lift')),
    'Aquarium': dict(
        root='Aquarium/Aquarium Combined.v2-raw-1024.coco/',
        train_ann='train/annotations_without_background.json',
        train_img='train/',
        eval_ann='test/annotations_without_background.json',
        eval_img='test/',
        classes=('fish', 'jellyfish', 'penguin', 'puffin', 'shark',
                 'starfish', 'stingray')),
    'CottontailRabbits': dict(
        root='CottontailRabbits/',
        train_ann='train/annotations_without_background.json',
        train_img='train/',
        eval_ann='test/annotations_without_background.json',
        eval_img='test/',
        classes=('Cottontail-Rabbit', )),
    'EgoHands': dict(
        root='EgoHands/generic/',
        train_ann='train/annotations_without_background.json',
        train_img='train/',
        eval_ann='test/annotations_without_background.json',
        eval_img='test/',
        classes=('hand', )),
    # 公式は学習に v1 の train、評価に v2 の train を使う（v2 に test は無い）。
    'NorthAmericaMushrooms': dict(
        root='NorthAmericaMushrooms/',
        train_ann='North American Mushrooms.v1-416x416.coco/train/'
                  'annotations_without_background.json',
        train_img='North American Mushrooms.v1-416x416.coco/train/',
        eval_ann='North American Mushrooms.v2-416x416augmented.coco/train/'
                 'annotations_without_background.json',
        eval_img='North American Mushrooms.v2-416x416augmented.coco/train/',
        classes=('CoW', 'chanterelle')),
    'Packages': dict(
        root='Packages/augmented-v1/',
        train_ann='train/annotations_without_background.json',
        train_img='train/',
        eval_ann='test/annotations_without_background.json',
        eval_img='test/',
        classes=('package', )),
    # 公式も PascalVOC だけは valid で評価する。
    'PascalVOC': dict(
        root='PascalVOC/',
        train_ann='train/annotations_without_background.json',
        train_img='train/',
        eval_ann='valid/annotations_without_background.json',
        eval_img='valid/',
        classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                 'train', 'tvmonitor')),
    # pistols は export 直下に画像があり、ann だけ train_/test_ の接頭辞を持つ。
    'pistols': dict(
        root='pistols/export/',
        train_ann='train_annotations_without_background.json',
        train_img='',
        eval_ann='test_annotations_without_background.json',
        eval_img='',
        classes=('pistol', )),
    'pothole': dict(
        root='pothole/',
        train_ann='train/annotations_without_background.json',
        train_img='train/',
        eval_ann='test/annotations_without_background.json',
        eval_img='test/',
        classes=('pothole', )),
    'Raccoon': dict(
        root='Raccoon/Raccoon.v38-416x416-resize.coco/',
        train_ann='train/annotations_without_background.json',
        train_img='train/',
        eval_ann='test/annotations_without_background.json',
        eval_img='test/',
        classes=('raccoon', )),
    'ShellfishOpenImages': dict(
        root='ShellfishOpenImages/416x416/',
        train_ann='train/annotations_without_background.json',
        train_img='train/',
        eval_ann='test/annotations_without_background.json',
        eval_img='test/',
        classes=('Crab', 'Lobster', 'Shrimp')),
    'thermalDogsAndPeople': dict(
        root='thermalDogsAndPeople/',
        train_ann='train/annotations_without_background.json',
        train_img='train/',
        eval_ann='test/annotations_without_background.json',
        eval_img='test/',
        classes=('dog', 'person')),
    'VehiclesOpenImages': dict(
        root='VehiclesOpenImages/416x416/',
        train_ann='train/annotations_without_background.json',
        train_img='train/',
        eval_ann='test/annotations_without_background.json',
        eval_img='test/',
        classes=('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck')),
}

# odinw13 評価 config の記載順（逐次順のシャッフル元。design.md §4）
CONFIG_ORDER = [
    'AerialMaritimeDrone', 'Aquarium', 'CottontailRabbits', 'EgoHands',
    'NorthAmericaMushrooms', 'Packages', 'PascalVOC', 'pistols', 'pothole',
    'Raccoon', 'ShellfishOpenImages', 'thermalDogsAndPeople',
    'VehiclesOpenImages',
]


def task_order(seed: int = 42):
    """公式と同じく13タスクをランダム順に並べる（design.md §4）."""
    import random
    tasks = list(CONFIG_ORDER)
    random.Random(seed).shuffle(tasks)
    return tasks
