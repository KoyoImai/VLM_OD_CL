# =============================================================================
# exp_038: ODinW 評価（学習済みタスクのみ）— 自動生成物。直接編集しない。
#   生成: experiments/exp_038/gen_configs.py
#   時点: t=12（直前に学習したタスク: Aquarium）
#   対象タスク（逐次順）: pistols / PascalVOC / CottontailRabbits / Raccoon / VehiclesOpenImages / Packages / thermalDogsAndPeople / pothole / EgoHands / NorthAmericaMushrooms / AerialMaritimeDrone / Aquarium
# =============================================================================
_base_ = './odinw13_plain_eval.py'

dataset_prefixes = [
    'pistols',
    'PascalVOC',
    'CottontailRabbits',
    'Raccoon',
    'VehiclesOpenImages',
    'Packages',
    'thermalDogsAndPeople',
    'pothole',
    'EgoHands',
    'NorthAmericaMushrooms',
    'AerialMaritimeDrone',
    'Aquarium',
]
datasets = [
    _base_.dataset_pistols,
    _base_.dataset_PascalVOC,
    _base_.dataset_CottontailRabbits,
    _base_.dataset_Raccoon,
    _base_.dataset_VehiclesOpenImages,
    _base_.dataset_Packages,
    _base_.dataset_thermalDogsAndPeople,
    _base_.dataset_pothole,
    _base_.dataset_EgoHands,
    _base_.dataset_NorthAmericaMushrooms,
    _base_.dataset_AerialMaritimeDrone,
    _base_.dataset_Aquarium,
]
metrics = [
    _base_.val_evaluator_pistols,
    _base_.val_evaluator_PascalVOC,
    _base_.val_evaluator_CottontailRabbits,
    _base_.val_evaluator_Raccoon,
    _base_.val_evaluator_VehiclesOpenImages,
    _base_.val_evaluator_Packages,
    _base_.val_evaluator_thermalDogsAndPeople,
    _base_.val_evaluator_pothole,
    _base_.val_evaluator_EgoHands,
    _base_.val_evaluator_NorthAmericaMushrooms,
    _base_.val_evaluator_AerialMaritimeDrone,
    _base_.val_evaluator_Aquarium,
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
