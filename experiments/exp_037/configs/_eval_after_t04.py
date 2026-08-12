# =============================================================================
# exp_037: ODinW 評価（学習済みタスクのみ）— 自動生成物。直接編集しない。
#   生成: experiments/exp_037/gen_configs.py
#   時点: t=04（直前に学習したタスク: Raccoon）
#   対象タスク（逐次順）: pistols / PascalVOC / CottontailRabbits / Raccoon
# =============================================================================
_base_ = './dithub_odinw13_eval.py'

dataset_prefixes = [
    'pistols',
    'PascalVOC',
    'CottontailRabbits',
    'Raccoon',
]
datasets = [
    _base_.dataset_pistols,
    _base_.dataset_PascalVOC,
    _base_.dataset_CottontailRabbits,
    _base_.dataset_Raccoon,
]
metrics = [
    _base_.val_evaluator_pistols,
    _base_.val_evaluator_PascalVOC,
    _base_.val_evaluator_CottontailRabbits,
    _base_.val_evaluator_Raccoon,
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
