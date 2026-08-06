# =============================================================================
# exp_034: ODinW 評価（学習済みタスクのみ）— 自動生成物。直接編集しない。
#   生成: experiments/exp_034/make_eval_subset.py
#   対象タスク（逐次順）: pistols
# =============================================================================
_base_ = './odinw13_official_eval.py'

dataset_prefixes = [
    'pistols',
]
datasets = [
    _base_.dataset_pistols,
]
metrics = [
    _base_.val_evaluator_pistols,
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
