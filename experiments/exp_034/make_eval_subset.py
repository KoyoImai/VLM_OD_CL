#!/usr/bin/env python3
"""exp_034: 学習済みタスクだけを評価する config を生成する.

design.md §5 のとおり、各タスク終了後に「そこまでに学習済みのタスク」と ZCOCO を
評価する。ODinW 側は全13タスクを定義した odinw13_official_eval.py から、指定した
タスクだけを取り出した子 config を作る。

使い方:
    python experiments/exp_034/make_eval_subset.py <出力パス> <task> [<task> ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from odinw_official_tasks import CONFIG_ORDER  # noqa: E402

TMPL = '''\
# =============================================================================
# exp_034: ODinW 評価（学習済みタスクのみ）— 自動生成物。直接編集しない。
#   生成: experiments/exp_034/make_eval_subset.py
#   対象タスク（逐次順）: {names}
# =============================================================================
_base_ = './odinw13_official_eval.py'

dataset_prefixes = [
{prefixes}
]
datasets = [
{datasets}
]
metrics = [
{metrics}
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
'''


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out, names = sys.argv[1], sys.argv[2:]
    unknown = [n for n in names if n not in CONFIG_ORDER]
    if unknown:
        raise SystemExit(f'未知のタスク: {unknown}')
    body = TMPL.format(
        names=', '.join(names),
        prefixes='\n'.join(f'    {n!r},' for n in names),
        datasets='\n'.join(f'    _base_.dataset_{n},' for n in names),
        metrics='\n'.join(f'    _base_.val_evaluator_{n},' for n in names))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w') as f:
        f.write(body)
    print(f'generated {out}: {len(names)} tasks')


if __name__ == '__main__':
    main()
