#!/usr/bin/env python3
"""exp_038: 素の LoRA / ODinW-13 の config を生成する（自己完結・リテラルのパスで書き出す）.

mmengine の config は実行時に __file__ を持たないため、共通テーブルを import する
書き方ができない。そこで本スクリプトでタスク定義から config を生成し、生成物だけを
実験で使う（exp_034 / exp_037 と同じ方式）。

タスク定義は exp_034 の odinw_official_tasks.py をそのまま使う。データの版・split・
逐次順を exp_034（ZiRa）/ exp_037（DitHub）と完全に一致させるためで、複製しない
（design.md §2.4）。

評価は **plain な GroundingDINO** で行う。exp_038 は各タスク後に LoRA を base へ焼き込む
ため、評価対象の ckpt が plain 構造だからである（design.md §2.3 / §5）。これは exp_034
（ZiRa 検出器で評価）・exp_037（DitHub 検出器で評価）に対する独立な評価経路になるので、
t=0 で θ0 を測って exp_034 の値と一致することを確認する。

生成するもの:
    configs/odinw13_plain_eval.py           全13タスクの評価 config（plain GroundingDINO）
    configs/zcoco_eval.py                   ZCOCO 評価 config（plain GroundingDINO）
    configs/lora_odinw13_<task>.py          学習 config × 13
    configs/_eval_after_t{01..13}.py        学習済みタスクだけを評価する部分集合 × 13

使い方: python experiments/exp_038/gen_configs.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'exp_034'))
from odinw_official_tasks import (CONFIG_ORDER, DATA_ROOT,  # noqa: E402
                                  ODINW13, task_order)

CFG_DIR = os.path.join(HERE, 'configs')
SEED = 42
ORDER = task_order(SEED)

EVAL_HEADER = '''\
# =============================================================================
# exp_038: ODinW-13 評価 config（plain GroundingDINO / 公式と同一の版・split）
#
# 【自動生成】experiments/exp_038/gen_configs.py が
# experiments/exp_034/odinw_official_tasks.py から生成する。直接編集しない。
#
# データの版・split・クラス名は exp_034（ZiRa）/ exp_037（DitHub）と完全に同一。
# 差分の一覧は exp_034/design.md §3 を参照。
#
# exp_038 は各タスク後に LoRA を base へ焼き込むため、評価対象は plain な ckpt である。
# したがってモデルは事前学習 config のまま（LoRA も DitHub も ZiRa も挟まない）。
# NoDNGroundingDINOHead は損失側の split_outputs のみを上書きする実装で推論経路は
# 素の GroundingDINOHead と同一なため、評価では素のまま使う。
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'  # noqa

dataset_type = 'CocoDataset'
data_root = '{data_root}'

base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'caption_prompt')
'''

EVAL_TASK = '''
# --------------------- {idx} {name} ---------------------#
class_name = {classes}
metainfo = dict(classes=class_name)
_data_root = data_root + {root!r}
dataset_{name} = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file={ann!r},
    data_prefix=dict(img={img!r}),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_{name} = dict(
    type='CocoMetric',
    ann_file=_data_root + {ann!r},
    metric='bbox')
'''

EVAL_FOOTER = '''
# --------------------- Config ---------------------#
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

ZCOCO_TMPL = '''\
# =============================================================================
# exp_038: ZCOCO 評価（COCO2017-val ゼロショット）/ plain GroundingDINO
#   COCO の評価プロトコルは eval_base_coco.py（プロジェクト共通・exp_002.5 確定）を継承。
#   評価対象は LoRA をマージ済みの plain ckpt なので、モデルは事前学習 config のまま。
#   θ0 の基準値は 0.5040（exp_001 実測）。
#
# 【自動生成】experiments/exp_038/gen_configs.py。直接編集しない。
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/eval_base_coco.py'
'''

TRAIN_TMPL = '''\
# =============================================================================
# exp_038: 素の LoRA 学習 config / t={t:02d} {name}
#
# 【自動生成】experiments/exp_038/gen_configs.py が
# experiments/exp_034/odinw_official_tasks.py から生成する。直接編集しない。
#
# スケジュール・lr・optimizer・LoRA の挿入箇所は lora_odinw13_base.py を継承し、
# 本 config はデータとクラスだけを定義する。学習データは公式実装と同じ版の train split。
#   画像 {n_img} 枚 / box {n_box} / クラス {n_cls}（3000 iter × batch 2 = 6000 枚で {epochs} 周）
#
# 逐次学習では前タスクのマージ済み ckpt（θ_{{t-1}}）を load_from に上書きする
# （ドライバが実施）。LoRA は毎タスク B=0 から引き直すので、学習開始時点は θ_{{t-1}} と
# 厳密に等価である（design.md §2.3）。
# 逐次順は seed {seed}（exp_034 / exp_037 と同一）。
# =============================================================================
_base_ = './{base}'

class_name = {classes}
metainfo = dict(classes=class_name)
_data_root = 'data/odinw/' + {root!r}

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=_data_root,
        metainfo=metainfo,
        ann_file={ann!r},
        data_prefix=dict(img={img!r}),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        return_classes=True,
        pipeline=_base_.train_pipeline))
'''

SUBSET_TMPL = '''\
# =============================================================================
# exp_038: ODinW 評価（学習済みタスクのみ）— 自動生成物。直接編集しない。
#   生成: experiments/exp_038/gen_configs.py
#   時点: t={t:02d}（直前に学習したタスク: {last}）
#   対象タスク（逐次順）: {names}
# =============================================================================
_base_ = './odinw13_plain_eval.py'

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


def fmt_classes(classes, indent=14):
    if len(classes) == 1:
        return f'({classes[0]!r}, )'
    body = ', '.join(repr(c) for c in classes)
    if len(body) <= 68:
        return f'({body})'
    out, line = [], ''
    for c in classes:
        item = repr(c) + ', '
        if len(line) + len(item) > 68:
            out.append(line.rstrip())
            line = ''
        line += item
    out.append(line.rstrip().rstrip(','))
    return '(' + ('\n' + ' ' * indent).join(out) + ')'


def gen_eval():
    parts = [EVAL_HEADER.format(data_root=DATA_ROOT)]
    for i, name in enumerate(CONFIG_ORDER, 1):
        t = ODINW13[name]
        parts.append(EVAL_TASK.format(
            idx=i, name=name, classes=fmt_classes(t['classes']),
            root=t['root'], ann=t['eval_ann'], img=t['eval_img']))
    parts.append(EVAL_FOOTER.format(
        prefixes='\n'.join(f'    {n!r},' for n in CONFIG_ORDER),
        datasets='\n'.join(f'    dataset_{n},' for n in CONFIG_ORDER),
        metrics='\n'.join(f'    val_evaluator_{n},' for n in CONFIG_ORDER)))
    path = os.path.join(CFG_DIR, 'odinw13_plain_eval.py')
    with open(path, 'w') as f:
        f.write(''.join(parts))
    print('generated', path, f'({len(CONFIG_ORDER)} タスク)')

    path = os.path.join(CFG_DIR, 'zcoco_eval.py')
    with open(path, 'w') as f:
        f.write(ZCOCO_TMPL)
    print('generated', path)


def gen_train(prefix='lora_odinw13', base='lora_odinw13_base.py'):
    """学習 config を 13 本生成する。

    prefix / base を変えると別条件の config 群を出せる（design.md §10）。
        条件A: prefix='lora_odinw13'       base='lora_odinw13_base.py'       (232 層)
        条件B: prefix='lora_odinw13_condB' base='lora_odinw13_condB_base.py' (160 層)
    データ・クラス・逐次順は条件によらず同一。
    """
    for i, name in enumerate(ORDER, 1):
        t = ODINW13[name]
        classes = t['classes']
        ann_path = os.path.join('data/odinw', t['root'], t['train_ann'])
        with open(ann_path) as f:
            j = json.load(f)
        n_img, n_box = len(j['images']), len(j['annotations'])
        body = TRAIN_TMPL.format(
            t=i, name=name, seed=SEED, classes=fmt_classes(classes), base=base,
            root=t['root'], ann=t['train_ann'], img=t['train_img'],
            n_img=f'{n_img:,}', n_box=f'{n_box:,}', n_cls=len(classes),
            epochs=f'{6000 / max(n_img, 1):.1f}')
        path = os.path.join(CFG_DIR, f'{prefix}_{name}.py')
        with open(path, 'w') as f:
            f.write(body)
        print(f'generated {path}  (t={i:02d}, {len(classes)} cls, {n_img} imgs)')


def gen_subsets():
    for i in range(1, len(ORDER) + 1):
        learned = ORDER[:i]
        body = SUBSET_TMPL.format(
            t=i, last=ORDER[i - 1], names=' / '.join(learned),
            prefixes='\n'.join(f'    {n!r},' for n in learned),
            datasets='\n'.join(f'    _base_.dataset_{n},' for n in learned),
            metrics='\n'.join(
                f'    _base_.val_evaluator_{n},' for n in learned))
        path = os.path.join(CFG_DIR, f'_eval_after_t{i:02d}.py')
        with open(path, 'w') as f:
            f.write(body)
    print(f'generated {len(ORDER)} 個の _eval_after_t*.py')


if __name__ == '__main__':
    os.makedirs(CFG_DIR, exist_ok=True)
    print('逐次順 (seed', SEED, '):', ' -> '.join(ORDER))
    gen_eval()
    gen_train()                                        # 条件A（232 層）
    gen_train('lora_odinw13_condB',
              'lora_odinw13_condB_base.py')            # 条件B（160 層）
    gen_train('lora_odinw13_condC',
              'lora_odinw13_condC_base.py')            # 条件C（232 層 / lr 1e-4）
    gen_subsets()
