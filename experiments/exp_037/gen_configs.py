#!/usr/bin/env python3
"""exp_037: DitHub / ODinW-13 の config を生成する（自己完結・リテラルのパスで書き出す）.

mmengine の config は実行時に __file__ を持たないため、共通テーブルを import する
書き方ができない。そこで本スクリプトでタスク定義から config を生成し、生成物だけを
実験で使う（exp_034 と同じ方式）。

タスク定義は exp_034 の odinw_official_tasks.py をそのまま使う。データの版・split・
逐次順を exp_034（ZiRa）と完全に一致させるためで、複製しない（design.md §4.2）。

生成するもの:
    configs/dithub_odinw13_eval.py          全13タスクの評価 config（46クラスの DitHub）
    configs/zcoco_eval.py                   ZCOCO 評価 config（46クラスの DitHub）
    configs/dithub_odinw13_<task>.py        学習 config × 13
    configs/_eval_after_t{01..13}.py        学習済みタスクだけを評価する部分集合 × 13

使い方: python experiments/exp_037/gen_configs.py
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

# LoRA ライブラリのスロット（= 全タスクのクラスの和集合）。canonical_key は
# 小文字化するので 'Bus' と 'bus' は同じスロットになる。評価モデルはこれを全部確保し、
# 実際に合成へ使うのはライブラリ ckpt に載っているクラスだけ（design.md §1.3）。
_seen = {}
for _t in CONFIG_ORDER:
    for _c in ODINW13[_t]['classes']:
        _seen.setdefault(_c.lower(), _c)
ALL_CLASSES = tuple(_seen[k] for k in sorted(_seen))

MODEL_BLOCK = '''\
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
    dithub_classes={classes},
    bbox_head=dict(type='NoDNGroundingDINOHead'))
'''

EVAL_HEADER = '''\
# =============================================================================
# exp_037: ODinW-13 評価 config（DitHub モデル / 公式と同一の版・split）
#
# 【自動生成】experiments/exp_037/gen_configs.py が
# experiments/exp_034/odinw_official_tasks.py から生成する。直接編集しない。
#
# データの版・split・クラス名は exp_034（ZiRa）と完全に同一。差分の一覧は
# exp_034/design.md §3 と exp_037/design.md §5.2 を参照。
#
# dithub_classes は全13タスクのクラスの和集合（{n_cls} クラス）。ライブラリ ckpt の
# per_class_lora_A を漏れなく読み込むためにスロットを全確保する。評価時に合成へ
# 使われるのは ckpt に含まれるクラス（= 学習済み）だけである（design.md §1.3）。
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'  # noqa

{model}
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
# exp_037: ZCOCO 評価（COCO2017-val ゼロショット）/ DitHub モデル
#   COCO の評価プロトコルは eval_base_coco.py（プロジェクト共通・exp_002.5 確定）を継承。
#   モデル設定は exp_037 の学習と同一（dn 無効、ライブラリ {n_cls} スロット）。
#   COCO 80 クラスのうちライブラリにモジュールを持つものだけに ΔW が当たり、
#   他は凍結モデル（θ0）を通る。θ0 の基準値は 0.5040（exp_001 実測）。
#
# 【自動生成】experiments/exp_037/gen_configs.py。直接編集しない。
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/eval_base_coco.py'

{model}'''

TRAIN_TMPL = '''\
# =============================================================================
# exp_037: DitHub 学習 config / t={t:02d} {name}
#
# 【自動生成】experiments/exp_037/gen_configs.py が
# experiments/exp_034/odinw_official_tasks.py から生成する。直接編集しない。
#
# スケジュール・lr・optimizer・DitHub の設定は dithub_odinw13_base.py（公式準拠）を
# 継承し、本 config はデータとクラスだけを定義する。学習データは公式実装と同じ版の
# train split。
#   画像 {n_img} 枚 / box {n_box} / クラス {n_cls}（3000 iter × batch 2 = 6000 枚で {epochs} 周）
#
# dithub_classes は当該タスクのクラスのみ。公式は全タスク分のスロットを最初から確保するが、
# B=0 と enable_per_class の対象が現タスク限定なので学習は等価（design.md §1.7）。
# クラス数が 1 のタスクでは warmup を行わず iter 0 から per-class A を学習する
# （公式 do_warmup_a の条件。design.md §1.2）。本タスクは {single} である。
#
# trained_classes は「本タスクのクラスのうち、逐次順で先行するタスクに出現したもの」で、
# specialization 切替時に式3（A ← λ_A·A_warmup + (1-λ_A)·A_old）が発火する対象。
# 逐次順は seed {seed}（exp_034 と同一）。
#
# 逐次学習では前タスクの成長ライブラリ ckpt を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = './dithub_odinw13_base.py'

class_name = {classes}
metainfo = dict(classes=class_name)
_data_root = 'data/odinw/' + {root!r}

model = dict(dithub_classes=class_name)

custom_hooks = [
    dict(
        type='DitHubSeqPhaseHook',
        warmup_iters=1500,
        trained_classes={trained})
]

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
# exp_037: ODinW 評価（学習済みタスクのみ）— 自動生成物。直接編集しない。
#   生成: experiments/exp_037/gen_configs.py
#   時点: t={t:02d}（直前に学習したタスク: {last}）
#   対象タスク（逐次順）: {names}
# =============================================================================
_base_ = './dithub_odinw13_eval.py'

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


def fmt_list(items, indent=8):
    if not items:
        return '[]'
    body = ', '.join(repr(c) for c in items)
    return f'[{body}]'


def gen_eval():
    model = MODEL_BLOCK.format(classes=fmt_classes(ALL_CLASSES, indent=8))
    parts = [EVAL_HEADER.format(
        data_root=DATA_ROOT, model=model, n_cls=len(ALL_CLASSES))]
    for i, name in enumerate(CONFIG_ORDER, 1):
        t = ODINW13[name]
        parts.append(EVAL_TASK.format(
            idx=i, name=name, classes=fmt_classes(t['classes']),
            root=t['root'], ann=t['eval_ann'], img=t['eval_img']))
    parts.append(EVAL_FOOTER.format(
        prefixes='\n'.join(f'    {n!r},' for n in CONFIG_ORDER),
        datasets='\n'.join(f'    dataset_{n},' for n in CONFIG_ORDER),
        metrics='\n'.join(f'    val_evaluator_{n},' for n in CONFIG_ORDER)))
    path = os.path.join(CFG_DIR, 'dithub_odinw13_eval.py')
    with open(path, 'w') as f:
        f.write(''.join(parts))
    print('generated', path, f'({len(ALL_CLASSES)} クラス)')

    path = os.path.join(CFG_DIR, 'zcoco_eval.py')
    with open(path, 'w') as f:
        f.write(ZCOCO_TMPL.format(model=model, n_cls=len(ALL_CLASSES)))
    print('generated', path)


def gen_train():
    seen = set()
    for i, name in enumerate(ORDER, 1):
        t = ODINW13[name]
        classes = t['classes']
        trained = [c for c in classes if c.lower() in seen]
        ann_path = os.path.join('data/odinw', t['root'], t['train_ann'])
        with open(ann_path) as f:
            j = json.load(f)
        n_img, n_box = len(j['images']), len(j['annotations'])
        body = TRAIN_TMPL.format(
            t=i, name=name, seed=SEED, classes=fmt_classes(classes),
            root=t['root'], ann=t['train_ann'], img=t['train_img'],
            trained=fmt_list(trained),
            n_img=f'{n_img:,}', n_box=f'{n_box:,}', n_cls=len(classes),
            epochs=f'{6000 / max(n_img, 1):.1f}',
            single=('単一クラス（warmup なし）' if len(classes) == 1
                    else f'多クラス（{len(classes)} クラス、warmup あり）'))
        path = os.path.join(CFG_DIR, f'dithub_odinw13_{name}.py')
        with open(path, 'w') as f:
            f.write(body)
        print(f'generated {path}  (t={i:02d}, {len(classes)} cls, '
              f'式3対象={trained or "なし"})')
        seen.update(c.lower() for c in classes)


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
    gen_train()
    gen_subsets()
