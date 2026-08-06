#!/usr/bin/env python3
"""exp_034: config を生成する（自己完結・リテラルのパスで書き出す）.

mmengine の config は実行時に __file__ を持たないため、共通テーブルを import する
書き方ができない。そこで本スクリプトでタスク定義（odinw_official_tasks.py）から
config ファイルを生成し、生成物だけを実験で使う。生成物は人が読める通常の config で、
mmengine の dump / 再現に支障がない。

生成するもの:
    configs/odinw13_official_eval.py        全13タスクの評価 config
    configs/zira_odinw13_<task>.py          学習 config × 13

使い方: python experiments/exp_034/gen_configs.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from odinw_official_tasks import CONFIG_ORDER, DATA_ROOT, ODINW13  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, 'configs')

EVAL_HEADER = '''\
# =============================================================================
# exp_034: ODinW-13 評価 config（公式 ZiRa 実装と同一の版・split）
#
# 【自動生成】experiments/exp_034/gen_configs.py が
# experiments/exp_034/odinw_official_tasks.py から生成する。直接編集しない。
#
# 本リポジトリの configs/mm_grounding_dino/odinw/…odinw13.py との違い（design.md §3）:
#   - 評価は valid ではなく test（PascalVOC のみ valid）
#   - AerialMaritimeDrone は large ではなく tiled
#   - NorthAmericaMushrooms は評価に v2 の train を使う（v2 に test が無いため。公式同様）
#   - Packages は Raw ではなく augmented-v1
#   - Raccoon は v2-raw ではなく v38-416x416-resize
#   - ShellfishOpenImages は raw ではなく 416x416
# クラス名は odinw13 config と全タスク一致することを実測確認済みなので流用している。
#
# ZiRa の forward は全和（base + s*HLRB + LLRB）。公式は評価時 LLRB のみだが、
# Rep+ 後は HLRB=1e-8 / s=0.1 で寄与は 1e-9 オーダー（design.md §6.5(2)）。
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'  # noqa

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.layers.zira_layers',
        'mmdet.models.necks.zira_channel_mapper',
        'mmdet.models.detectors.zira_grounding_dino',
        'mmdet.models.dense_heads.nodn_grounding_dino_head',
    ],
    allow_failed_imports=False)

model = dict(
    type='ZiRaGroundingDINO',
    zil_loss_weight=0.1,
    use_dn=False,
    neck=dict(type='ZiRaChannelMapper'),
    bbox_head=dict(type='NoDNGroundingDINOHead'))

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

TRAIN_TMPL = '''\
# =============================================================================
# exp_034: ZiRa 学習 config / {name}
#
# 【自動生成】experiments/exp_034/gen_configs.py が
# experiments/exp_034/odinw_official_tasks.py から生成する。直接編集しない。
#
# スケジュール・lr・optimizer・ZiRa の設定は zira_odinw13_base.py（公式準拠）を継承し、
# 本 config はデータだけを定義する。学習データは公式実装と同じ版の train split。
#   画像 {n_img} 枚 / box {n_box} / クラス {n_cls}（2000 iter × batch 2 = 4000 枚で {epochs} 周）
# 逐次学習では前タスクの Rep+ 融合済み ckpt を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = './zira_odinw13_base.py'

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


def fmt_classes(classes):
    if len(classes) == 1:
        return f'({classes[0]!r}, )'
    body = ', '.join(repr(c) for c in classes)
    if len(body) <= 68:
        return f'({body})'
    # 長い場合は折り返す
    out, line = [], ''
    for c in classes:
        item = repr(c) + ', '
        if len(line) + len(item) > 68:
            out.append(line.rstrip())
            line = ''
        line += item
    out.append(line.rstrip().rstrip(','))
    return '(' + ('\n' + ' ' * 14).join(out) + ')'


def gen_eval():
    import json
    parts = [EVAL_HEADER.format(data_root=DATA_ROOT)]
    for i, name in enumerate(CONFIG_ORDER, 1):
        t = ODINW13[name]
        parts.append(
            EVAL_TASK.format(
                idx=i, name=name, classes=fmt_classes(t['classes']),
                root=t['root'], ann=t['eval_ann'], img=t['eval_img']))
    parts.append(
        EVAL_FOOTER.format(
            prefixes='\n'.join(f"    {n!r}," for n in CONFIG_ORDER),
            datasets='\n'.join(f'    dataset_{n},' for n in CONFIG_ORDER),
            metrics='\n'.join(
                f'    val_evaluator_{n},' for n in CONFIG_ORDER)))
    path = os.path.join(CFG_DIR, 'odinw13_official_eval.py')
    with open(path, 'w') as f:
        f.write(''.join(parts))
    print('generated', path)


def gen_train():
    import json
    for name in CONFIG_ORDER:
        t = ODINW13[name]
        ann_path = os.path.join('data/odinw', t['root'], t['train_ann'])
        with open(ann_path) as f:
            j = json.load(f)
        n_img, n_box = len(j['images']), len(j['annotations'])
        body = TRAIN_TMPL.format(
            name=name, classes=fmt_classes(t['classes']), root=t['root'],
            ann=t['train_ann'], img=t['train_img'], n_img=f'{n_img:,}',
            n_box=f'{n_box:,}', n_cls=len(t['classes']),
            epochs=f'{4000 / n_img:.1f}')
        path = os.path.join(CFG_DIR, f'zira_odinw13_{name}.py')
        with open(path, 'w') as f:
            f.write(body)
        print('generated', path)


if __name__ == '__main__':
    os.makedirs(CFG_DIR, exist_ok=True)
    gen_eval()
    gen_train()
