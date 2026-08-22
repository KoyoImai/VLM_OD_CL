#!/usr/bin/env python3
"""exp_049: 評価 config を生成する。

- eval_after_t{01..13}.py : t 時点で学習済みのタスク（公式順の先頭 t 個）を
  per-dataset 評価（MultiDatasetsEvaluator。exp_034 と同じ流儀）。
  モデルは DGS（画像単位ルーティング。タスク ID は与えない）。
- eval_zcoco_dgs.py       : COCO2017 val の ZCOCO（公式 ZCOCO.py に対応。
  ood_th=200 に下げる点も公式どおり）。

seen_tasks / task_id / task_id_mapping_path はドライバが --cfg-options で渡す。
データパスの正本は experiments/exp_034/odinw_official_tasks.py。

使い方: python experiments/exp_049/gen_eval_configs.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'exp_034'))
from odinw_official_tasks import DATA_ROOT, ODINW13  # noqa: E402

CFG_DIR = os.path.join(HERE, 'configs')

OFFICIAL_ORDER = [
    'AerialMaritimeDrone', 'Aquarium', 'CottontailRabbits', 'EgoHands',
    'NorthAmericaMushroom', 'Packages', 'PascalVOC', 'pistols', 'pothole',
    'Raccoon', 'ShellfishOpenImages', 'thermalDogsAndPeople',
    'VehiclesOpenImages'
]
ALIAS = {'NorthAmericaMushroom': 'NorthAmericaMushrooms'}

HEADER = '''\
# =============================================================================
# exp_049: {title}
# 【自動生成】experiments/exp_049/gen_eval_configs.py。直接編集しない。
# seen_tasks / task_id / domain_predictor_cfg.task_id_mapping_path は
# ドライバが --cfg-options で上書きする。
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'  # noqa

custom_imports = dict(imports=['projects.dgs_cl'], allow_failed_imports=False)

moe_cfg = dict(
    type='moe_adaptive_expand_lora', experts_num=1, top_k=1,
    r=16, alpha=32.0, dropout=0.0,
    group_cfg=dict(type='rep', merge_method='ema', lambda_A=0.2, lambda_B=0.2),
    replace_layer_type=['enc_ffn_img', 'enc_ffn_text'],
    replace_enc_layer_ids=[0, 1, 2, 3, 4, 5], replace_dec_layer_ids=[])

model = dict(
    type='GroundingDINO_DGS_Base',
    num_tasks=13,
    task_id={task_id},
    seen_tasks='{seen_csv}',
    moe_cfg=moe_cfg,
    vis_cfg=dict(type='none', save_path=''),
    frozen_cfg=dict(
        backbone_frozen=True, language_model_frozen=True, neck_frozen=True,
        encoder_frozen=True, decoder_frozen=True, head_frozen=True,
        exclude_keywords=['lora_']),
    domain_predictor_cfg=dict(
        type='svd',
        feat_path='experiments/exp_049/feats/',
        stats_path='experiments/exp_049/stats/',
        task_id_mapping_path='experiments/exp_049/work_dirs/task_id_mapping.yaml',
        multilevel=False,
        expand_th=150,
        ood_th={ood_th},
        min_eig_ratio=1e-3),
    bbox_head=dict(
        type='GroundingDINOHead_inc',
        setting='cur_text',
        trunc_class=[0, 256]),
)

# ckpt ロード時のキー写像（plain → .base_layer.）。DGS 学習 ckpt はそのままでも
# 読めるが、θ0 を直接評価する場合に必要。group_init は評価では作用しない。
custom_hooks = [
    dict(type='WeightsTransformHook', cfg=[dict(type='moe_lora')]),
    dict(type='DomainPredictorHooK'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities')),
]
'''

DS_TMPL = '''
# --------------------- {name} ---------------------#
dataset_{ident} = dict(
    type='CocoDataset',
    metainfo=dict(classes={classes}),
    data_root='{root}',
    ann_file='{ann}',
    data_prefix=dict(img='{img}'),
    test_mode=True,
    pipeline=test_pipeline,
    return_classes=True)
val_evaluator_{ident} = dict(
    type='CocoMetric',
    ann_file='{root}{ann}',
    metric='bbox')
'''

TAIL = '''
val_dataloader = dict(
    batch_size=1,
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=[{ds_list}]))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=[{metric_list}],
    dataset_prefixes={prefixes})
test_evaluator = val_evaluator
'''

ZCOCO_TAIL = '''
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root='/workspace/kouyou/datasets/coco2017/',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        return_classes=True,
        backend_args=None))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file='/workspace/kouyou/datasets/coco2017/annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator
'''


def gen():
    os.makedirs(CFG_DIR, exist_ok=True)
    for t in range(1, 14):
        seen = OFFICIAL_ORDER[:t]
        body = HEADER.format(
            title=f'DGS 評価（t={t} 時点・学習済み {t} タスク per-dataset）',
            task_id=t - 1, seen_csv=','.join(seen), ood_th=500)
        for name in seen:
            spec = ODINW13[ALIAS.get(name, name)]
            body += DS_TMPL.format(
                name=name, ident=name, classes=repr(spec['classes']),
                root=DATA_ROOT + spec['root'], ann=spec['eval_ann'],
                img=spec['eval_img'])
        body += TAIL.format(
            ds_list=', '.join(f'dataset_{n}' for n in seen),
            metric_list=', '.join(f'val_evaluator_{n}' for n in seen),
            prefixes=repr(seen))
        path = os.path.join(CFG_DIR, f'eval_after_t{t:02d}.py')
        open(path, 'w').write(body)
        print('generated', path)

    # ZCOCO（公式 ZCOCO.py: ood_th=200 / COCO val / batch 1）
    body = HEADER.format(
        title='DGS ZCOCO 評価（COCO2017 val・ood_th=200 は公式 ZCOCO.py どおり）',
        task_id=12, seen_csv=','.join(OFFICIAL_ORDER), ood_th=200)
    body += ZCOCO_TAIL
    path = os.path.join(CFG_DIR, 'eval_zcoco_dgs.py')
    open(path, 'w').write(body)
    print('generated', path)


if __name__ == '__main__':
    gen()
