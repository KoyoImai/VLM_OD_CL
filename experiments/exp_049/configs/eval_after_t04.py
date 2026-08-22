# =============================================================================
# exp_049: DGS 評価（t=4 時点・学習済み 4 タスク per-dataset）
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
    task_id=3,
    seen_tasks='AerialMaritimeDrone,Aquarium,CottontailRabbits,EgoHands',
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
        ood_th=500,
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

# --------------------- AerialMaritimeDrone ---------------------#
dataset_AerialMaritimeDrone = dict(
    type='CocoDataset',
    metainfo=dict(classes=('boat', 'car', 'dock', 'jetski', 'lift')),
    data_root='data/odinw/AerialMaritimeDrone/tiled/',
    ann_file='test/annotations_without_background.json',
    data_prefix=dict(img='test/'),
    test_mode=True,
    pipeline=test_pipeline,
    return_classes=True)
val_evaluator_AerialMaritimeDrone = dict(
    type='CocoMetric',
    ann_file='data/odinw/AerialMaritimeDrone/tiled/test/annotations_without_background.json',
    metric='bbox')

# --------------------- Aquarium ---------------------#
dataset_Aquarium = dict(
    type='CocoDataset',
    metainfo=dict(classes=('fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray')),
    data_root='data/odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/',
    ann_file='test/annotations_without_background.json',
    data_prefix=dict(img='test/'),
    test_mode=True,
    pipeline=test_pipeline,
    return_classes=True)
val_evaluator_Aquarium = dict(
    type='CocoMetric',
    ann_file='data/odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/test/annotations_without_background.json',
    metric='bbox')

# --------------------- CottontailRabbits ---------------------#
dataset_CottontailRabbits = dict(
    type='CocoDataset',
    metainfo=dict(classes=('Cottontail-Rabbit',)),
    data_root='data/odinw/CottontailRabbits/',
    ann_file='test/annotations_without_background.json',
    data_prefix=dict(img='test/'),
    test_mode=True,
    pipeline=test_pipeline,
    return_classes=True)
val_evaluator_CottontailRabbits = dict(
    type='CocoMetric',
    ann_file='data/odinw/CottontailRabbits/test/annotations_without_background.json',
    metric='bbox')

# --------------------- EgoHands ---------------------#
dataset_EgoHands = dict(
    type='CocoDataset',
    metainfo=dict(classes=('hand',)),
    data_root='data/odinw/EgoHands/generic/',
    ann_file='test/annotations_without_background.json',
    data_prefix=dict(img='test/'),
    test_mode=True,
    pipeline=test_pipeline,
    return_classes=True)
val_evaluator_EgoHands = dict(
    type='CocoMetric',
    ann_file='data/odinw/EgoHands/generic/test/annotations_without_background.json',
    metric='bbox')

val_dataloader = dict(
    batch_size=1,
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=[dataset_AerialMaritimeDrone, dataset_Aquarium, dataset_CottontailRabbits, dataset_EgoHands]))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=[val_evaluator_AerialMaritimeDrone, val_evaluator_Aquarium, val_evaluator_CottontailRabbits, val_evaluator_EgoHands],
    dataset_prefixes=['AerialMaritimeDrone', 'Aquarium', 'CottontailRabbits', 'EgoHands'])
test_evaluator = val_evaluator
