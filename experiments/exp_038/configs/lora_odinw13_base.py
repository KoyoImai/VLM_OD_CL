# =============================================================================
# exp_038: 素の LoRA（特徴抽出部＋融合部）の ODinW-13 逐次学習 共通ベース
#
#   exp_023 の fullft_replay_base.py（事前学習 config 継承・num_classes=256・θ0・seed 0）を
#   土台に、学習設定を exp_037（DitHub 準拠）と同一に置き換える。DitHub 固有の要素
#   （DitHubSeqPhaseHook, find_unused_parameters）は持ち込まない。
#
#   exp_037 と同一の値（design.md §2.2）:
#     3000 iter, batch 2（1 GPU）, AdamW lr 1e-3 / wd 1e-2, iter 1200 で ×0.1,
#     lr の linear warmup なし, grad clip 0.1(L2), dn 無効, AMP/EMA なし, seed 0,
#     num_classes 256, COCO 形式の train_pipeline。
#
#   LoRA の挿入箇所（design.md §2.1、実測 232 層 / 5,671,936 params = 3.17%）:
#     backbone(Swin-T) 51 + language_model(BERT) 72 + encoder 108 + text_feat_map 1
#     encoder のテキスト self-attention は DecomposedMHA で q/k/v/o に分解して挿入する。
#     decoder / bbox_head / memory_trans_fc には入れない。
#
#   optimizer には LoRA だけを登録する（TrainableParamsConstructor）。事前学習 config の
#   paramwise_cfg（backbone/language lr_mult=0.1）は使わないので _delete_ で置き換える。
#
#   データとクラスは各タスク config が定義する（gen_configs.py が生成）。
# =============================================================================
_base_ = '../../exp_023/configs/fullft_replay_base.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'projects.lora_cl',
        'mmdet.models.dense_heads.nodn_grounding_dino_head',
    ],
    allow_failed_imports=False)

model = dict(
    type='GroundingDINOLoRA',
    use_dn=False,
    bbox_head=dict(type='NoDNGroundingDINOHead'),
    lora=dict(
        r=16,
        alpha=8,
        # Swin と BERT を対象に含める（既定除外を空にする）
        exclude_components=[],
        # feature enhancer のテキスト self-attention を q/k/v/o へ分解する
        decompose_mha=[r'^encoder\.text_layers\.\d+\.self_attn\.attn$'],
        include=['backbone', 'language_model', 'text_feat_map', 'encoder']))

# LoRA のみを optimizer に登録する。lr / wd / clip は exp_037 と同一。
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    constructor='TrainableParamsConstructor',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    clip_grad=dict(max_norm=0.1, norm_type=2))

# スケジュール（exp_037 と同一）
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=3000,
    val_interval=100000)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=3000,
        by_epoch=False,
        milestones=[1200],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=3000,
        max_keep_ckpts=1,
        save_best=None),
    logger=dict(type='LoggerHook', interval=50))
log_processor = dict(by_epoch=False)

# COCO 形式用の train_pipeline（exp_034 / exp_037 と同一）。
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

# 学習中の val は行わない（評価は逐次ドライバが実施する）。
val_dataloader = None
val_evaluator = None
val_cfg = None
test_dataloader = None
test_evaluator = None
test_cfg = None
