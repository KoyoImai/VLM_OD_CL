# =============================================================================
# exp_049: DGS stage1（新グループ・無制約学習）/ ODinW-13 IVLOD
#
# 公式 projects/DGS/configs/IVLOD/dgs_stage1.py を本環境（MM-GDINO θ0・
# data/odinw の ZiRa 形式レイアウト）に合わせたもの。design.md §2。
# タスクごとの値（データパス・metainfo・seen_tasks・task_id・load_from）は
# ドライバ run_dgs_odinw13.sh が --cfg-options で毎回上書きする。
#
# 公式との意図的な差分（implementation_plan §2.2）:
#   - _base_ が MM-GDINO 事前学習 config（θ0 互換。contrastive_cfg は
#     log_scale='auto'/bias=True の MM-GDINO 既定を継承する）
#   - バッチ 4/GPU × 4 GPU = 16（公式 2×8=16。合計一致）
#   - encoder num_cp は継承既定（6）のまま（公式 base も num_cp=6）
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'  # noqa

custom_imports = dict(imports=['projects.dgs_cl'], allow_failed_imports=False)

# IGA（公式 IVLOD と同一）: enhancer 6 層の画像側・テキスト側 FFN を
# グループ別 LoRA（r=16, alpha=32）付きに置換
moe_cfg = dict(
    type='moe_adaptive_expand_lora', experts_num=1, top_k=1,
    r=16, alpha=32.0, dropout=0.0,
    group_cfg=dict(type='rep', merge_method='ema', lambda_A=0.2, lambda_B=0.2),
    replace_layer_type=['enc_ffn_img', 'enc_ffn_text'],
    replace_enc_layer_ids=[0, 1, 2, 3, 4, 5], replace_dec_layer_ids=[])

model = dict(
    type='GroundingDINO_DGS_Base',
    # 3 つともドライバが --cfg-options で毎タスク上書きする
    num_tasks=13,
    task_id=0,
    seen_tasks='AerialMaritimeDrone',
    moe_cfg=moe_cfg,
    vis_cfg=dict(type='none', save_path=''),
    # 公式 stage1: base を全凍結（lora_ のみ学習）
    frozen_cfg=dict(
        backbone_frozen=True,
        language_model_frozen=True,
        neck_frozen=True,
        encoder_frozen=True,
        decoder_frozen=True,
        head_frozen=True,
        exclude_keywords=['lora_']),
    # DTG / ルーティング（公式 IVLOD と同一。特徴・統計は本実験のディレクトリ）
    domain_predictor_cfg=dict(
        type='svd',
        feat_path='experiments/exp_049/feats/',
        stats_path='experiments/exp_049/stats/',
        multilevel=False,
        expand_th=150,
        ood_th=500,
        min_eig_ratio=1e-3),
    bbox_head=dict(
        type='GroundingDINOHead_inc',
        setting='cur_text',
        trunc_class=[0, 256]),
)
# dn_cfg は継承（有効）。公式 stage1 も dn 有効（dn_cfg=None は stage2 のみ）。

# 学習パイプライン（公式 IVLOD と同一。COCO 形式・ori_text メタキー付き）
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
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
                   'scale_factor', 'flip', 'flip_direction', 'text', 'ori_text',
                   'custom_entities'))
]

# 学習データ（タスクごとの実パスはドライバが上書き。ここは t=1 の既定値）
train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ODinW13_Dataset',
        metainfo='AerialMaritimeDrone',
        data_root='data/odinw/AerialMaritimeDrone/tiled/',
        ann_file='train/annotations_without_background.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        return_classes=True))

# スケジュール（公式: 12 epoch / MultiStepLR [11] / stage1 lr 8e-4）
optim_wrapper = dict(optimizer=dict(lr=0.0008))
max_epochs = 12
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=max_epochs + 1)  # in-training val はしない（評価はドライバ）

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=max_epochs, max_keep_ckpts=-1,
        save_optimizer=False, save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=50))

# DGS の学習フック（公式と同一）:
#   WeightsTransformHook: θ_{t-1} ロード時に plain キー → .base_layer. へ写像
#                          （moe_lora）と Group Init（base LoRA → 新 LoRA コピー）
#   MergeHook: 最終 epoch 後の EMA マージ（新 ← 0.2·base + 0.8·新 → base 昇格）
#   DomainPredictorHooK: ルーティング精度のログ
custom_hooks = [
    dict(type='WeightsTransformHook',
         cfg=[dict(type='moe_lora'), dict(type='moe_group_init')]),
    dict(type='MergeHook', cfg=dict(type='ema')),
    dict(type='DomainPredictorHooK'),
]

randomness = dict(deterministic=False, seed=42)  # 公式ドライバの SEED=42
