# =============================================================================
# exp_034: ZiRa（ODinW-13 再現）共通ベース
#
#   exp_023 の zira_replayfree_base.py（論文準拠の lr / スケジュール / λ / η）を継承し、
#   公式実装に合わせて次の2点を戻す（design.md §2）:
#     - dn を無効化（use_dn=False + NoDNGroundingDINOHead。公式は dn_number=0）
#     - batch を合計 2 にする（公式 total_batch_size=2。学習は 1 GPU）
#   さらに train_pipeline を COCO 形式用に置き換える（ODVG 専用の RandomSamplingNegPos と
#   FilterAnnotations を外す。design.md §3）。
#
#   公式と一致していることを確認済みの値（design.md §6.6）:
#     AdamW lr 1e-3 / wd 1e-4、2000 iter、iter 800 で ×0.1、warmup なし、
#     grad clip 0.1(L2)、λ=0.1、η=0.2（llrb の lr_mult）、AMP/EMA なし。
#   num_classes は 256 のまま（13タスクで ckpt の形を一定に保つ。design.md §2）。
#   randomness の seed は 0（プロジェクト規約。2026-08-05 確定）。
#
#   データは各タスク config が定義する（gen_configs.py が生成）。
# =============================================================================
_base_ = '../../exp_023/configs/zira_replayfree_base.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.layers.zira_layers',
        'mmdet.models.necks.zira_channel_mapper',
        'mmdet.models.detectors.zira_grounding_dino',
        'mmdet.models.dense_heads.nodn_grounding_dino_head',
    ],
    allow_failed_imports=False)

# 公式準拠: dn を無効化する。config だけでは切れないため専用の実装を使う（design.md §6.1）。
model = dict(
    use_dn=False,
    bbox_head=dict(type='NoDNGroundingDINOHead'))

# COCO 形式用の train_pipeline。事前学習 pipeline から ODVG 専用の
# FilterAnnotations と RandomSamplingNegPos を外したもので、RF100 用 finetune config
# （grounding_dino_swin-t_finetune_8xb4_20e_*.py）と同一構成。公式 ZiRa の augmentation
# （RandomFlip ＋ 多スケール resize、または resize(400/500/600)→RandomCrop(384,600)→
# 多スケール resize）と対応する。
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
train_cfg = dict(val_interval=100000)
