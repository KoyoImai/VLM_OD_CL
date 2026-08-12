# =============================================================================
# exp_037: DitHub（ODinW-13 再現）共通ベース
#
#   exp_023 の dithub_replayfree_base.py（論文準拠の lr / スケジュール / フェーズ切替）を
#   継承し、公式実装に合わせて次を戻す（design.md §1.4, §4.1）:
#     - dn を無効化（use_dn=False + NoDNGroundingDINOHead。公式は dn_number=0）
#     - 検出器を DitHubGroundingDINO に戻す（ODinW は COCO 形式で text がクラス名の
#       タプルなので、ODVG のキャプション分割は不要）
#     - batch を合計 2 にする（公式 total_batch_size=2。学習は 1 GPU）
#     - activation checkpointing を無効化（batch 2 では不要。出力は数学的に同一）
#   さらに train_pipeline を COCO 形式用に置き換える（ODVG 専用の RandomSamplingNegPos と
#   FilterAnnotations を外す。design.md §1.8 / §4.1）。公式の augmentation
#   （50% で Flip+多スケール resize、50% で Flip+resize(400/500/600)+RandomCrop(384,600)
#   +多スケール resize）と対応する。
#
#   継承して公式と一致している値（design.md §4.1 で照合済み）:
#     AdamW lr 1e-3 / wd 1e-2、3000 iter、iter 1200 で ×0.1、lr の linear warmup なし、
#     grad clip 0.1(L2)、warmup→specialization の切替 iter 1500、r=16 / alpha=8 /
#     dropout 0、λ_A=0.3、λ_B=0.7、AMP/EMA なし。
#   num_classes は 256 のまま（13 タスクで ckpt の形を一定に保つ。dn 無効なので
#   dn_query_generator.label_embedding は参照されない）。
#   randomness の seed は 0（プロジェクト規約。exp_023 の base から継承）。
#
#   dithub_classes と trained_classes とデータは各タスク config が定義する
#   （gen_configs.py が生成）。
# =============================================================================
_base_ = '../../exp_023/configs/dithub_replayfree_base.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.layers.dithub_layers',
        'mmdet.models.detectors.dithub_grounding_dino',
        'mmdet.engine.hooks.dithub_phase_hook',
        'mmdet.engine.hooks.dithub_seq_phase_hook',
        'mmdet.models.dense_heads.nodn_grounding_dino_head',
    ],
    allow_failed_imports=False)

model = dict(
    type='DitHubGroundingDINO',
    use_dn=False,
    encoder_cp=0,
    bbox_head=dict(type='NoDNGroundingDINOHead'))

# COCO 形式用の train_pipeline（exp_034 と同一）。
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
