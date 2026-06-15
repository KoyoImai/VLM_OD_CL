# =============================================================================
# Roboflow100（RF100）ドメイン 共通評価プロトコル（exp_002.5 で確定）
# -----------------------------------------------------------------------------
# 本研究の RF100 各ドメイン評価は、必ず本プロトコルを適用して行う。
# 全ドメインで評価設定を完全に揃え、ドメイン間の比較・継続学習の交絡を避ける。
#
# 確定プロトコル（全ドメイン共通）:
#   - データ形式    : 各ドメイン valid/_annotations.coco.json（CocoDataset）
#   - リサイズ      : FixScaleResize scale=(800,1333) keep_ratio=True（解像度は変更しない）
#   - batch_size    : 1
#   - 評価指標      : CocoMetric (bbox) = COCO mAP
#   - テキスト      : 各ドメインのクラス名を連結（return_classes=True）
#
# 本ファイルは「事前学習 config を継承し、評価プロトコル（解像度・batch・dataloader）を固定」した
# 単一ベースである。ドメイン固有の値（data_root / metainfo / ann_file / num_classes / 評価器）は
# 継承する側のドメイン eval config で与える。
#
# 使い方（ドメイン eval config の例）:
#   _base_ = 'eval_base_rf100.py'                 # ← 単一ベースとして継承（重複キー回避）
#   data_root = '/workspace/kouyou/datasets/rf100_domain/<domain>/'
#   metainfo  = dict(classes=(...))               # ドメインのクラス名
#   model = dict(bbox_head=dict(num_classes=<N>)) # videogames は contrastive_cfg(max_text_len=512) も
#   val_dataloader = dict(dataset=dict(
#       data_root=data_root, metainfo=metainfo,
#       ann_file='valid/_annotations.coco.json', data_prefix=dict(img='valid/')))
#   test_dataloader = val_dataloader
#   val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
#   test_evaluator = val_evaluator
#
# 注: 既存の学習用ドメイン config（grounding_dino_swin-t_finetune_8xb4_20e_<domain>.py）も
#     同一プロトコル（800,1333）を満たすため、単発評価はそれを直接 test.py に渡してもよい。
#     本ファイルは「評価プロトコルを明文化・固定」し、将来の派生で必ず継承させるためのもの。
#
# 参照: experiments/exp_002.5/summary.md（評価プロトコル）、exp_001（lower bound 基準値）
# =============================================================================
_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

# 評価パイプライン（解像度 800,1333 を明示）
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities', 'tokens_positive')),
]

# 評価データローダの共通骨格（data_root / metainfo / ann_file はドメイン側で設定）
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        test_mode=True,
        return_classes=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# 評価器（ann_file はドメイン側で上書きする）
val_evaluator = dict(type='CocoMetric', metric='bbox', format_only=False)
test_evaluator = val_evaluator
