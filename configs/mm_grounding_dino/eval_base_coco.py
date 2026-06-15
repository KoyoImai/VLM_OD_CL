# =============================================================================
# COCO2017 用 共通評価プロトコル（exp_002.5 で確定）
# -----------------------------------------------------------------------------
# 本研究の COCO 評価（汎用知識＝ZCOCO 相当）は、必ず本 config を経由して行う。
# これにより COCO 評価設定を一箇所に固定し、実験間での揺れを防ぐ（再現性）。
#
# 確定プロトコル:
#   - データ        : COCO2017 val (instances_val2017.json / 80クラス)
#   - リサイズ      : FixScaleResize scale=(800,1333) keep_ratio=True（解像度は変更しない）
#   - batch_size    : 1
#   - 評価指標      : CocoMetric (bbox) = COCO mAP
#   - テキスト      : 80クラス名を連結（return_classes=True）
#
# 使い方（チェックポイントは test.py の引数で渡す。zero-shot は事前学習重みを渡す）:
#   python tools/test.py configs/mm_grounding_dino/eval_base_coco.py <ckpt> --work-dir <dir>
#   bash  tools/dist_test.sh configs/mm_grounding_dino/eval_base_coco.py <ckpt> 4 --work-dir <dir>
#
# 参照: experiments/exp_002.5/summary.md（評価プロトコル）、exp_001（COCO=0.504 の基準値）
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
val_dataloader = dict(
    batch_size=1,
    dataset=dict(return_classes=True, pipeline=test_pipeline))
test_dataloader = val_dataloader
