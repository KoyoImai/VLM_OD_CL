# exp_002 / aerial / s4_640sq: FixScaleResize (640,640) keep_ratio=False（正方形固定）
# 元の config を継承し、評価時リサイズ（FixScaleResize）のみ上書き（再現性のため config 化）。
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_aerial.py'

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=(640, 640), keep_ratio=False, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities', 'tokens_positive'))
]
val_dataloader = dict(dataset=dict(return_classes=True, pipeline=test_pipeline))
test_dataloader = val_dataloader
