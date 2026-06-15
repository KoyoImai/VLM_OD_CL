# exp_002 / coco / s3_1333: FixScaleResize (1333,2000) keep_ratio=True（高解像度）
# 元の config を継承し、評価時リサイズ（FixScaleResize）のみ上書き（再現性のため config 化）。
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=(1333, 2000), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities', 'tokens_positive'))
]
val_dataloader = dict(dataset=dict(return_classes=True, pipeline=test_pipeline))
test_dataloader = val_dataloader
