# exp_023 評価: electromagnetic valid（ODVG 学習済みモデル用）
#   既存 finetune を継承。num_classes=256 に上書き（39クラス=140トークンで256に収まる）。
#   既存ファイルは無変更。
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_electromagnetic.py'  # noqa

model = dict(bbox_head=dict(num_classes=256))
