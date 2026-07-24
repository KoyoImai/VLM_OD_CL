# exp_023 評価: underwater valid（ODVG 学習済みモデル用）
#   既存 finetune を継承し valid/評価設定を再利用。ODVG 学習済み ckpt に合わせて
#   num_classes=256 に上書き（max_text_len は 256 のまま。28クラス=111トークンで収まる）。
#   既存ファイルは無変更。
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py'  # noqa

model = dict(bbox_head=dict(num_classes=256))
