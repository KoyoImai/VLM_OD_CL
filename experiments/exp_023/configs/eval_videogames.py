# exp_023 評価: videogames valid（ODVG 学習済みモデル用）
#   既存 finetune を継承。ODVG 学習済み ckpt に合わせて num_classes=256・max_text_len=256 に
#   上書き。videogames は 87クラス=265トークンで256を超えるため、chunked_size で分割推論する。
#   既存ファイルは無変更。
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_videogames.py'  # noqa

model = dict(
    bbox_head=dict(
        num_classes=256,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True)),
    test_cfg=dict(chunked_size=40))
