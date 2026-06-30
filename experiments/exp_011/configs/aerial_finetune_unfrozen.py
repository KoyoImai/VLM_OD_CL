# =============================================================================
# exp_011: aerial 単独FT（Image Backbone(Swin)・Text Backbone(BERT) も 0.1×学習, seed=0）
#   標準 aerial finetune config を継承し、backbone/language を lr_mult=0.1、seed=0 を設定するのみ。
#   exp_010(underwater) と同一方針を残り5ドメインへ展開。frozen×seed=0 は exp_005 を流用。
#   参照: experiments/exp_011/design.md, experiments/exp_010/results/underwater_frozen_vs_unfrozen.md
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_aerial.py'  # noqa

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1),
            language_model=dict(lr_mult=0.1),
        )))

# seed=0 固定（全実験共通方針）。
randomness = dict(deterministic=False, seed=0)
