# =============================================================================
# exp_005 / videogames: backbone・言語エンコーダ 凍結 fine-tune（exp_004 設定を継承）
# -----------------------------------------------------------------------------
# 目的: underwater(exp_004) と同様に、videogames の fine-tune でも COCO 精度低下（忘却）が
#       起きることを確認する。
# 設定: exp_004 と同一（lr=1e-4 / 実効64=per-GPU8×4GPU×累積2 / 20ep[15] /
#       AdamW wd=1e-4 / backbone・language_model 凍結(lr_mult=0.0)）。
#       ドメイン固有値（data/クラス/num_classes、videogamesはmax_text_len=512）は元 config を継承。
# 参照: experiments/exp_005/design.md, experiments/exp_004/
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_videogames.py'

# 実効バッチ 64（per-GPU 8 × 4GPU × 累積2）
train_dataloader = dict(batch_size=8)

# lr=1e-4、累積2、backbone・言語エンコーダ 凍結
optim_wrapper = dict(
    accumulative_counts=2,
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.0),
            language_model=dict(lr_mult=0.0),
        )))

# 乱数 seed を固定（再現性・公平な比較のため）。deterministic は別途決定。
randomness = dict(seed=0, deterministic=False)
