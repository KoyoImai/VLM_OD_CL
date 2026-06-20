# =============================================================================
# exp_006 / aerial -> underwater: 2ドメイン逐次学習（D1=aerial, D2=underwater）
# -----------------------------------------------------------------------------
# 目的: aerial(D1) を学習済みのモデルから underwater(D2) を継続学習し、
#       D2 学習後に (a) aerial(D1) の性能低下＝D1忘却、(b) underwater 適応、(c) COCO累積忘却 を測る。
# 設定: exp_005 の凍結FTと同一（backbone・言語凍結 / lr=1e-4 / 実効64=per-GPU8×4×累積2 /
#       20ep[15] / AdamW wd=1e-4 / seed=0,deterministic=False）。
#       唯一の差: load_from を事前学習重み → **aerial(D1) チェックポイント** に変更（=逐次継続）。
# 参照: experiments/exp_006/design.md
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py'

# D1(aerial) 学習済みモデルから継続学習する（逐次学習の肝）
load_from = '/workspace/kouyou/mmdetection/experiments/exp_005/aerial_work_dir/best_coco_bbox_mAP_epoch_17.pth'

# 実効バッチ 64（per-GPU8×4GPU×累積2）
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

# 乱数 seed 固定
randomness = dict(seed=0, deterministic=False)
