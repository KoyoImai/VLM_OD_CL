# =============================================================================
# exp_003: underwater 単一ドメイン fine-tune 用 学習 config
# -----------------------------------------------------------------------------
# 元のドメイン config を継承し、学習設定（凍結範囲・バッチ・学習率）のみ変更する。
#   変更点1: backbone・language_model を「凍結(lr_mult=0.0)」→「0.1× で学習(lr_mult=0.1)」
#            （MM-Grounding DINO の COCO fine-tune に準拠。適応性能=upper bound を狙う）
#   変更点2: train_dataloader.batch_size 4 → 8（per-GPU）。4GPU × 勾配累積2 で実効バッチ = 8×4×2 = 64。
#            ※当初 per-GPU 16(累積なし)で実効64を狙ったが、多スケール拡張のピークで GPU3 が OOM。
#              VRAM 制約上 per-GPU は 8 に下げ、累積2 で実効64 を維持する（2026-06-14 OOM 対応）。
#   変更点3: lr=1e-4（実効バッチ64。線形整合値 2e-4 より保守的に半分へ設定）。
#   据え置き: 20 epochs(milestone[15]) / AdamW wd=1e-4 / clip_grad / 拡張・評価プロトコル。
#
# 参照: experiments/exp_003/design.md, experiments/exp_002.5/summary.md
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py'

# 実効バッチ 64（per-GPU 8 × 4GPU × 勾配累積2）
train_dataloader = dict(batch_size=8)

# lr=1e-4（実効64・保守的設定）、勾配累積2、backbone・言語を 0.1× で学習
optim_wrapper = dict(
    accumulative_counts=2,
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1),
            language_model=dict(lr_mult=0.1),
        )))
