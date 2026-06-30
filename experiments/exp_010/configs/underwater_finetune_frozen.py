# =============================================================================
# exp_010: underwater 単一ドメイン fine-tune（backbone・言語エンコーダ 凍結, seed=0）
# -----------------------------------------------------------------------------
# 標準 underwater finetune config（backbone・language_model を凍結=lr_mult=0.0）を継承し、
# seed=0 を固定するのみ。
#   = exp_004（frozen underwater, ただしランダムseed=384152389）の seed=0 版。
#   exp_005 が他5ドメインの frozen×seed=0 を持つのに underwater だけ欠けていたため、本実験で補完。
#
# 据え置き: base lr=1e-4 / 実効64 / 20ep[15] / AdamW wd=1e-4 / clip_grad / 評価(800,1333)。
# 対比: 同ドメインの unfrozen×seed=0（underwater_finetune_unfrozen.py）と並べ、
#       backbone 凍結 vs 0.1×学習 の適応/忘却トレードオフを seed 統一で比較する。
# 参照: experiments/exp_010/design.md
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py'  # noqa

# 標準configのまま backbone(Swin)・language_model(BERT) は凍結(lr_mult=0.0)。
# seed=0 固定のみ追加（研究計画 §5・exp_005 と整合。標準configは randomness 未設定）。
randomness = dict(deterministic=False, seed=0)
