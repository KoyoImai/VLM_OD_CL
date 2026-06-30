# =============================================================================
# exp_010: underwater 単一ドメイン fine-tune（Image Backbone(Swin)・Text Backbone(BERT)
#          も学習可能にする）
# -----------------------------------------------------------------------------
# 標準 underwater finetune config（backbone・language_model を凍結=lr_mult=0.0）を継承し、
#   変更点: backbone・language_model を「凍結(0.0)」→「0.1× 学習(lr_mult=0.1, 実効1e-5相当)」。
#           → head/encoder/decoder/neck に加え、Swin backbone と BERT も更新する。
#   据え置き（exp_004/005 と完全同一）: base lr=1e-4 / 実効バッチ64(per-GPU8×4GPU×累積2) /
#           20 epochs(milestone[15]) / AdamW wd=1e-4 / clip_grad / 拡張・評価プロトコル(800,1333) / seed=0。
#
# 目的: 凍結FT(exp_004/005) に対し backbone まで学習させたときの 適応(自ドメイン mAP) と
#       事前学習知識の忘却(COCO mAP=ZCOCO) を測る。underwater は exp_003(0.1×学習) とほぼ一致する想定。
# 参照: experiments/exp_010/design.md
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py'  # noqa

# backbone(Swin)・language_model(BERT) を 0.1× で学習可能にする（他は標準configを継承）。
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1),
            language_model=dict(lr_mult=0.1),
        )))

# seed=0 固定（研究計画 §5・exp_005 と整合）。標準configは randomness 未設定でランダム seed になるため明示。
randomness = dict(deterministic=False, seed=0)
