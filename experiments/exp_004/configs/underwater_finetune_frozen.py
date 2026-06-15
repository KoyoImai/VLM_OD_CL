# =============================================================================
# exp_004: underwater 単一ドメイン fine-tune（backbone・言語エンコーダ 凍結）
# -----------------------------------------------------------------------------
# exp_003 の config を継承し、**学習可能パラメータの範囲のみ**変更する。
#   変更点: backbone・language_model を「0.1× 学習(lr_mult=0.1)」→「凍結(lr_mult=0.0)」。
#           → 検出 head/encoder/decoder/neck のみ学習。Swin backbone と BERT は更新しない。
#   据え置き（exp_003 と完全同一）: lr=1e-4 / 実効バッチ64(per-GPU8×4GPU×累積2) /
#           20 epochs(milestone[15]) / AdamW wd=1e-4 / clip_grad / 拡張・評価プロトコル。
#
# 目的: exp_003（全体 fine-tune, COCO −13.1%）と比較し、本体凍結が忘却をどれだけ抑えるか、
#       適応性能（underwater）がどれだけ犠牲になるかを測る（凍結範囲 ablation）。
#
# 参照: experiments/exp_004/design.md, experiments/exp_003/results/finetune_summary.md
# =============================================================================
_base_ = '../../exp_003/configs/underwater_finetune.py'

# backbone・言語エンコーダを凍結（lr_mult=0.0）。他の最適化設定は exp_003 を継承。
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.0),
            language_model=dict(lr_mult=0.0),
        )))
