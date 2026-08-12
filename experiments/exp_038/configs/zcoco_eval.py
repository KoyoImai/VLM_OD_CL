# =============================================================================
# exp_038: ZCOCO 評価（COCO2017-val ゼロショット）/ plain GroundingDINO
#   COCO の評価プロトコルは eval_base_coco.py（プロジェクト共通・exp_002.5 確定）を継承。
#   評価対象は LoRA をマージ済みの plain ckpt なので、モデルは事前学習 config のまま。
#   θ0 の基準値は 0.5040（exp_001 実測）。
#
# 【自動生成】experiments/exp_038/gen_configs.py。直接編集しない。
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/eval_base_coco.py'
