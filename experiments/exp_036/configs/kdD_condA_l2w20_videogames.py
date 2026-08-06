# =============================================================================
# exp_036-D: 条件A + 蒸留D（融合後（feature enhancer 出力）・L2）+ リプレイ / videogames / λ = 2.0
#   exp_028 との差は **蒸留係数 loss_weight のみ**（1.0 -> 2.0）。
#   exp_033 と同一内容だが、実行側で LAMBDA を明示して実際に λ が効くようにしたもの。
#   design: experiments/exp_036/design.md ／ 共通仕様 experiments/exp_028/design.md
# =============================================================================
_base_ = '../../exp_028/configs/kdD_condA_l2_videogames.py'

model = dict(kd=dict(loss_weight=2.0))
