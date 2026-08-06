# =============================================================================
# exp_035-E: 条件A + 蒸留E（3項平均（画像・テキスト・融合後）・L2）+ リプレイ / videogames / λ = 10.0
#   exp_028 との差は **蒸留係数 loss_weight のみ**（1.0 -> 10.0）。
#   重みの決め方は固定係数（weight_mode の既定 'const'）。
#   design: experiments/exp_035/design.md ／ 共通仕様 experiments/exp_028/design.md
# =============================================================================
_base_ = '../../exp_028/configs/kdE_condA_l2_videogames.py'

model = dict(kd=dict(loss_weight=10.0))
