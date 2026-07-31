# =============================================================================
# exp_033-E: 条件A + 蒸留E（3項平均・L2）+ リプレイ / electromagnetic / λ = 5.0
#   exp_028 との差は **蒸留係数 loss_weight のみ**（1.0 -> 5.0）。
#   重みの決め方は固定係数（weight_mode の既定 'const'）。
#   design: experiments/exp_033/design.md ／ 共通仕様 experiments/exp_028/design.md
# =============================================================================
_base_ = '../../exp_028/configs/kdE_condA_l2_electromagnetic.py'

model = dict(kd=dict(loss_weight=5.0))
