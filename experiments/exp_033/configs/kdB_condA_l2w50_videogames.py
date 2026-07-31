# =============================================================================
# exp_033-B: 条件A + 蒸留B（テキスト（text_feat_map 出力）・L2）+ リプレイ / videogames / λ = 5.0
#   exp_028 との差は **蒸留係数 loss_weight のみ**（1.0 -> 5.0）。
#   重みの決め方は固定係数（weight_mode の既定 'const'）。
#   design: experiments/exp_033/design.md ／ 共通仕様 experiments/exp_028/design.md
# =============================================================================
_base_ = '../../exp_028/configs/kdB_condA_l2_videogames.py'

model = dict(kd=dict(loss_weight=5.0))
