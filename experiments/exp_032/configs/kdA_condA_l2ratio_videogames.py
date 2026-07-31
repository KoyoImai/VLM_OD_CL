# =============================================================================
# exp_032-A: 条件A + 蒸留A（画像（neck 出力）・L2）+ リプレイ / videogames
#   exp_028 との差は**蒸留重みの決め方だけ**。
#   c_t = target_ratio * L_det / L_KD を毎イテレーション計算し、蒸留損失の値が
#   検出損失の target_ratio 倍になるようにする（design: experiments/exp_032/design.md §1）。
# =============================================================================
_base_ = '../../exp_028/configs/kdA_condA_l2_videogames.py'

model = dict(
    kd=dict(
        weight_mode='ratio',   # c_t を毎イテレーション計算
        target_ratio=0.1,      # 蒸留損失 = 検出損失の 10%
        max_weight=100.0))     # 学習初期の暴走を抑える上限（§1.3）
