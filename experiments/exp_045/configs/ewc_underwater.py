# =============================================================================
# exp_045: EWC（全モジュール・λはドライバ指定） / underwater（逐次 t=1）
#
# 【自動生成】experiments/exp_045/gen_configs.py。直接編集しない。
#
# データ・スケジュール（バッファ不使用・batch 4/GPU・DefaultSampler・ODVG・dn 有効）は
# ../../exp_024/configs/fullft_replayfree_underwater.py を継承し、
# 本 config が上書きするのはモデルだけ（design.md §2.1）。
# 逐次学習では前タスクの θ_{t-1} を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '../../exp_024/configs/fullft_replayfree_underwater.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'projects.lora_cl',
        'projects.ewc_cl',
    ],
    allow_failed_imports=False)

# EWC（design.md §2.2）: 学習対象 = EWC 対象 = 全モジュール。
# λ（本走 10^3）と EWC 状態はドライバが --cfg-options model.ewc.lam= /
# model.ewc.state_path= で毎タスク明示する（t=1 は状態無し = ペナルティ不活性）。
# dn は RF100 枠に従い有効のまま（use_dn を渡さない = 既定 True）。
model = dict(
    type='EWCGroundingDINO',
    ewc=dict(target_components='all', lam=0.0, state_path=None))
