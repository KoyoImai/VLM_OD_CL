# =============================================================================
# exp_045: EWC（全モジュール・λはドライバ指定） / documents（逐次 t=6）
#
# 【自動生成】experiments/exp_045/gen_configs.py。直接編集しない。
#
# データ・スケジュール（バッファ不使用・batch 4/GPU・DefaultSampler・ODVG・dn 有効）は
# ../../exp_040/configs/replayfree_documents.py を継承し、
# 本 config が上書きするのはモデルだけ（design.md §2.1）。
# 逐次学習では前タスクの θ_{t-1} を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '../../exp_040/configs/replayfree_documents.py'

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
#
# encoder=dict(num_cp=0) は必須（2026-08-19 修正）。既定の num_cp=6 は fairscale の
# **再入型** checkpoint_wrapper を encoder に掛けるが、EWC ペナルティは θ を forward の
# 外で直接使うため勾配が 2 経路になり、再入型 checkpoint の入れ子 backward と DDP の
# 組で「Expected to mark a variable ready only once」で落ちる（t=2 でペナルティが
# 活性化した時点でクラスタで発生。最小再現で cp あり=エラー / cp なし=正常を実証済み）。
# t=1 はペナルティ不活性のため num_cp=6 でも通る。checkpointing の有無は勾配を
# 浮動小数点レベルで変えるため、対照（replayfree/InfLoRA は num_cp=6）との但し書きは
# design.md §6 に記録。
model = dict(
    type='EWCGroundingDINO',
    encoder=dict(num_cp=0),
    ewc=dict(target_components='all', lam=0.0, state_path=None))
