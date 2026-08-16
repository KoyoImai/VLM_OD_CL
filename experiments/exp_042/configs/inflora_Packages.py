# =============================================================================
# exp_042: InfLoRA（232 層・r=16） 学習 config / t=06 Packages
#
# 【自動生成】experiments/exp_042/gen_configs.py。直接編集しない。
#
# データ・スケジュール（3000 iter / batch 2 / dn 無効 / num_classes 256 / seed 0 /
# COCO 形式 pipeline）は exp_038 条件C のタスク config を継承し、本 config が
# 上書きするのは手法（model）と optimizer（lr 1e-4 / wd 1e-4。design.md §2.1）だけ。
#
# 逐次学習では前タスクの θ_{t-1} を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '../../exp_038/configs/lora_odinw13_condC_Packages.py'

# 手法: InfLoRA（design.md §2.3）。挿入箇所・r=16 は exp_038 条件A/C と同一の 232 層を
# 継承し、alpha=16（scaling=1 = 公式の ΔW=B·A）に変更。design_path はドライバが
# --cfg-options model.inflora.design_path= で渡す（inflora_prepare.py の出力）。
model = dict(
    type='GroundingDINOInfLoRA',
    lora=dict(alpha=16),
    inflora=dict(design_path=None))

# lr 1e-4 / wd 1e-4（exp_038 条件C は lr 1e-4 / wd 1e-2。design.md §2.1）
optim_wrapper = dict(optimizer=dict(lr=1e-4, weight_decay=1e-4))
