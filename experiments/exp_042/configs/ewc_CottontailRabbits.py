# =============================================================================
# exp_042: EWC（全モジュール） 学習 config / t=03 CottontailRabbits
#
# 【自動生成】experiments/exp_042/gen_configs.py。直接編集しない。
#
# データ・スケジュール（3000 iter / batch 2 / dn 無効 / num_classes 256 / seed 0 /
# COCO 形式 pipeline）は exp_038 条件C のタスク config を継承し、本 config が
# 上書きするのは手法（model）と optimizer（lr 1e-4 / wd 1e-4。design.md §2.1）だけ。
#
# 逐次学習では前タスクの θ_{t-1} を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '../../exp_038/configs/lora_odinw13_condC_CottontailRabbits.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'projects.lora_cl',
        'projects.ewc_cl',
        'mmdet.models.dense_heads.nodn_grounding_dino_head',
    ],
    allow_failed_imports=False)

# 手法: 継承した model から LoRA を除き、EWC 検出器に差し替える（design.md §2.2）。
# 学習対象 = EWC 対象 = 全モジュール（2026-08-15 確定）。
# λ と state_path は**ドライバが --cfg-options model.ewc.lam= / model.ewc.state_path=
# で毎回明示する**（t=1 は state_path=None でペナルティ不活性 = 素のフル FT）。
model = _base_.model
model.pop('lora')
model.update(dict(
    type='EWCGroundingDINO',
    ewc=dict(target_components='all', lam=0.0, state_path=None)))

# lr 1e-4 / wd 1e-4（exp_038 条件C は lr 1e-4 / wd 1e-2。design.md §2.1）
optim_wrapper = dict(optimizer=dict(lr=1e-4, weight_decay=1e-4))
