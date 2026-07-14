# exp_021: DitHub 再現・標準版 (documents)
# 20 epoch (warmup 10 + specialization 10) / lr 1e-4 / seed 0 / dn 有効
# 実装: mmdet/models/layers/dithub_layers.py ほか (implementation_plan.md の台帳参照)
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_documents.py'

custom_imports = dict(
    imports=[
        'mmdet.models.layers.dithub_layers',
        'mmdet.models.detectors.dithub_grounding_dino',
        'mmdet.engine.hooks.dithub_phase_hook',
    ],
    allow_failed_imports=False)

randomness = dict(deterministic=False, seed=0)

model = dict(
    type='DitHubGroundingDINO',
    dithub_classes={{_base_.class_name}},
    # 上流 (fairscale) の再入版 activation checkpointing を無効化する。
    # 再入 backward は DDP のパラメータ二重マークを起こし、可変な使用パラメータ
    # 集合 (クラス別 A) と両立しない。代わりに DitHubGroundingDINO 側で
    # 非再入版 (torch, use_reentrant=False) を encoder_cp 層に掛け直す。
    # 出力・勾配は checkpointing なしと数学的に同一 (検証済み)。
    encoder=dict(num_cp=0),
    encoder_cp=6)

# フェーズ切替: 公式の等分規則 (学習量の半分) を epoch に写像
custom_hooks = [dict(type='DitHubPhaseHook', warmup_epochs=10)]

# specialization 中は選択されなかったクラスの A が不使用になるため必須
# (上流 num_cp=0 + 非再入 checkpointing とセット。再入版とは両立しない)
find_unused_parameters = True

# 裁定④: 報告用 best は specialization 期 (ep11-20) から選ぶため、
# 全該当 epoch の ckpt を保持する (interval=1, 直近11個 = ep10-20)
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=11, save_best=None))
