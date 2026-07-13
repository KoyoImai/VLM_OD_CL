# exp_021: DitHub 再現・標準版 (microscopic)
# 20 epoch (warmup 10 + specialization 10) / lr 1e-4 / seed 0 / dn 有効
# 実装: mmdet/models/layers/dithub_layers.py ほか (implementation_plan.md の台帳参照)
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_microscopic.py'

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
    # activation checkpointing を無効化 (数学的に同一。再入 backward が
    # find_unused_parameters=True と両立しないため。GPU メモリは余裕あり)
    encoder=dict(num_cp=0))

# フェーズ切替: 公式の等分規則 (学習量の半分) を epoch に写像
custom_hooks = [dict(type='DitHubPhaseHook', warmup_epochs=10)]

# specialization 中は選択されなかったクラスの A が不使用になるため必須
# (num_cp=0 とセット。checkpointing 有効時は再入 backward と衝突する)
find_unused_parameters = True

# 裁定④: 報告用 best は specialization 期 (ep11-20) から選ぶため、
# 全該当 epoch の ckpt を保持する (interval=1, 直近11個 = ep10-20)
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=11, save_best=None))
