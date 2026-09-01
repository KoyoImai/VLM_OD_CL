# =============================================================================
# exp_059: 蒸留係数感度 w50（λ=5.0）/ microscopic（t=5）
# 【自動生成】experiments/exp_059/gen_configs.py。直接編集しない。
# ベース kdE（λ=10）との差分は loss_weight と ckpt 保存方針のみ。
# λ は本 config に焼き込み。ドライバは loss_weight を渡さない（exp_033 事故防止）。
# =============================================================================
_base_ = '/workspace/kouyou/mmdetection/experiments/exp_040/configs/kdE_microscopic.py'

model = dict(kd=dict(loss_weight=5.0))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=-1,
                    save_optimizer=False, save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=50))
