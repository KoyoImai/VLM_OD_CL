#!/usr/bin/env python3
# =============================================================================
# exp_059: 蒸留係数 λ 感度分析の config 生成（design.md §2）
#   w50/w150/w200 × 6 ドメイン = 18 本。exp_035（t1-3）/exp_040（t4-6）の kdE を
#   継承し、上書きは model.kd.loss_weight と ckpt 保存方針だけ。
#   λ は config に焼き込み、ドライバは渡さない（exp_033 事故の再発防止。design §1.1）。
# =============================================================================
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(HERE, 'configs')
os.makedirs(OUT, exist_ok=True)

DOMS = ['underwater', 'electromagnetic', 'videogames', 'aerial',
        'microscopic', 'documents']
CONDS = {'w50': 5.0, 'w150': 15.0, 'w200': 20.0}


def base_cfg(dom):
    if dom in DOMS[:3]:
        return f'{ROOT}/experiments/exp_035/configs/kdE_condA_l2w100_{dom}.py'
    return f'{ROOT}/experiments/exp_040/configs/kdE_{dom}.py'


for cond, lam in CONDS.items():
    for t, dom in enumerate(DOMS, 1):
        body = f'''# =============================================================================
# exp_059: 蒸留係数感度 {cond}（λ={lam}）/ {dom}（t={t}）
# 【自動生成】experiments/exp_059/gen_configs.py。直接編集しない。
# ベース kdE（λ=10）との差分は loss_weight と ckpt 保存方針のみ。
# λ は本 config に焼き込み。ドライバは loss_weight を渡さない（exp_033 事故防止）。
# =============================================================================
_base_ = '{base_cfg(dom)}'

model = dict(kd=dict(loss_weight={lam}))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=-1,
                    save_optimizer=False, save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=50))
'''
        open(os.path.join(OUT, f'{cond}_{dom}.py'), 'w').write(body)
print('生成: 18 本')
