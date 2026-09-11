#!/usr/bin/env python3
# =============================================================================
# exp_061: バッファサイズ感度（Ours=kdE / ER=replay）の config 生成。
#   ベース（Ours: exp_035 t1-3 / exp_040 t4-6 kdE、ER: exp_027 t1-3 / exp_040 t4-6 replay）
#   を解決し、train_dataloader のバッファ ann_file だけを exp_058/buffer の拡張版に
#   差し替える（内訳サンプラーは base のまま＝[4,1,1] 系。exp_058 と同一方式）。
#   base(1x) は既存実測を再利用するので生成しない。b2..b5 のみ生成。
#   手法 2 × サイズ 4 × ドメイン 6 = 48 本。
# =============================================================================
import os, pprint
from mmengine.config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
BUF = os.path.join(ROOT, 'experiments', 'exp_058', 'buffer')   # 既存の入れ子拡張を再利用
OUT = os.path.join(HERE, 'configs')
os.makedirs(OUT, exist_ok=True)

DOMS = ['underwater', 'electromagnetic', 'videogames', 'aerial', 'microscopic', 'documents']
CONDS = {'b2': (2000, 1000), 'b3': (3000, 1500), 'b4': (4000, 2000), 'b5': (5000, 2500)}
METHODS = ['ours', 'er']


def base_cfg(method, dom):
    if method == 'ours':
        return (f'{ROOT}/experiments/exp_035/configs/kdE_condA_l2w100_{dom}.py'
                if dom in DOMS[:3] else f'{ROOT}/experiments/exp_040/configs/kdE_{dom}.py')
    return (f'{ROOT}/experiments/exp_027/configs/fullft_replay_{dom}.py'
            if dom in DOMS[:3] else f'{ROOT}/experiments/exp_040/configs/replay_{dom}.py')


def rewrite(node, ref_n, past_n):
    n = 0
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if k == 'ann_file' and isinstance(v, str):
                if 'reference_o365v1_1000.odvg.json' in v:
                    node[k] = f'{BUF}/reference_o365v1_{ref_n}.odvg.json'; n += 1
                else:
                    for d in DOMS:
                        if v.endswith(f'{d}_500.odvg.json'):
                            node[k] = f'{BUF}/{d}_{past_n}.odvg.json'; n += 1
            else:
                n += rewrite(v, ref_n, past_n)
    elif isinstance(node, list):
        for v in node:
            n += rewrite(v, ref_n, past_n)
    return n


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [plain(v) for v in x]
        return tuple(t) if isinstance(x, tuple) else t
    return x


total = 0
for method in METHODS:
    for cond, (ref_n, past_n) in CONDS.items():
        for t, dom in enumerate(DOMS, 1):
            cfg = Config.fromfile(base_cfg(method, dom))
            dl = plain(cfg.train_dataloader)
            n = rewrite(dl, ref_n, past_n)
            expect = 1 + max(0, t - 1)   # 参照 1 + 過去 t-1 本
            assert n == expect, (method, cond, dom, n, expect)
            dl['_delete_'] = True
            body = f'''# exp_061 バッファサイズ感度 {method}/{cond}（参照 {ref_n}・過去 {past_n}/ドメイン）/ {dom}（t={t}）
# 【自動生成】experiments/exp_061/gen_configs.py。直接編集しない。
# ベースとの差分はバッファ ann_file（{n} 箇所）と ckpt 保存方針のみ（内訳は base のまま）。
_base_ = '{base_cfg(method, dom)}'

train_dataloader = \\
{pprint.pformat(dl, width=100, sort_dicts=False)}

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=-1,
                    save_optimizer=False, save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=50))
'''
            open(os.path.join(OUT, f'{method}_{cond}_{dom}.py'), 'w').write(body)
            total += 1
print('exp_061 生成:', total, '本')
