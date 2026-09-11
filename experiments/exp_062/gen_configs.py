#!/usr/bin/env python3
# =============================================================================
# exp_062: バッチ内訳感度（内訳 [4,2,2]・batch 8）の config 生成。Ours=kdE / ER=replay。
#   ベース（exp_061 と同一の kdE/replay）を解決し、
#     (1) train_dataloader のバッファ ann_file を exp_058/buffer の各サイズに差し替え
#         （base=1x は元のまま＝exp_023/buffer を継承。b2..b5 のみ差し替え）
#     (2) サンプラーの内訳を上書き: t>=2 は source_ratio=[4,2,2]・batch_size=8。
#         t=1 は 2 ソース（現在・参照）で base の [2,1]batch6（=現在4:参照2）を維持。
#   手法 2 × サイズ 5（base..b5）× ドメイン 6 = 60 本。
# =============================================================================
import os, pprint
from mmengine.config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
BUF = os.path.join(ROOT, 'experiments', 'exp_058', 'buffer')
OUT = os.path.join(HERE, 'configs')
os.makedirs(OUT, exist_ok=True)

DOMS = ['underwater', 'electromagnetic', 'videogames', 'aerial', 'microscopic', 'documents']
# base は 1x（buffer 差し替えなし）。b2..b5 は exp_058/buffer に差し替え。
CONDS = {'base': None, 'b2': (2000, 1000), 'b3': (3000, 1500),
         'b4': (4000, 2000), 'b5': (5000, 2500)}
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
    for cond, sizes in CONDS.items():
        for t, dom in enumerate(DOMS, 1):
            cfg = Config.fromfile(base_cfg(method, dom))
            dl = plain(cfg.train_dataloader)
            # (1) バッファ差し替え（base は無し）
            if sizes is not None:
                ref_n, past_n = sizes
                n = rewrite(dl, ref_n, past_n)
                assert n == 1 + max(0, t - 1), (method, cond, dom, n)
            else:
                n = 0
            # (2) 内訳の上書き（t>=2 のみ [4,2,2] batch 8。t=1 は base のまま）
            if t >= 2:
                assert dl['sampler']['source_ratio'] == [4, 1, 1], (dom, dl['sampler'])
                dl['sampler']['source_ratio'] = [4, 2, 2]
                dl['sampler']['batch_size'] = 8
            dl['_delete_'] = True
            body = f'''# exp_062 バッチ内訳感度 {method}/{cond}（内訳 [4,2,2] batch8・t>=2）/ {dom}（t={t}）
# 【自動生成】experiments/exp_062/gen_configs.py。直接編集しない。
# ベースとの差分: バッファ ann_file {n} 箇所 ＋ 内訳（t>=2: source_ratio[4,2,2]/batch8）。
# t=1 は 2 ソースのため base の [2,1]batch6（現在4:参照2）を維持。
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
print('exp_062 生成:', total, '本')
