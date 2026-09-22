#!/usr/bin/env python3
"""exp_062 実行前検証（静的＋動的）。
  1. 全 config が build でき、ベースとの diff が想定内
     （exp_061: バッファ ann_file のみ / exp_062: バッファ ann_file ＋ 内訳 [4,2,2]/batch8(t>=2)）
  2. 参照するバッファ（exp_058/buffer）が実在する
  3. DataLoader 本体と sampler の batch_size が一致（2026-09-22 のバグの再発防止:
     sampler だけ 8 だと DataLoader が 6 個ずつに切り直し、2 バッチ目以降で内訳が崩れる）
  4. 【動的】代表 config で実際に dataloader を構築し、5 バッチ分の内訳
     （先頭 4=現在ドメイン / 末尾 2=現在ドメイン以外）を位置ごとに検査
     （1-step 検証では初回バッチしか見えず今回のバグを見逃した教訓）
使い方: python experiments/exp_062/check_exp062_setup.py
"""
import glob, os, sys
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from mmengine.config import Config
from mmengine.registry import init_default_scope
init_default_scope('mmdet')
EXP = 'experiments/exp_062'
ok_all = True
def ck(name, ok, d=''):
    global ok_all; ok_all &= ok
    print(f'[{"OK" if ok else "NG"}] {name}' + (f'  {d}' if d else ''))

cfgs = sorted(glob.glob(f'{EXP}/configs/*.py'))
exp_n = 62
expect = 48 if exp_n == 61 else 60
ck(f'config 本数 {len(cfgs)}=={expect}', len(cfgs) == expect)

# build + サンプラー/バッファ検算
bad = []
miss_buf = set()
for p in cfgs:
    c = Config.fromfile(p)
    s = c.train_dataloader.sampler
    dom_is_t1 = any(p.endswith(f'_{d}.py') for d in ['underwater']) and '_underwater.py' in p
    # t は config 名から復元できないので sampler のソース数で t>=2 を判定
    ratio = s['source_ratio']; bs = s['batch_size']
    if exp_n == 62 and len(ratio) == 3:
        if ratio != [4, 2, 2] or bs != 8:
            bad.append((os.path.basename(p), ratio, bs))
    if exp_n == 61 and len(ratio) == 3:
        if ratio != [4, 1, 1] or bs != 6:
            bad.append((os.path.basename(p), ratio, bs))
    # DataLoader 本体と sampler の batch_size 一致（★2026-09-22 バグの検査）
    if c.train_dataloader.batch_size != bs:
        bad.append((os.path.basename(p), 'outer!=sampler',
                    c.train_dataloader.batch_size, bs))
    # バッファ実在
    import re
    for a in re.findall(r"'ann_file': '([^']+)'", str(c.train_dataloader)):
        if 'exp_058/buffer' in a and not os.path.exists(a):
            miss_buf.add(a)
ck('全 config の内訳・batch_size 整合が想定どおり', not bad, str(bad[:3]))
ck('参照バッファ（exp_058/buffer）が実在', not miss_buf, f'欠落 {len(miss_buf)} 本' if miss_buf else '')

# ---- 動的検査: 実 dataloader を構築し 5 バッチの内訳を位置で検査 ----------------
from mmengine.runner import Runner

def check_batches(cfg_path, dom, n_batches=5):
    c = Config.fromfile(cfg_path)
    dl_cfg = c.train_dataloader
    dl_cfg.num_workers = 2
    dl_cfg.persistent_workers = False
    loader = Runner.build_dataloader(dl_cfg, seed=0)
    bs = dl_cfg.batch_size
    cur_key = f'/{dom}/train'
    issues = []
    it = iter(loader)
    for b in range(n_batches):
        data = next(it)
        paths = [ds.img_path for ds in data['data_samples']]
        if len(paths) != bs:
            issues.append(f'batch{b}: 実バッチ {len(paths)} != {bs}')
            continue
        for i, pth in enumerate(paths[:4]):
            if cur_key not in pth:
                issues.append(f'batch{b} 先頭位置{i} が現在ドメイン以外: {pth}')
        for i, pth in enumerate(paths[bs - 2:], bs - 2):
            if cur_key in pth:
                issues.append(f'batch{b} 末尾位置{i} が現在ドメイン: {pth}')
    return issues

for cfg_path, dom in [(f'{EXP}/configs/ours_b3_electromagnetic.py', 'electromagnetic'),
                      (f'{EXP}/configs/er_b3_electromagnetic.py', 'electromagnetic'),
                      (f'{EXP}/configs/ours_base_documents.py', 'documents')]:
    try:
        issues = check_batches(cfg_path, dom)
        ck(f'動的 5 バッチ内訳 {os.path.basename(cfg_path)}', not issues, '; '.join(issues[:3]))
    except Exception as e:
        ck(f'動的 5 バッチ内訳 {os.path.basename(cfg_path)}', False, f'{type(e).__name__}: {e}')

print(f'\n{"ALL OK" if ok_all else "NG あり"}')
sys.exit(0 if ok_all else 1)
