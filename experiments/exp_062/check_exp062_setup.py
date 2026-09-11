#!/usr/bin/env python3
"""exp_062 実行前検証（静的）。
  1. 全 config が build でき、ベースとの diff が想定内
     （exp_061: バッファ ann_file のみ / exp_062: バッファ ann_file ＋ 内訳 [4,2,2]/batch8(t>=2)）
  2. 参照するバッファ（exp_058/buffer）が実在する
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
    # バッファ実在
    import re
    for a in re.findall(r"'ann_file': '([^']+)'", str(c.train_dataloader)):
        if 'exp_058/buffer' in a and not os.path.exists(a):
            miss_buf.add(a)
ck('全 config の内訳が想定どおり', not bad, str(bad[:3]))
ck('参照バッファ（exp_058/buffer）が実在', not miss_buf, f'欠落 {len(miss_buf)} 本' if miss_buf else '')
print(f'\n{"ALL OK" if ok_all else "NG あり"}')
sys.exit(0 if ok_all else 1)
