# exp_022 実行前検証: 各条件の学習対象パラメータが design.md の定義と一致するか。
#
# 検証内容（design.md §5）:
#   1. underwater の 5 config で optimizer の param group を実構築し、
#      lr>0 のパラメータ集合（トップモジュール単位）が条件の定義と一致すること
#   2. 学習するモジュールの実効 lr が条件3 と同一であること
#      （Swin/BERT 1e-5、neck/text_feat_map/encoder/level_embed 1e-4、凍結 0）
#   3. videogames / electromagnetic の config は paramwise_cfg が underwater と
#      同一であること（分割の定義がドメイン間で一致）
#   4. 学習可能+凍結 = 総パラメータ数（帳尻の確認）
#
# 実行: python experiments/exp_022/check_param_assignment.py
import sys

from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

register_all_modules()

CONDS = {
    'a1_img':    {'backbone', 'neck'},
    'a3_fus':    {'encoder', 'level_embed'},
    'a4_imgtxt': {'backbone', 'language_model', 'neck', 'text_feat_map'},
    'a5_imgfus': {'backbone', 'neck', 'encoder', 'level_embed'},
    'a6_txtfus': {'language_model', 'text_feat_map', 'encoder', 'level_embed'},
}
# 実効 lr の期待値（base lr 1e-4 × lr_mult）
LR_01 = {'backbone', 'language_model'}          # 0.1 → 1e-5
EXPECT_M = {  # design.md §2 の学習可能パラメータ数 (M)
    'a1_img': 29.64, 'a3_fus': 21.91, 'a4_imgtxt': 138.73,
    'a5_imgfus': 51.55, 'a6_txtfus': 131.00,
}
DOMAINS = ['underwater', 'videogames', 'electromagnetic']
ROOT = 'experiments/exp_022/configs'

ok = True


def fail(msg):
    global ok
    ok = False
    print(f'  NG: {msg}')


print('==== 検証 1・2・4: underwater で param group を実構築 ====')
for tag, expect_set in CONDS.items():
    cfg = Config.fromfile(f'{ROOT}/{tag}_underwater.py')
    model = MODELS.build(cfg.model)
    ow = build_optim_wrapper(model, cfg.optim_wrapper)
    trained, frozen = {}, {}
    lr_bad = []
    name_of = {p: n for n, p in model.named_parameters()}
    for g in ow.optimizer.param_groups:
        for p in g['params']:
            top = name_of[p].split('.')[0]
            if g['lr'] > 0:
                trained[top] = trained.get(top, 0) + p.numel()
                want = 1e-5 if top in LR_01 else 1e-4
                if abs(g['lr'] - want) > 1e-12:
                    lr_bad.append((name_of[p], g['lr'], want))
            else:
                frozen[top] = frozen.get(top, 0) + p.numel()
    total = sum(p.numel() for p in model.parameters())
    t_sum = sum(trained.values())
    print(f'[{tag}] 学習 {t_sum/1e6:.2f} M / 凍結 {sum(frozen.values())/1e6:.2f} M'
          f' / 合計 {total/1e6:.2f} M')
    if set(trained) != expect_set:
        fail(f'学習集合が不一致: got {sorted(trained)} want {sorted(expect_set)}')
    if any(m in trained for m in frozen):
        fail(f'同一モジュールが学習と凍結に混在: {set(trained) & set(frozen)}')
    if lr_bad:
        fail(f'実効 lr 不一致 {len(lr_bad)} 件 (例 {lr_bad[0]})')
    if abs(t_sum / 1e6 - EXPECT_M[tag]) > 0.02:
        fail(f'学習パラメータ数が design.md と不一致: {t_sum/1e6:.2f} M'
             f' vs {EXPECT_M[tag]} M')
    if t_sum + sum(frozen.values()) != total:
        fail('学習+凍結 != 総パラメータ')
    del model, ow

print('==== 検証 3: 他ドメインの paramwise_cfg が underwater と同一 ====')
for tag in CONDS:
    base = Config.fromfile(f'{ROOT}/{tag}_underwater.py').optim_wrapper[
        'paramwise_cfg']['custom_keys']
    for d in DOMAINS[1:]:
        other = Config.fromfile(f'{ROOT}/{tag}_{d}.py').optim_wrapper[
            'paramwise_cfg']['custom_keys']
        if dict(base) != dict(other):
            fail(f'{tag}_{d}: custom_keys が underwater と不一致')
    print(f'[{tag}] 3 ドメインで custom_keys 一致')

print()
print('==== 結果:', 'ALL OK' if ok else 'NG あり ====')
sys.exit(0 if ok else 1)
