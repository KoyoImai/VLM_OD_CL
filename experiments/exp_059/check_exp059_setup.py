"""exp_059 実行前検証（design.md §1.1・§2）。

使い方:
    python experiments/exp_059/check_exp059_setup.py          # 静的（GPU 不要）
    python experiments/exp_059/check_exp059_setup.py --step   # + λ 比例の実測（GPU 1 枚）

検証項目
  1. 18 本の config build。ベース kdE との全キー diff が
     「kd.loss_weight ＋ ckpt 保存方針」だけであること。実効 loss_weight の値も確認
  2. （--step）同一バッチ・決定的モードで 1 step の loss_kd を w50/w200 で実測し、
     比が 4.0 に一致すること（= λ が実際に効いている。exp_033 事故の再発防止）
"""
import copy
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
os.chdir(ROOT)
THETA0 = os.path.join(
    ROOT, 'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det'
    '_20231204_095047-b448804b.pth')
DOMS = ['underwater', 'electromagnetic', 'videogames', 'aerial',
        'microscopic', 'documents']
CONDS = {'w50': 5.0, 'w150': 15.0, 'w200': 20.0}
results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


def flat(x, pre=''):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flat(v, f'{pre}.{k}' if pre else str(k)))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out.update(flat(v, f'{pre}[{i}]'))
    else:
        out[pre] = x
    return out


def base_cfg(dom):
    if dom in DOMS[:3]:
        return f'{ROOT}/experiments/exp_035/configs/kdE_condA_l2w100_{dom}.py'
    return f'{ROOT}/experiments/exp_040/configs/kdE_{dom}.py'


if __name__ == '__main__':
    do_step = '--step' in sys.argv
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet')

    bad, n_ok, cfgs = [], 0, {}
    for cond, lam in CONDS.items():
        for dom in DOMS:
            try:
                c = Config.fromfile(f'{HERE}/configs/{cond}_{dom}.py')
                cfgs[(cond, dom)] = c
            except Exception as e:
                bad.append(f'{cond}_{dom}: build {type(e).__name__}')
                continue
            if c.model['kd']['loss_weight'] != lam:
                bad.append(f'{cond}_{dom}: loss_weight={c.model["kd"]["loss_weight"]}')
            ref = Config.fromfile(base_cfg(dom))
            fa, fb = flat(ref.to_dict()), flat(c.to_dict())
            diffs = [k for k in set(fa) | set(fb)
                     if fa.get(k, '<無>') != fb.get(k, '<無>')
                     and not k.startswith('work_dir')]
            extra = [k for k in diffs
                     if 'loss_weight' not in k
                     and 'default_hooks.checkpoint' not in k
                     and 'default_hooks.logger' not in k]
            if extra:
                bad.append(f'{cond}_{dom}: 想定外差分 {extra[:2]}')
            n_ok += 1
    check(1, '18 本 build・差分 = loss_weight と ckpt 方針のみ・実効 λ 確認',
          n_ok == 18 and not bad, f'{bad[:3]}' if bad else '18/18')

    if do_step:
        from mmengine.runner import Runner
        from mmengine.runner.checkpoint import load_checkpoint
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        kd_vals = {}
        loader = None
        batch = None
        for cond in ('w50', 'w200'):
            c = cfgs[(cond, 'underwater')]
            import_modules_from_strings(**c['custom_imports'])
            m = copy.deepcopy(c.model)
            m['teacher_ckpt'] = THETA0
            model = MODELS.build(m).cuda()
            model.init_weights()
            load_checkpoint(model, THETA0, map_location='cpu')
            if loader is None:
                loader = Runner.build_dataloader(copy.deepcopy(c.train_dataloader))
                batch = next(iter(loader))
            model.eval(); model.training = True     # dropout を止めて決定化
            with torch.no_grad():
                data = model.data_preprocessor(copy.deepcopy(batch), True)
                kd_vals[cond] = float(model.loss(data['inputs'],
                                                 data['data_samples'])['loss_kd'])
            del model
            torch.cuda.empty_cache()
        ratio = kd_vals['w200'] / kd_vals['w50']
        check(2, 'loss_kd の λ 比例（w200/w50 = 4.0）', abs(ratio - 4.0) < 1e-3,
              f'w50={kd_vals["w50"]:.3e} w200={kd_vals["w200"]:.3e} 比={ratio:.6f}')

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
