"""exp_047 実行前検証（design.md §4.2）。

使い方:
    python experiments/exp_047/check_exp047_setup.py          # config のみ（GPU 不要）
    python experiments/exp_047/check_exp047_setup.py --step   # + 実データ 1 step・マージ（GPU）

検証項目
  1. 6 本の config が build でき、学習設定・データ経路が exp_045（replayfree 継承）と一致。
     機械的 diff: model の差分が exp_045 inflora に対して「type と inflora キーの有無」
     だけ（lora 設定は同一）
  2. （--step）232 層挿入・A/B のみ学習（464 params・5.67M）。**学習率の分布が
     design.md §2.2 の決定どおり**（Swin 102 / BERT 144 params が 1e-5、
     encoder 216 / text_feat_map 2 params が 1e-4）
  3. （--step）ODVG 実データ 1 step が有限（dn 有効）。VRAM 実測
  4. （--step）merge_lora（scaling=1）→ plain モデルで欠落キーなくロードできること
"""
import copy
import os
import sys
import tempfile

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CFG_DIR = os.path.join(HERE, 'configs')

FULL_ORDER = ['underwater', 'electromagnetic', 'videogames',
              'aerial', 'microscopic', 'documents']
THETA0 = os.path.join(
    ROOT, 'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det'
    '_20231204_095047-b448804b.pth')
results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


def _flat(d, pre=''):
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else []):
        key = f'{pre}.{k}' if pre else k
        if isinstance(v, dict):
            out.update(_flat(v, key))
        else:
            out[key] = str(v) if isinstance(v, (list, tuple)) else v
    return out


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


if __name__ == '__main__':
    do_step = '--step' in sys.argv
    init_default_scope('mmdet')

    # --- 1. build・設定・機械的 diff（exp_045 inflora と）--------------------
    cfgs, bad = {}, []
    for i, d in enumerate(FULL_ORDER):
        p = os.path.join(CFG_DIR, f'lora_{d}.py')
        try:
            c = Config.fromfile(p)
            cfgs[d] = c
        except Exception as e:
            bad.append(f'{d}: {type(e).__name__}')
            continue
        for label, got, want in (
                ('max_epochs', c.train_cfg.get('max_epochs'), 20),
                ('lr', c.optim_wrapper['optimizer']['lr'], 0.0001),
                ('wd', c.optim_wrapper['optimizer']['weight_decay'], 0.0001),
                ('batch', c.train_dataloader['batch_size'], 4),
                ('sampler', c.train_dataloader['sampler']['type'], 'DefaultSampler'),
                ('seed', c.randomness['seed'], 0),
                ('type', c.model['type'], 'GroundingDINOLoRA'),
                ('r', c.model['lora']['r'], 16),
                ('alpha', c.model['lora']['alpha'], 16)):
            if got != want:
                bad.append(f'{d}: {label}={got}(期待 {want})')
        if 'use_dn' in c.model:
            bad.append(f'{d}: use_dn が上書きされている')
        ref = Config.fromfile(os.path.join(
            ROOT, 'experiments', 'exp_045', 'configs', f'inflora_{d}.py'))
        if _plain(ref.train_dataloader) != _plain(c.train_dataloader):
            bad.append(f'{d}: train_dataloader が exp_045 と不一致')
        fa, fb = _flat(ref.model), _flat(c.model)
        for k in sorted(set(fa) | set(fb)):
            if fa.get(k) == fb.get(k) or k in ('type', 'inflora.design_path'):
                continue
            bad.append(f'{d}: model.{k} 045={fa.get(k)!r} 047={fb.get(k)!r}')
        oa, ob = _flat(ref.optim_wrapper), _flat(c.optim_wrapper)
        if oa != ob:
            bad.append(f'{d}: optim_wrapper が exp_045 と不一致')
    check(1, '6 本 build・設定・機械的 diff（exp_045 inflora と type/inflora のみ差）',
          len(cfgs) == 6 and not bad,
          f'{len(cfgs)}/6 / 不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 2〜4. 挿入・学習率分布・1 step・マージ（--step）--------------------
    if do_step:
        from mmengine.dataset import pseudo_collate
        from mmengine.optim import build_optim_wrapper
        from mmengine.runner import Runner
        from mmengine.runner.checkpoint import load_checkpoint
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import DATASETS, MODELS

        c = cfgs['underwater']
        import_modules_from_strings(**c['custom_imports'])
        model = MODELS.build(copy.deepcopy(c.model))
        tr = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        n_tr = sum(p.numel() for _, p in tr)
        ok_ins = (len(tr) == 464 and all('lora_' in n for n, _ in tr))

        # 学習率の分布（design.md §2.2 の決定: Swin/BERT 1e-5、encoder/tfm 1e-4）
        ow = build_optim_wrapper(model, copy.deepcopy(c.optim_wrapper))
        id2lr = {id(p): g['lr'] for g in ow.optimizer.param_groups
                 for p in g['params']}
        import collections
        dist = collections.Counter(
            (n.split('.')[0], id2lr.get(id(p))) for n, p in tr)
        want = {('backbone', 1e-05): 102, ('language_model', 1e-05): 144,
                ('encoder', 0.0001): 216, ('text_feat_map', 0.0001): 2}
        ok_lr = dict(dist) == want
        check(2, '232 層挿入・A/B のみ 464 params・学習率分布が決定どおり',
              ok_ins and ok_lr,
              f'{len(tr)} params ({n_tr/1e6:.2f}M) / lr 分布 {dict(dist)}')

        load_checkpoint(model, THETA0, map_location='cpu')
        model = model.cuda().train()
        torch.cuda.reset_peak_memory_stats()
        loader = Runner.build_dataloader(copy.deepcopy(c.train_dataloader))
        batch = next(iter(loader))
        data = model.data_preprocessor(batch, True)
        losses = model.loss(data['inputs'], data['data_samples'])
        total = sum(v.sum() for v in losses.values()
                    if isinstance(v, torch.Tensor) and v.requires_grad)
        total.backward()
        vram = torch.cuda.max_memory_allocated() / 2**30
        check(3, 'ODVG 実データ 1 step が有限（dn 有効）',
              bool(torch.isfinite(total)),
              f'loss={float(total):.2f} VRAM={vram:.1f}GB')

        tmp = tempfile.mkdtemp(prefix='exp047_check_')
        ck = os.path.join(tmp, 'epoch_20.pth')
        torch.save({'state_dict': model.state_dict()}, ck)
        merged = os.path.join(tmp, 'theta_t1.pth')
        import subprocess
        r = subprocess.run(
            [sys.executable, 'projects/lora_cl/merge_lora.py', ck, merged],
            capture_output=True, text=True, cwd=ROOT)
        ok_m = r.returncode == 0 and os.path.exists(merged)
        n_missing = -1
        if ok_m:
            ev = Config.fromfile(os.path.join(
                ROOT, 'experiments', 'exp_023', 'configs', 'eval_underwater.py'))
            plain = MODELS.build(copy.deepcopy(ev.model))
            ckd = torch.load(merged, map_location='cpu')
            sd = ckd.get('state_dict', ckd)
            n_missing = len(set(plain.state_dict()) - set(sd))
            del plain
        check(4, 'merge_lora → plain モデルで欠落キーなし', ok_m and n_missing == 0,
              (r.stdout.strip().splitlines()[-1] if ok_m else
               (r.stderr or r.stdout)[-200:]) + f' / 欠落 {n_missing}')
        import shutil
        shutil.rmtree(tmp)
        del model, loader
        torch.cuda.empty_cache()

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
