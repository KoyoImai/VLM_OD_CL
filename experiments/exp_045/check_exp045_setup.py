"""exp_045 実行前検証（design.md §4.2）。本環境で実施し、クラスタへは push して持ち込む。

使い方:
    python experiments/exp_045/check_exp045_setup.py          # config のみ（GPU 不要）
    python experiments/exp_045/check_exp045_setup.py --build  # + model build（GPU 1 枚）
    python experiments/exp_045/check_exp045_setup.py --step   # + ODVG 実データの一連（GPU 1 枚）

検証項目
  1. 12 本の config が build でき、学習設定・データ経路がリプレイフリー系列と一致
     （20 epoch / lr 1e-4 / wd 1e-4 / batch 4 / DefaultSampler / ODVGDataset /
     dn 有効 = use_dn を上書きしない）。機械的 diff: model 以外は継承元と同一、
     model の差分は意図したキーだけ
  2. --cfg-options の到達性（model.ewc.lam / model.ewc.state_path /
     model.inflora.design_path。model. なしの誤記は届かない）
  3. （--build）EWC は全パラメータ学習可、InfLoRA は dn 有効の構成でも 232 層に挿入され
     lora_B のみ学習可
  4. （--step）EWC: ODVG 実データ 1 step が有限（状態なし）。estimate_fisher が走り、
     状態ありの 1 step で loss_ewc が有限
  5. （--step）InfLoRA: 設計（ランク不足の実測込み）→ 学習 1 step（A 不変・B 更新）→
     メモリ更新（T=6）→ メモリあり再設計、の一連が ODVG 経路で通ること
  6. VRAM 実測（batch 4/GPU）
"""
import copy
import os
import re
import subprocess
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
ALLOWED = {
    # encoder.num_cp=0 は 2026-08-19 の修正（EWC ペナルティ × 再入型 cp × DDP の衝突回避）
    'ewc': {'type', 'ewc.target_components', 'ewc.lam', 'ewc.state_path',
            'encoder.num_cp'},
    'inflora': {'type', 'inflora.design_path', 'lora.r', 'lora.alpha',
                'lora.exclude_components', 'lora.decompose_mha', 'lora.include'},
}
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


def base_path(d):
    if d in FULL_ORDER[:3]:
        return os.path.join(ROOT, 'experiments', 'exp_024', 'configs',
                            f'fullft_replayfree_{d}.py')
    return os.path.join(ROOT, 'experiments', 'exp_040', 'configs',
                        f'replayfree_{d}.py')


if __name__ == '__main__':
    do_build = '--build' in sys.argv
    do_step = '--step' in sys.argv
    init_default_scope('mmdet')

    # --- 1. build・設定・機械的 diff -----------------------------------------
    cfgs, bad = {}, []
    for m in ('ewc', 'inflora'):
        for d in FULL_ORDER:
            p = os.path.join(CFG_DIR, f'{m}_{d}.py')
            try:
                c = Config.fromfile(p)
                cfgs[(m, d)] = c
            except Exception as e:
                bad.append(f'{m}_{d}: {type(e).__name__}')
                continue
            tag = f'{m}_{d}'
            for label, got, want in (
                    ('max_epochs', c.train_cfg.get('max_epochs'), 20),
                    ('milestones', list(c.param_scheduler[0]['milestones']), [15]),
                    ('lr', c.optim_wrapper['optimizer']['lr'], 0.0001),
                    ('wd', c.optim_wrapper['optimizer']['weight_decay'], 0.0001),
                    ('clip', c.optim_wrapper['clip_grad']['max_norm'], 0.1),
                    ('batch', c.train_dataloader['batch_size'], 4),
                    ('sampler', c.train_dataloader['sampler']['type'],
                     'DefaultSampler'),
                    ('dataset', c.train_dataloader['dataset']['type'],
                     'ODVGDataset'),
                    ('seed', c.randomness['seed'], 0),
                    ('num_classes', c.model['bbox_head']['num_classes'], 256)):
                if got != want:
                    bad.append(f'{tag}: {label}={got}(期待 {want})')
            if d not in c.train_dataloader['dataset']['ann_file']:
                bad.append(f'{tag}: 現在ドメインのデータでない')
            if 'use_dn' in c.model:
                bad.append(f'{tag}: use_dn が上書きされている（dn は有効のまま）')
            if m == 'ewc' and c.model.get('encoder', {}).get('num_cp') != 0:
                bad.append(f'{tag}: encoder.num_cp が 0 でない（DDP×ペナルティ衝突）')
            if m == 'inflora' and c.model.get('encoder', {}).get('num_cp') != 6:
                bad.append(f'{tag}: inflora の num_cp が既定 6 でない')
            ref = Config.fromfile(base_path(d))
            if _plain(ref.train_dataloader) != _plain(c.train_dataloader):
                bad.append(f'{tag}: train_dataloader が継承元と不一致')
            fa, fb = _flat(ref.model), _flat(c.model)
            for k in sorted(set(fa) | set(fb)):
                if fa.get(k) == fb.get(k) or k in ALLOWED[m]:
                    continue
                bad.append(f'{tag}: model.{k} base={fa.get(k)!r} 045={fb.get(k)!r}')
    check(1, '12 本 build・設定・データ経路・機械的 diff',
          len(cfgs) == 12 and not bad,
          f'{len(cfgs)}/12 / 不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 2. --cfg-options の到達性 -------------------------------------------
    from mmengine.config import DictAction
    bad = []
    act = DictAction(option_strings=['--cfg-options'], dest='o')

    def parse(args):
        ns = type('NS', (), {})()
        act(None, ns, args)
        return ns.o

    c = copy.deepcopy(cfgs[('ewc', 'underwater')])
    c.merge_from_dict(parse(['model.ewc.lam=1000.0',
                             'model.ewc.state_path=/tmp/s.pth']))
    if c.model['ewc']['lam'] != 1000.0 or c.model['ewc']['state_path'] != '/tmp/s.pth':
        bad.append('model.ewc.* が届かない')
    c2 = copy.deepcopy(cfgs[('ewc', 'underwater')])
    c2.merge_from_dict(parse(['ewc.lam=7']))
    if c2.model['ewc']['lam'] == 7:
        bad.append('model. なしの誤記が届いてしまう')
    c3 = copy.deepcopy(cfgs[('inflora', 'underwater')])
    c3.merge_from_dict(parse(['model.inflora.design_path=/tmp/d.pth']))
    if c3.model['inflora']['design_path'] != '/tmp/d.pth':
        bad.append('model.inflora.design_path が届かない')
    check(2, '--cfg-options が model に届く（誤記は届かない）', not bad,
          f'{bad[:2]}' if bad else 'OK')

    # --- 3. model build（--build）-------------------------------------------
    if do_build:
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        bad, info = [], []
        for m in ('ewc', 'inflora'):
            c = cfgs[(m, 'underwater')]
            import_modules_from_strings(**c['custom_imports'])
            model = MODELS.build(copy.deepcopy(c.model))
            tr = [n for n, p in model.named_parameters() if p.requires_grad]
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if m == 'ewc':
                ok = len(tr) == sum(1 for _ in model.named_parameters())
            else:
                ok = all(n.endswith('lora_B') for n in tr) and len(tr) == 232
            if not ok:
                bad.append(f'{m}: 学習対象が想定外 ({len(tr)} params)')
            info.append(f'{m}: {len(tr)} params ({n_tr/1e6:.2f}M)')
            del model
            torch.cuda.empty_cache()
        check(3, 'EWC は全学習可・InfLoRA は dn 有効でも lora_B のみ 232 層',
              not bad, ' / '.join(info))

    # --- 4〜6. ODVG 実データの一連（--step）---------------------------------
    if do_step:
        from mmengine.dataset import pseudo_collate
        from mmengine.runner import Runner
        from mmengine.runner.checkpoint import load_checkpoint
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import DATASETS, MODELS
        tmpdir = tempfile.mkdtemp(prefix='exp045_check_')

        def one_step(model, cfg, train=True):
            loader = Runner.build_dataloader(copy.deepcopy(cfg.train_dataloader))
            batch = next(iter(loader))
            data = model.data_preprocessor(batch, True)
            losses = model.loss(data['inputs'], data['data_samples'])
            total = sum(v.sum() for v in losses.values()
                        if isinstance(v, torch.Tensor) and v.requires_grad)
            if train:
                total.backward()
            del loader
            return losses, total

        # 4. EWC: 状態なし 1 step → Fisher → 状態あり 1 step
        c = cfgs[('ewc', 'underwater')]
        import_modules_from_strings(**c['custom_imports'])
        cfg_dump = os.path.join(tmpdir, 'ewc_underwater.py')
        c.dump(cfg_dump)
        m0 = copy.deepcopy(c.model)
        model = MODELS.build(m0).cuda()
        load_checkpoint(model, THETA0, map_location='cpu')
        model.train()
        torch.cuda.reset_peak_memory_stats()
        losses, total = one_step(model, c)
        v1 = torch.cuda.max_memory_allocated() / 2**30
        ok_a = bool(torch.isfinite(total)) and 'loss_ewc' not in losses
        del model
        torch.cuda.empty_cache()

        state = os.path.join(tmpdir, 'state_t1.pth')
        r = subprocess.run(
            [sys.executable, 'projects/ewc_cl/estimate_fisher.py', cfg_dump,
             THETA0, '--out', state, '--max-samples', '4', '--seed', '0',
             '--device', 'cuda:0'], capture_output=True, text=True, cwd=ROOT)
        ok_f = r.returncode == 0 and os.path.exists(state)

        m1 = copy.deepcopy(c.model)
        m1['ewc'].update(lam=1000.0, state_path=state)
        model = MODELS.build(m1).cuda()
        load_checkpoint(model, THETA0, map_location='cpu')
        model.train()
        losses, total = one_step(model, c)
        pen = losses.get('loss_ewc')
        ok_b = pen is not None and bool(torch.isfinite(pen)) \
            and bool(torch.isfinite(total))
        check(4, 'EWC: ODVG 1 step（状態なし/あり）と Fisher 推定',
              ok_a and ok_f and ok_b,
              f'状態なし loss={float(total):.2f} / Fisher {"OK" if ok_f else "NG"}'
              f' / 状態あり loss_ewc={float(pen) if pen is not None else "なし"}'
              f' / VRAM={v1:.1f}GB'
              + ('' if ok_f else f' / {(r.stderr or r.stdout)[-200:]}'))
        del model
        torch.cuda.empty_cache()

        # 5. InfLoRA: 設計 → 1 step → メモリ更新（T=6）→ 再設計
        c = cfgs[('inflora', 'underwater')]
        import_modules_from_strings(**c['custom_imports'])
        cfg_dump = os.path.join(tmpdir, 'inflora_underwater.py')
        c.dump(cfg_dump)
        design1 = os.path.join(tmpdir, 'design_t1.pth')
        mem1 = os.path.join(tmpdir, 'memory_t1.pth')
        design2 = os.path.join(tmpdir, 'design_t2.pth')

        def run_cli(script, ckpt, *extra):
            return subprocess.run(
                [sys.executable, f'projects/lora_cl/{script}', cfg_dump, ckpt,
                 '--max-samples', '8', '--seed', '0', '--device', 'cuda:0',
                 *extra], capture_output=True, text=True, cwd=ROOT)

        r1 = run_cli('inflora_prepare.py', THETA0, '--out', design1)
        ok1 = r1.returncode == 0 and os.path.exists(design1)
        n_def = len(re.findall(r'ランク不足:', r1.stdout)) if ok1 else -1

        ok2 = ok3 = ok4 = False
        detail = ''
        if ok1:
            m2 = copy.deepcopy(c.model)
            m2['inflora'] = dict(design_path=design1)
            model = MODELS.build(m2).cuda()
            load_checkpoint(model, THETA0, map_location='cpu')
            model.train()
            torch.cuda.reset_peak_memory_stats()
            a_before = {n: mm.lora_A.detach().clone()
                        for n, mm in model.inflora_layers()}
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-4)
            losses, total = one_step(model, c)
            opt.step()
            a_moved = sum(1 for n, mm in model.inflora_layers()
                          if not torch.equal(a_before[n], mm.lora_A.detach()))
            b_moved = sum(1 for _, mm in model.inflora_layers()
                          if mm.lora_B.abs().max() > 0)
            ok2 = bool(torch.isfinite(total)) and a_moved == 0 and b_moved > 0
            v2 = torch.cuda.max_memory_allocated() / 2**30
            ck = os.path.join(tmpdir, 'inflora_step.pth')
            torch.save({'state_dict': model.state_dict()}, ck)
            del model, opt
            torch.cuda.empty_cache()

            r2 = run_cli('inflora_update_memory.py', ck, '--out', mem1,
                         '--task-index', '0', '--total', '6')
            ok3 = r2.returncode == 0 and os.path.exists(mem1)
            r3 = run_cli('inflora_prepare.py', THETA0, '--out', design2,
                         '--memory', mem1)
            ok4 = r3.returncode == 0 and os.path.exists(design2)
            detail = (f'ランク不足 {n_def} 層 / 1step loss={float(total):.2f} '
                      f'A変動 {a_moved} / B非零 {b_moved}/232 / VRAM={v2:.1f}GB '
                      f'/ T=6 メモリ更新 {"OK" if ok3 else "NG"} / 再設計 '
                      f'{"OK" if ok4 else "NG"}')
            if not (ok3 and ok4):
                bad_r = r2 if not ok3 else r3
                detail += f' / {(bad_r.stderr or bad_r.stdout)[-200:]}'
        else:
            detail = (r1.stderr or r1.stdout)[-300:]
        check(5, 'InfLoRA: 設計→1 step→メモリ更新(T=6)→再設計（ODVG 経路）',
              ok1 and ok2 and ok3 and ok4, detail)

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
