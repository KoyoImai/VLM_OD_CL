"""exp_042 実行前検証（design.md §4.2 の実験側。実装自体の検証は
projects/ewc_cl/check_ewc_setup.py と projects/lora_cl/check_inflora_setup.py）。

使い方:
    python experiments/exp_042/check_exp042_setup.py            # config のみ（GPU 不要）
    python experiments/exp_042/check_exp042_setup.py --build    # + model build（GPU 1 枚）
    python experiments/exp_042/check_exp042_setup.py --smoke    # + 実データ 1 step ×2 手法

検証項目
  1. 26 本の config が build でき、継承元が exp_038 条件C のタスク config であること
  2. 学習設定が共通枠と一致（3000 iter / milestones[1200] / lr 1e-4 / wd 1e-4 /
     clip 0.1 / batch 2 / seed 0 / num_classes 256 / dn 無効）
  3. 手法固有値（EWC: 対象 all・λ/state はドライバ渡し / InfLoRA: r=16・alpha=16）
  4. 機械的 diff: exp_038 条件C と比べ、model の差分が意図したキーだけ
     （type / lora / ewc / inflora）、train_dataloader が同一、optimizer の差分が
     lr / wd だけであること
  5. --cfg-options の到達性: model.ewc.lam / model.ewc.state_path /
     model.inflora.design_path が model に届くこと（model. を付けない誤記は
     トップレベルに行き model に届かないことも確認）
  6. （--build）EWC は全パラメータ学習可、InfLoRA は 232 層の lora_B のみ学習可
  7. （--smoke）実データ 1 step で loss が有限（生成 config の統合確認）
"""
import copy
import os
import sys

from mmengine.config import Config
from mmengine.registry import init_default_scope

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'exp_034'))
from odinw_official_tasks import task_order  # noqa: E402

CFG_DIR = os.path.join(HERE, 'configs')
ORDER = task_order(42)
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
            out[key] = v
    return out


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


if __name__ == '__main__':
    do_build = '--build' in sys.argv
    do_smoke = '--smoke' in sys.argv
    init_default_scope('mmdet')

    # --- 1. build と継承元 ---------------------------------------------------
    cfgs, bad = {}, []
    for m in ('ewc', 'inflora'):
        for task in ORDER:
            p = os.path.join(CFG_DIR, f'{m}_{task}.py')
            if not os.path.exists(p):
                bad.append(f'{m}_{task}: なし')
                continue
            try:
                cfgs[(m, task)] = Config.fromfile(p)
            except Exception as e:
                bad.append(f'{m}_{task}: {type(e).__name__}: {e}')
                continue
            if f'condC_{task}.py' not in open(p).read():
                bad.append(f'{m}_{task}: 継承元が条件C でない')
    check(1, '26 本の config が build でき継承元が exp_038 条件C',
          len(cfgs) == 26 and not bad,
          f'{len(cfgs)}/26' + (f' / 問題 {bad[:2]}' if bad else ''))

    # --- 2. 学習設定 ---------------------------------------------------------
    bad = []
    for (m, task), c in cfgs.items():
        tag = f'{m}_{task}'
        for label, got, want in (
                ('max_iters', c.train_cfg.get('max_iters'), 3000),
                ('loop', c.train_cfg.get('type'), 'IterBasedTrainLoop'),
                ('milestones', list(c.param_scheduler[0]['milestones']), [1200]),
                ('lr', c.optim_wrapper['optimizer']['lr'], 0.0001),
                ('wd', c.optim_wrapper['optimizer']['weight_decay'], 0.0001),
                ('clip', c.optim_wrapper['clip_grad']['max_norm'], 0.1),
                ('batch', c.train_dataloader['batch_size'], 2),
                ('seed', c.randomness['seed'], 0),
                ('num_classes', c.model['bbox_head']['num_classes'], 256),
                ('use_dn', c.model.get('use_dn'), False),
                ('head', c.model['bbox_head']['type'], 'NoDNGroundingDINOHead')):
            if got != want:
                bad.append(f'{tag}: {label}={got}(期待 {want})')
    check(2, '学習設定が共通枠（lr 1e-4 / wd 1e-4 ほか）と一致', not bad,
          f'26 本 × 11 項目 / 不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 3. 手法固有値 -------------------------------------------------------
    bad = []
    for (m, task), c in cfgs.items():
        tag = f'{m}_{task}'
        if m == 'ewc':
            if c.model['type'] != 'EWCGroundingDINO':
                bad.append(f'{tag}: type={c.model["type"]}')
            if 'lora' in c.model:
                bad.append(f'{tag}: lora が残っている')
            e = c.model.get('ewc', {})
            if e.get('target_components') != 'all' or e.get('state_path') is not None:
                bad.append(f'{tag}: ewc={dict(e)}')
            if 'projects.ewc_cl' not in c['custom_imports']['imports']:
                bad.append(f'{tag}: projects.ewc_cl が import されない')
        else:
            if c.model['type'] != 'GroundingDINOInfLoRA':
                bad.append(f'{tag}: type={c.model["type"]}')
            lo = c.model.get('lora', {})
            if lo.get('r') != 16 or lo.get('alpha') != 16:
                bad.append(f'{tag}: r={lo.get("r")} alpha={lo.get("alpha")}')
            if c.model.get('inflora', {}).get('design_path') is not None:
                bad.append(f'{tag}: design_path が None でない')
    check(3, '手法固有値が design.md §2.2 / §2.3 と一致', not bad,
          f'不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 4. 機械的 diff（exp_038 条件C と）-----------------------------------
    ALLOWED = {
        'ewc': {'type', 'ewc.target_components', 'ewc.lam', 'ewc.state_path'},
        'inflora': {'type', 'inflora.design_path', 'lora.alpha'},
    }
    bad = []
    for (m, task), c in cfgs.items():
        ref = Config.fromfile(os.path.join(
            ROOT, 'experiments', 'exp_038', 'configs',
            f'lora_odinw13_condC_{task}.py'))
        fa, fb = _flat(ref.model), _flat(c.model)
        for k in sorted(set(fa) | set(fb)):
            if fa.get(k) == fb.get(k):
                continue
            base = k if not k.startswith('lora.') or m == 'inflora' else 'lora'
            if m == 'ewc' and k.startswith('lora.'):
                continue          # lora 一式の削除は意図どおり（項目3で検査）
            if k in ALLOWED[m]:
                continue
            bad.append(f'{m}_{task}: model.{k} 038={fa.get(k)!r} 042={fb.get(k)!r}')
        if _plain(ref.train_dataloader) != _plain(c.train_dataloader):
            bad.append(f'{m}_{task}: train_dataloader が不一致')
        oa, ob = _flat(ref.optim_wrapper), _flat(c.optim_wrapper)
        for k in sorted(set(oa) | set(ob)):
            if oa.get(k) != ob.get(k) and k not in (
                    'optimizer.lr', 'optimizer.weight_decay'):
                bad.append(f'{m}_{task}: optim.{k}')
    check(4, 'exp_038 条件C との差分が意図したキーのみ', not bad,
          f'26 本 / 想定外の差分 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 5. --cfg-options の到達性 -------------------------------------------
    from mmengine.config import DictAction
    bad = []
    act = DictAction(option_strings=['--cfg-options'], dest='o')

    def parse(args):
        # DictAction は namespace の既存辞書へマージするため、毎回新しい namespace を使う
        ns = type('NS', (), {})()
        act(None, ns, args)
        return ns.o

    c = copy.deepcopy(cfgs[('ewc', ORDER[0])])
    c.merge_from_dict(parse(['model.ewc.lam=123.0', 'model.ewc.state_path=/tmp/x.pth']))
    if c.model['ewc']['lam'] != 123.0 or c.model['ewc']['state_path'] != '/tmp/x.pth':
        bad.append('model.ewc.* が届かない')
    c2 = copy.deepcopy(cfgs[('ewc', ORDER[0])])
    c2.merge_from_dict(parse(['ewc.lam=999']))
    if c2.model['ewc']['lam'] == 999:
        bad.append('model. なしの誤記が model に届いてしまう')
    c3 = copy.deepcopy(cfgs[('inflora', ORDER[0])])
    c3.merge_from_dict(parse(['model.inflora.design_path=/tmp/d.pth']))
    if c3.model['inflora']['design_path'] != '/tmp/d.pth':
        bad.append('model.inflora.design_path が届かない')
    check(5, '--cfg-options が model に届く（誤記は届かない）', not bad,
          f'{bad[:2]}' if bad else 'lam / state_path / design_path とも OK')

    # --- 6. model build ------------------------------------------------------
    if do_build:
        import torch
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        bad, info = [], []
        for m in ('ewc', 'inflora'):
            c = cfgs[(m, ORDER[0])]
            import_modules_from_strings(**c['custom_imports'])
            model = MODELS.build(copy.deepcopy(c.model))
            tr = [n for n, p in model.named_parameters() if p.requires_grad]
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if m == 'ewc':
                ok = len(tr) == sum(1 for _ in model.named_parameters())
            else:
                ok = (all(n.endswith('lora_B') for n in tr) and len(tr) == 232)
            if not ok:
                bad.append(f'{m}: 学習対象が想定外 ({len(tr)} params)')
            info.append(f'{m}: {len(tr)} params ({n_tr/1e6:.2f}M)')
            del model
            torch.cuda.empty_cache()
        check(6, 'EWC は全学習可・InfLoRA は lora_B のみ 232 層', not bad,
              ' / '.join(info))

    # --- 7. 実データ 1 step --------------------------------------------------
    if do_smoke:
        import torch
        from mmengine.dataset import pseudo_collate
        from mmengine.runner.checkpoint import load_checkpoint
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import DATASETS, MODELS
        bad, info = [], []
        for m in ('ewc', 'inflora'):
            c = cfgs[(m, ORDER[0])]
            import_modules_from_strings(**c['custom_imports'])
            model = MODELS.build(copy.deepcopy(c.model))
            load_checkpoint(model, c.load_from, map_location='cpu')
            model = model.cuda().train()
            ds = DATASETS.build(c.train_dataloader['dataset'])
            batch = pseudo_collate([ds[0], ds[1]])
            data = model.data_preprocessor(batch, True)
            losses = model.loss(data['inputs'], data['data_samples'])
            total = sum(v.sum() for v in losses.values()
                        if isinstance(v, torch.Tensor) and v.requires_grad)
            total.backward()
            if not torch.isfinite(total):
                bad.append(f'{m}: loss が有限でない')
            info.append(f'{m}: loss={float(total):.2f}')
            del model
            torch.cuda.empty_cache()
        check(7, '実データ 1 step で loss が有限（生成 config の統合確認）',
              not bad, ' / '.join(info))

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
