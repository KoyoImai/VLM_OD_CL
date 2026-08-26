"""exp_053 実行前検証（design.md §4）。

使い方:
    python experiments/exp_053/check_exp052_setup.py          # config のみ（GPU 不要）
    python experiments/exp_053/check_exp052_setup.py --step   # + 実データ 1 step（GPU 1 枚）

検証項目
  1. 15 本の config が build でき、model と train_dataloader が exp_040 kdE と完全一致
     （＝本実験の差分が paramwise と ckpt 保存方針だけであること）。schedule・seed も確認。
  2. 各条件の学習対象パラメータの実測。mmengine の optimizer constructor を実際に
     組み、param group の lr>0 の集合が意図したモジュール境界（design.md §2.2）と
     厳密一致すること。custom_keys の部分一致で意図しないパラメータが lr>0 になって
     いないか（未カバー→既定 lr_mult=1.0 の事故）を全パラメータについて確認する。
  3. --cfg-options の到達性（model.teacher_ckpt。誤記の負テスト込み）。
  4. （--step）実データ 1 step（underwater t=1、5 条件）: loss・loss_kd が有限、
     AdamW 1 step 後に凍結パラメータがビット単位で不変・学習対象のみ変化。
"""
import copy
import os
import sys

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CFG_DIR = os.path.join(HERE, 'configs')

DOMAINS = ['aerial', 'microscopic', 'documents']
THETA0 = os.path.join(
    ROOT, 'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det'
    '_20231204_095047-b448804b.pth')

# 条件 -> 学習対象のパラメータ名プレフィックス（design.md §2.2 = exp_015 の区分）
EXPECTED = {
    'swinNeck': ('backbone.', 'neck.'),
    'bertTfm': ('language_model.', 'text_feat_map.'),
    'enhancer': ('encoder.',),
    'qsel': ('memory_trans_fc.', 'memory_trans_norm.',
             'level_embed', 'query_embedding.'),
    'decoder': ('decoder.',),
}
CONDS = list(EXPECTED)
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
    do_step = '--step' in sys.argv
    init_default_scope('mmdet')

    # --- 1. build・exp_035 kdE との機械的 diff --------------------------------
    cfgs, bad = {}, []
    for cond in CONDS:
        for d in DOMAINS:
            p = os.path.join(CFG_DIR, f'{cond}_{d}.py')
            try:
                c = Config.fromfile(p)
                cfgs[(cond, d)] = c
            except Exception as e:
                bad.append(f'{cond}_{d}: {type(e).__name__}')
                continue
            ref = Config.fromfile(os.path.join(
                ROOT, 'experiments', 'exp_040', 'configs',
                f'kdE_{d}.py'))
            if _plain(ref.model) != _plain(c.model):
                bad.append(f'{cond}_{d}: model が exp_040 kdE と不一致')
            if _plain(ref.train_dataloader) != _plain(c.train_dataloader):
                bad.append(f'{cond}_{d}: train_dataloader が exp_040 kdE と不一致')
            for label, got, want in (
                    ('max_epochs', c.train_cfg.get('max_epochs'), 20),
                    ('lr', c.optim_wrapper['optimizer']['lr'], 0.0001),
                    ('batch', c.train_dataloader['batch_size'], 6),
                    ('ratio', list(c.train_dataloader['sampler']['source_ratio']), [4, 1, 1]),
                    ('seed', c.randomness['seed'], 0),
                    ('λ', c.model['kd']['loss_weight'], 10.0),
                    ('buf', c.model['kd']['num_buffer_per_batch'], 2),
                    ('ckpt_interval',
                     c.default_hooks['checkpoint']['interval'], 20),
                    ('save_optimizer',
                     c.default_hooks['checkpoint']['save_optimizer'], False)):
                if got != want:
                    bad.append(f'{cond}_{d}: {label}={got}(期待 {want})')
    check(1, '15 本 build・model/dataloader が exp_040 kdE と一致・schedule/ckpt 方針',
          len(cfgs) == 15 and not bad,
          f'{len(cfgs)}/15 / 不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 2. 学習対象パラメータの実測（optimizer を実際に組む）-----------------
    from mmengine.optim import build_optim_wrapper
    from mmdet.registry import MODELS
    from mmengine.utils import import_modules_from_strings

    c0 = cfgs[(CONDS[0], 'aerial')]
    import_modules_from_strings(**c0['custom_imports'])
    m = copy.deepcopy(c0.model)
    m['type'] = 'GroundingDINO'  # 教師 deepcopy を避け素の構造で数える
    m.pop('kd'); m.pop('teacher_ckpt', None)
    model = MODELS.build(m)
    name_of = {id(p): n for n, p in model.named_parameters()}

    bad, info = [], []
    for cond in CONDS:
        c = cfgs[(cond, 'aerial')]
        ow = build_optim_wrapper(model, copy.deepcopy(c.optim_wrapper))
        got_train, got_frozen, uncovered = set(), set(), []
        n_elem = 0
        for g in ow.optimizer.param_groups:
            for p in g['params']:
                n = name_of[id(p)]
                if g['lr'] > 0:
                    got_train.add(n)
                    n_elem += p.numel()
                else:
                    got_frozen.add(n)
        exp_prefix = EXPECTED[cond]
        want_train = {n for n in name_of.values()
                      if n.startswith(exp_prefix)}
        extra = got_train - want_train      # 意図せず lr>0（未カバー事故を含む）
        missing = want_train - got_train    # 意図したのに lr=0
        if extra or missing:
            bad.append(f'{cond}: 余分 {sorted(extra)[:3]} 欠落 {sorted(missing)[:3]}')
        if len(got_train) + len(got_frozen) != len(name_of):
            bad.append(f'{cond}: optimizer 群のパラメータ数不一致')
        info.append(f'{cond}: {len(got_train)}個 {n_elem/1e6:.2f}M')
    check(2, '各条件の lr>0 集合が意図したモジュール境界と厳密一致', not bad,
          ' / '.join(info) + (f' / {bad[:2]}' if bad else ''))
    del model

    # --- 3. --cfg-options の到達性 -------------------------------------------
    from mmengine.config import DictAction
    bad = []
    act = DictAction(option_strings=['--cfg-options'], dest='o')

    def parse(args):
        ns = type('NS', (), {})()
        act(None, ns, args)
        return ns.o

    c = copy.deepcopy(cfgs[('qsel', 'aerial')])
    c.merge_from_dict(parse(['model.teacher_ckpt=/tmp/t.pth', 'load_from=/tmp/t.pth']))
    if c.model['teacher_ckpt'] != '/tmp/t.pth' or c.load_from != '/tmp/t.pth':
        bad.append('model.teacher_ckpt / load_from が届かない')
    c2 = copy.deepcopy(cfgs[('qsel', 'aerial')])
    c2.merge_from_dict(parse(['teacher_ckpt=/tmp/x.pth']))
    if c2.model.get('teacher_ckpt') is not None:
        bad.append('model. なしの誤記が届いてしまう')
    check(3, 'model.teacher_ckpt / load_from が届く（誤記は届かない）', not bad,
          f'{bad[:2]}' if bad else 'OK')

    # --- 4. 実データ 1 step（--step）-----------------------------------------
    if do_step:
        from mmengine.runner import Runner
        from mmengine.runner.checkpoint import load_checkpoint
        bad, info = [], []
        loader = Runner.build_dataloader(
            copy.deepcopy(cfgs[('qsel', 'aerial')].train_dataloader))
        batch = next(iter(loader))
        for cond in CONDS:
            c = cfgs[(cond, 'aerial')]
            m = copy.deepcopy(c.model)
            m['teacher_ckpt'] = THETA0
            model = MODELS.build(m).cuda()
            model.init_weights()               # ★ init_weights が先（exp_048 の教訓）
            load_checkpoint(model, THETA0, map_location='cpu')
            ow = build_optim_wrapper(model, copy.deepcopy(c.optim_wrapper))
            before = {n: p.detach().clone()
                      for n, p in model.named_parameters()}
            model.train()
            data = model.data_preprocessor(copy.deepcopy(batch), True)
            losses = model.loss(data['inputs'], data['data_samples'])
            total = sum(v.sum() for v in losses.values()
                        if isinstance(v, torch.Tensor) and v.requires_grad)
            total.backward()
            ow.optimizer.step()
            exp_prefix = EXPECTED[cond]
            changed_bad = [n for n, p in model.named_parameters()
                           if not n.startswith(exp_prefix)
                           and not torch.equal(before[n], p.detach())]
            changed_ok = sum(1 for n, p in model.named_parameters()
                             if n.startswith(exp_prefix)
                             and not torch.equal(before[n], p.detach()))
            kd = float(losses['loss_kd'])
            if not (torch.isfinite(total) and changed_ok > 0
                    and not changed_bad):
                bad.append(f'{cond}: loss={float(total):.2f} 変化{changed_ok} '
                           f'凍結側変化 {changed_bad[:3]}')
            info.append(f'{cond}: loss={float(total):.1f} kd={kd:.2f} '
                        f'更新={changed_ok}個')
            del model, ow, before
            torch.cuda.empty_cache()
        check(4, '実データ 1 step: 有限・凍結不変・学習対象のみ更新', not bad,
              ' / '.join(info) + (f' / {bad[:1]}' if bad else ''))

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
