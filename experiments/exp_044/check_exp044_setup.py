"""exp_044 実行前検証（design.md §4.2）。本環境で実施し、クラスタへは push して持ち込む。

使い方:
    python experiments/exp_044/check_exp044_setup.py          # config のみ（GPU 不要）
    python experiments/exp_044/check_exp044_setup.py --build  # + projector の構造確認
    python experiments/exp_044/check_exp044_setup.py --step   # + 等価性・全データ化・1 step（GPU）

検証項目
  1. 3 本の config が build でき、学習設定・バッチ構成が既存系列と一致
     （20 epoch / lr 1e-4 / wd 1e-4 / batch 6 / t=1 [2,1] / t≥2 [4,1,1]）。
     機械的 diff: model が exp_040 kdE と「type と蒸留対象の全データ化」だけの差、
     train_dataloader が継承元（exp_023 / exp_040 のリプレイ）と同一
  2. --cfg-options の到達性（model.teacher_ckpt。誤記の負テスト込み）
  3. （--build）projector が 4 点・各 131,584 パラメータ・学習可能であること
  4. （--step）**等価性**: projector を恒等関数に固定すると、蒸留損失が親クラス
     （KDGroundingDINO・全データ化）と一致すること（決定的モードで比較）
  5. （--step）**全データ化**: バッファ以外（現在ドメイン）のサンプルを変えると
     loss_kd が変わること
  6. （--step）実データ 1 step（t=1 / t≥2）で loss・loss_kd が有限、projector に勾配が
     流れ optimizer step で動くこと、ckpt に kd_proj_* キーが含まれること。VRAM 実測
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

FULL_ORDER = ['underwater', 'electromagnetic', 'videogames']  # 前半3のみ（2026-08-16 変更）
THETA0 = os.path.join(
    ROOT, 'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det'
    '_20231204_095047-b448804b.pth')
PROJ_KEYS = ('kd_proj_img', 'kd_proj_txt', 'kd_proj_fus_img', 'kd_proj_fus_txt')

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


def det_mode(model):
    """dropout 全無効＋loss 経路維持（ビット決定的。exp_043 検証と同じ方式）。"""
    model.eval()
    model.training = True


def build_model(cfg, model_override=None, teacher=THETA0):
    from mmengine.runner.checkpoint import load_checkpoint
    from mmdet.registry import MODELS
    m = copy.deepcopy(cfg.model)
    if model_override:
        m.update(model_override)
    m['teacher_ckpt'] = teacher
    model = MODELS.build(m).cuda()
    model.init_weights()                       # 教師構築＋捕捉 hook（Runner と同順）
    load_checkpoint(model, THETA0, map_location='cpu')
    return model


if __name__ == '__main__':
    do_build = '--build' in sys.argv
    do_step = '--step' in sys.argv
    init_default_scope('mmdet')

    # --- 1. build・設定・機械的 diff -----------------------------------------
    cfgs, bad = {}, []
    ref40 = Config.fromfile(os.path.join(
        ROOT, 'experiments', 'exp_040', 'configs', 'kdE_aerial.py'))
    for i, d in enumerate(FULL_ORDER):
        t = i + 1
        p = os.path.join(CFG_DIR, f'kdEp_{d}.py')
        try:
            c = Config.fromfile(p)
            cfgs[d] = c
        except Exception as e:
            bad.append(f'{d}: {type(e).__name__}')
            continue
        for label, got, want in (
                ('max_epochs', c.train_cfg.get('max_epochs'), 20),
                ('milestones', list(c.param_scheduler[0]['milestones']), [15]),
                ('lr', c.optim_wrapper['optimizer']['lr'], 0.0001),
                ('wd', c.optim_wrapper['optimizer']['weight_decay'], 0.0001),
                ('batch', c.train_dataloader['batch_size'], 6),
                ('seed', c.randomness['seed'], 0),
                ('num_classes', c.model['bbox_head']['num_classes'], 256),
                ('type', c.model['type'], 'KDProjGroundingDINO'),
                ('λ', c.model['kd']['loss_weight'], 10.0)):
            if got != want:
                bad.append(f'{d}: {label}={got}(期待 {want})')
        ratio = list(c.train_dataloader['sampler']['source_ratio'])
        want_ratio = [2, 1] if t == 1 else [4, 1, 1]
        if ratio != want_ratio:
            bad.append(f'{d}: ratio={ratio}(期待 {want_ratio})')
        # model の機械的 diff（exp_040 kdE と）: 差分は type と
        # kd.num_buffer_per_batch（全データ化のため config から外した）だけ
        fa, fb = _flat(ref40.model), _flat(c.model)
        for k in sorted(set(fa) | set(fb)):
            if fa.get(k) == fb.get(k) or k in ('type', 'kd.num_buffer_per_batch'):
                continue
            bad.append(f'{d}: model.{k} 040={fa.get(k)!r} 044={fb.get(k)!r}')
        # データ経路は継承元と同一
        refd = Config.fromfile(os.path.join(
            ROOT, 'experiments', 'exp_023', 'configs', f'fullft_replay_{d}.py'))
        if _plain(refd.train_dataloader) != _plain(c.train_dataloader):
            bad.append(f'{d}: train_dataloader が継承元と不一致')
    check(1, '3 本 build・設定・バッチ構成・機械的 diff', len(cfgs) == 3 and not bad,
          f'{len(cfgs)}/3 / 不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 2. --cfg-options の到達性 -------------------------------------------
    from mmengine.config import DictAction
    bad = []
    act = DictAction(option_strings=['--cfg-options'], dest='o')

    def parse(args):
        ns = type('NS', (), {})()
        act(None, ns, args)
        return ns.o

    c = copy.deepcopy(cfgs['underwater'])
    c.merge_from_dict(parse(['model.teacher_ckpt=/tmp/t.pth']))
    if c.model['teacher_ckpt'] != '/tmp/t.pth':
        bad.append('model.teacher_ckpt が届かない')
    c2 = copy.deepcopy(cfgs['underwater'])
    c2.merge_from_dict(parse(['teacher_ckpt=/tmp/x.pth']))
    if c2.model.get('teacher_ckpt') is not None:
        bad.append('model. なしの誤記が届いてしまう')
    check(2, 'model.teacher_ckpt が届く（誤記は届かない）', not bad,
          f'{bad[:2]}' if bad else 'OK')

    # --- 3. projector の構造（--build）--------------------------------------
    if do_build:
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        c = cfgs['underwater']
        import_modules_from_strings(**c['custom_imports'])
        m = copy.deepcopy(c.model)
        m['teacher_ckpt'] = THETA0
        model = MODELS.build(m)
        bad, info = [], []
        for k in PROJ_KEYS:
            mod = getattr(model, k, None)
            if mod is None:
                bad.append(f'{k} が無い')
                continue
            n = sum(p.numel() for p in mod.parameters())
            tr = all(p.requires_grad for p in mod.parameters())
            if n != 131584 or not tr:
                bad.append(f'{k}: params={n} trainable={tr}')
            info.append(f'{k}={n}')
        if model.num_buffer_per_batch < 10**6:
            bad.append(f'num_buffer_per_batch={model.num_buffer_per_batch}'
                       '（全データ化されていない）')
        check(3, 'projector 4 点・各 131,584 params・学習可・全データ化', not bad,
              ' / '.join(info[:2]) + ' ...')
        del model
        torch.cuda.empty_cache()

    # --- 4〜6. 等価性・全データ化・1 step（--step）--------------------------
    if do_step:
        import torch.nn as nn
        from mmengine.dataset import pseudo_collate
        from mmengine.runner import Runner
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import DATASETS

        c = cfgs['electromagnetic']          # t=2（3 ソース構成）
        import_modules_from_strings(**c['custom_imports'])
        ds = DATASETS.build(c.train_dataloader['dataset'])
        loader = Runner.build_dataloader(copy.deepcopy(c.train_dataloader))
        batch = next(iter(loader))

        # 4. 等価性: projector=恒等 → 親クラス（全データ化）と loss_kd 一致。
        # 学生=教師=θ0 だと蒸留損失が厳密に 0 になり「0 == 0」の自明な一致に
        # なってしまうため、学生の neck に同一 seed の微小摂動を入れて
        # KD が非零の状態で比較する。
        def perturb_neck(model):
            g = torch.Generator(device='cpu').manual_seed(0)
            with torch.no_grad():
                for p in model.neck.parameters():
                    p.add_(torch.randn(p.shape, generator=g).to(p.device) * 1e-3)

        proj = build_model(c)
        perturb_neck(proj)
        det_mode(proj)
        for k in PROJ_KEYS:
            setattr(proj, k, nn.Identity())
        with torch.no_grad():
            data = proj.data_preprocessor(copy.deepcopy(batch), True)
            l_proj = proj.loss(data['inputs'], data['data_samples'])
        kd_a = float(l_proj['loss_kd'])
        del proj
        torch.cuda.empty_cache()

        parent = build_model(
            c, model_override=dict(
                type='KDGroundingDINO',
                kd=dict(targets=['img', 'txt', 'fus'],
                        loss=dict(type='FeatureDistillLoss', form='l2'),
                        loss_weight=10.0, num_buffer_per_batch=10**9)))
        perturb_neck(parent)
        det_mode(parent)
        with torch.no_grad():
            data = parent.data_preprocessor(copy.deepcopy(batch), True)
            l_par = parent.loss(data['inputs'], data['data_samples'])
        kd_b = float(l_par['loss_kd'])
        rel = abs(kd_a - kd_b) / max(abs(kd_b), 1e-12)
        check(4, '恒等 projector で親クラス（直接・全データ）と loss_kd 一致（KD 非零）',
              kd_b > 0 and rel < 1e-4,
              f'proj={kd_a:.6f} 親={kd_b:.6f} 相対差 {rel:.2e}')
        del parent
        torch.cuda.empty_cache()

        # 5. 全データ化: 現在ドメイン（バッチ先頭）のサンプルを変えると loss_kd が変わる
        proj = build_model(c)
        det_mode(proj)
        with torch.no_grad():
            data = proj.data_preprocessor(copy.deepcopy(batch), True)
            kd_0 = float(proj.loss(data['inputs'], data['data_samples'])['loss_kd'])
            data2 = proj.data_preprocessor(copy.deepcopy(batch), True)
            data2['inputs'][0].add_(torch.randn_like(data2['inputs'][0]) * 10)
            kd_1 = float(proj.loss(data2['inputs'], data2['data_samples'])['loss_kd'])
        check(5, '現在ドメインのサンプル変更で loss_kd が変化（蒸留の全データ化）',
              abs(kd_1 - kd_0) > 1e-6, f'{kd_0:.6f} -> {kd_1:.6f}')

        # 6. 実データ 1 step（train モード）: 有限・projector 勾配・ckpt キー
        bad, info = [], []
        for d in ('underwater', 'electromagnetic'):
            cc = cfgs[d]
            model = build_model(cc)
            model.train()
            torch.cuda.reset_peak_memory_stats()
            ld = Runner.build_dataloader(copy.deepcopy(cc.train_dataloader))
            b = next(iter(ld))
            data = model.data_preprocessor(b, True)
            losses = model.loss(data['inputs'], data['data_samples'])
            total = sum(v.sum() for v in losses.values()
                        if isinstance(v, torch.Tensor) and v.requires_grad)
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-4)
            before = {k: getattr(model, k)[0].weight.detach().clone()
                      for k in PROJ_KEYS}
            total.backward()
            gnorm = sum(float(p.grad.abs().sum()) for k in PROJ_KEYS
                        for p in getattr(model, k).parameters()
                        if p.grad is not None)
            opt.step()
            moved = all(not torch.equal(before[k],
                                        getattr(model, k)[0].weight.detach())
                        for k in PROJ_KEYS)
            sd_keys = sum(1 for k in model.state_dict() if k.startswith('kd_proj'))
            kd = losses['loss_kd']
            if not (torch.isfinite(total) and torch.isfinite(kd)
                    and gnorm > 0 and moved and sd_keys == 16):
                bad.append(f'{d}: total={float(total):.2f} kd={float(kd):.2f} '
                           f'grad={gnorm:.2e} moved={moved} sd_keys={sd_keys}')
            info.append(f'{d}: loss={float(total):.2f} kd={float(kd):.4f} '
                        f'VRAM={torch.cuda.max_memory_allocated()/2**30:.1f}GB')
            del model, ld, opt
            torch.cuda.empty_cache()
        check(6, '1 step 有限・projector 勾配/更新・state_dict に kd_proj（16 キー）',
              not bad, ' / '.join(info) + (f' / {bad[:1]}' if bad else ''))

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
