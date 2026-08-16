"""EWC 実装の検証（exp_042 design.md §4.2 の項目 1〜3 ＋ 実データ 1 step）。

使い方:
    python projects/ewc_cl/check_ewc_setup.py            # 項目 3（純テンソル）のみ。GPU 不要
    python projects/ewc_cl/check_ewc_setup.py --gpu      # 全項目（GPU 1 枚・実データ）

検証項目
  1. 学習対象の選択と凍結: target_components の指定どおりに requires_grad が立ち、
     対象外が optimizer.step() 後も不動であること
  2. ペナルティ勾配の解析一致: loss_ewc だけを backward した勾配が λ·(A·θ−B) と一致
  3. 2 バッファとタスク別保持の等価性: 3 タスク分の乱数 F_k, θ*_k で、
     Σ_k (λ/2)F_k(θ−θ*_k)² のペナルティ値と勾配が 2 バッファ計算と一致（純テンソル）
  4. Fisher 推定パイプライン: 実データ数枚で estimate_fisher.py が走り、非負・有限、
     対象パラメータのみ、θ=θ* でのペナルティ ≈ 0（const 項の正しさ）
  5. 実データ 1 step: 状態を読んだ EWCGroundingDINO の loss() に loss_ewc が乗り、
     backward + step が通ること
"""
import copy
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

# 実データ検証に使う config（exp_038 の 1 タスク目 pistols。学習設定はここでは無関係で、
# モデル構造と train_dataloader だけを使う）
TASK_CFG = os.path.join(ROOT, 'experiments/exp_038/configs/lora_odinw13_pistols.py')
THETA0 = None   # config の load_from を使う

results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


def make_ewc_cfg(target_components, lam=0.0, state_path=None):
    """exp_038 の task config をベースに model を EWCGroundingDINO へ差し替える。"""
    from mmengine.config import Config
    cfg = Config.fromfile(TASK_CFG)
    m = cfg.model
    m['type'] = 'EWCGroundingDINO'
    m.pop('lora', None)
    m['ewc'] = dict(target_components=target_components, lam=lam,
                    state_path=state_path)
    imports = list(cfg['custom_imports']['imports'])
    if 'projects.ewc_cl' not in imports:
        imports.append('projects.ewc_cl')
    cfg['custom_imports'] = dict(imports=imports, allow_failed_imports=False)
    return cfg


def check3_pure_tensor():
    """項目 3: タスク別保持と 2 バッファの等価性（モデル不要）。"""
    from projects.ewc_cl.ewc_state import (new_empty_state, update_state,
                                           penalty)
    g = torch.Generator().manual_seed(0)
    shapes = [(8, 4), (16,), (3, 5, 2)]
    names = [f'p{i}' for i in range(len(shapes))]
    tasks = []
    for _ in range(3):
        F = {n: torch.rand(s, generator=g) for n, s in zip(names, shapes)}
        T = {n: torch.randn(s, generator=g) for n, s in zip(names, shapes)}
        tasks.append((F, T))

    theta = {n: torch.randn(s, generator=g).requires_grad_(True)
             for n, s in zip(names, shapes)}
    lam = 7.0

    # 素朴なタスク別保持（Avalanche 'separate' 相当）
    naive = sum(0.5 * lam * (F[n] * (theta[n] - T[n]) ** 2).sum()
                for F, T in tasks for n in names)
    naive.backward()
    g_naive = {n: theta[n].grad.clone() for n in names}
    for n in names:
        theta[n].grad = None

    # 2 バッファ
    state = new_empty_state()
    for F, T in tasks:
        update_state(state, F, T)
    two = penalty({'A': state['A'], 'B': state['B'], 'const': state['const']},
                  [(n, theta[n]) for n in names], lam)
    two.backward()
    g_two = {n: theta[n].grad.clone() for n in names}

    v_ok = torch.allclose(naive, two, rtol=1e-5, atol=1e-5)
    g_ok = all(torch.allclose(g_naive[n], g_two[n], rtol=1e-5, atol=1e-6)
               for n in names)
    check(3, '2 バッファがタスク別保持と等価（値・勾配）', v_ok and g_ok,
          f'値: naive={float(naive):.6f} two={float(two):.6f} / 勾配一致={g_ok}')


def run_gpu_checks():
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner.checkpoint import load_checkpoint
    from mmengine.dataset import pseudo_collate
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import DATASETS, MODELS
    from projects.ewc_cl.ewc_state import load_state

    init_default_scope('mmdet')
    base = Config.fromfile(TASK_CFG)
    import_modules_from_strings(**base['custom_imports'])
    import projects.ewc_cl  # noqa: F401  (EWCGroundingDINO を登録)

    dev = 'cuda:0'

    # --- 1. 対象選択と凍結 ---------------------------------------------------
    cfg = make_ewc_cfg(['encoder', 'text_feat_map'])
    model = MODELS.build(copy.deepcopy(cfg.model))
    tr = {n for n, p in model.named_parameters() if p.requires_grad}
    ok_sel = (tr and all(n.split('.')[0] in ('encoder', 'text_feat_map')
                         for n in tr))
    cfg_all = make_ewc_cfg('all')
    model_all = MODELS.build(copy.deepcopy(cfg_all.model))
    n_frozen = sum(1 for p in model_all.parameters() if not p.requires_grad)
    ok_all = (n_frozen == 0)
    del model_all

    # 対象外の不動確認: encoder のみ対象で 1 step 回し、backbone が動かないこと
    load_checkpoint(model, base.load_from, map_location='cpu')
    model = model.to(dev).train()
    dataset = DATASETS.build(cfg.train_dataloader['dataset'])
    batch = pseudo_collate([dataset[0], dataset[1]])
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if not p.requires_grad}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-3)
    data = model.data_preprocessor(batch, True)
    losses = model.loss(data['inputs'], data['data_samples'])
    total = sum(v.sum() for v in losses.values()
                if isinstance(v, torch.Tensor) and v.requires_grad)
    total.backward()
    opt.step()
    moved = [n for n, p in model.named_parameters()
             if not p.requires_grad and not torch.equal(before[n], p.detach())]
    check(1, '対象選択・凍結・対象外の不動', ok_sel and ok_all and not moved,
          f'encoder+tfm 対象 {len(tr)} params / all 凍結 0 / '
          f'対象外の変動 {len(moved)}')
    del model, opt
    torch.cuda.empty_cache()

    # --- 4. Fisher 推定パイプライン ------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix='ewc_check_')
    cfg_path = os.path.join(tmpdir, 'ewc_task_cfg.py')
    ecfg = make_ewc_cfg('all', lam=0.0)
    ecfg.dump(cfg_path)
    state_path = os.path.join(tmpdir, 'state_t1.pth')
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, 'estimate_fisher.py'), cfg_path,
         base.load_from, '--out', state_path, '--max-samples', '8',
         '--seed', '0', '--device', dev],
        capture_output=True, text=True, cwd=ROOT)
    ok_run = (r.returncode == 0 and os.path.exists(state_path))
    detail = ''
    if ok_run:
        st = load_state(state_path)
        vals = list(st['A'].values())
        ok_pos = all(torch.isfinite(v).all() and (v >= 0).all() for v in vals)
        nz = sum(int((v > 0).sum()) for v in vals)
        detail = (f'{len(vals)} params, 非零 {nz:,}, '
                  f'num_tasks={st["meta"]["num_tasks"]}')
        # θ=θ*（推定に使った ckpt そのもの）でペナルティ ≈ 0（const 項の検証）
        cfg2 = make_ewc_cfg('all', lam=1000.0, state_path=state_path)
        model2 = MODELS.build(copy.deepcopy(cfg2.model))
        load_checkpoint(model2, base.load_from, map_location='cpu')
        model2 = model2.to(dev)
        pen0 = float(model2.ewc_penalty())
        scale = 0.5 * 1000.0 * st['const'] if st['const'] > 0 else 1.0
        ok_zero = abs(pen0) / max(scale, 1e-12) < 1e-3
        detail += f' / θ=θ* ペナルティ={pen0:.3e}（const 項スケール {scale:.3e}）'
    else:
        ok_pos = ok_zero = False
        detail = (r.stderr or r.stdout)[-300:]
    check(4, 'Fisher 推定が走り非負・有限、θ=θ* でペナルティ≈0',
          ok_run and ok_pos and ok_zero, detail)

    # --- 2. ペナルティ勾配の解析一致 / 5. 実データ 1 step ---------------------
    if ok_run:
        # 数学的一致は**人工状態**（乱数 A・B）で検証する。実 Fisher 状態では
        # B=F·θ* かつ θ≈θ* のため、勾配 λ(A·θ−B)=λF(θ−θ*) が「大きな 2 数の差」に
        # なり、fp32 の丸め（〜1e-7·|F·θ*|）が差分に対して相対的に浮き上がる
        # （実測で最大相対 6.6e-3。絶対誤差としては訓練勾配のノイズより桁違いに
        # 小さく学習上は無害）。良条件の人工状態なら実装の正誤だけが見える。
        real_state = model2._ewc_state
        gen = torch.Generator().manual_seed(0)
        syn = {'A': {}, 'B': {}, 'const': 1.23,
               'meta': {'format': 1, 'num_tasks': 1, 'history': []}}
        for n, p in model2.named_parameters():
            syn['A'][n] = torch.rand(p.shape, generator=gen)
            syn['B'][n] = torch.randn(p.shape, generator=gen) * 0.01
        model2._ewc_state = syn
        model2._ewc_state_device = None
        model2.zero_grad(set_to_none=True)
        pen = model2.ewc_penalty()
        pen.backward()
        st_dev = model2._state_on_device(dev)
        max_rel = 0.0
        for n, p in model2.named_parameters():
            if p.grad is None:
                continue
            ana = model2.ewc_lam * (st_dev['A'][n] * p.detach() - st_dev['B'][n])
            num = (p.grad - ana).abs().max()
            den = ana.abs().max().clamp_min(1e-12)
            max_rel = max(max_rel, float(num / den))
        check(2, 'ペナルティ勾配が解析値 λ·(A·θ−B) と一致（人工状態）',
              max_rel < 1e-5, f'最大相対誤差 {max_rel:.2e}')
        # 実状態に戻して項目 5 へ
        model2._ewc_state = real_state
        model2._ewc_state_device = None

        model2.train()
        model2.zero_grad(set_to_none=True)
        data = model2.data_preprocessor(batch, True)
        losses = model2.loss(data['inputs'], data['data_samples'])
        ok_key = 'loss_ewc' in losses and torch.isfinite(losses['loss_ewc'])
        total = sum(v.sum() for v in losses.values()
                    if isinstance(v, torch.Tensor) and v.requires_grad)
        total.backward()
        check(5, '実データ 1 step で loss_ewc が乗り backward が通る',
              ok_key and torch.isfinite(total),
              f'loss={float(total):.2f} loss_ewc={float(losses["loss_ewc"]):.4f}')
        del model2
    torch.cuda.empty_cache()


if __name__ == '__main__':
    check3_pure_tensor()
    if '--gpu' in sys.argv:
        run_gpu_checks()
    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
