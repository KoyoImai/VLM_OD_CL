"""InfLoRA 実装の検証（exp_042 design.md §4.2 の項目 4〜6 ＋ 実データ確認）。

使い方:
    python projects/lora_cl/check_inflora_setup.py            # 純テンソル項目のみ。GPU 不要
    python projects/lora_cl/check_inflora_setup.py --gpu      # 全項目（GPU 1 枚・実データ）

検証項目
  1. 挿入と学習対象: exp_038 条件A と同じ 232 層に InfLoRALinear が入り、学習対象が
     lora_B のみ（A は凍結）、lora_scaling=1（alpha=r → ΔW=B·A）であること
  2. design_A の性質（純テンソル）: メモリ無しで A の行が正規直交（×1/√3）、
     'remove' メモリと直交、'retain' メモリの張る空間内に入ること
  3. DualGPM 更新の性質（純テンソル）: 基底の列が正規直交、retain 型が層次元の
     半分以下、複数タスクで一貫して更新されること
  4. 共分散収集が forward を変えないこと（収集 on/off で loss が一致）
  5. 実データパイプライン: prepare（設計）→ 学習 1 step（A 不変・B のみ更新・loss 有限）
     → update_memory → メモリ有り prepare で設計 A が 'remove' メモリと直交（実データ版）
  6. init_weights で設計 A が保存されること（DINO の xavier 一括初期化に耐える）
  7. 層単位のマージ等価: (W + B·A)x が分岐 forward と一致（lora_scaling=1 の確認込み）
"""
import copy
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

TASK_CFG = os.path.join(ROOT, 'experiments/exp_038/configs/lora_odinw13_pistols.py')

results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


def make_inflora_cfg(design_path=None):
    from mmengine.config import Config
    cfg = Config.fromfile(TASK_CFG)
    m = cfg.model
    m['type'] = 'GroundingDINOInfLoRA'
    m['lora']['alpha'] = m['lora']['r']          # scaling=1（公式の ΔW=B·A）
    m['inflora'] = dict(design_path=design_path)
    return cfg


def check_pure_tensor():
    from projects.lora_cl.dual_gpm import (design_A, new_empty_memory,
                                           update_dual_gpm)
    rng = np.random.default_rng(0)
    d, r = 32, 4

    def rand_cov(k=48, decay=1.0):
        # decay < 1 でスペクトルが集中した共分散（実活性に近い）→ remove 型の経路。
        # decay = 1 は等方に近く r >= d/2 → retain 型（直交補保存）の経路を通す。
        X = rng.standard_normal((k, d)).astype(np.float32)
        X = X * (decay ** np.arange(d, dtype=np.float32))[None, :]
        return X.T @ X / k

    # --- 2. design_A の性質 --------------------------------------------------
    A0 = design_A(rand_cov(), r)                      # メモリ無し
    ortho = np.abs(3.0 * (A0 @ A0.T) - np.eye(r)).max()   # 行は 1/√3 スケール
    M = np.linalg.qr(rng.standard_normal((d, 6)))[0].astype(np.float32)
    Ar = design_A(rand_cov(), r, feature=M, ptype='remove')
    perp = np.abs(Ar @ M).max()                       # remove: A ⊥ M
    At = design_A(rand_cov(), r, feature=M, ptype='retain')
    inside = np.abs(At - (At @ M) @ M.T).max()        # retain: A ∈ span(M)
    check(2, 'design_A: 正規直交・remove と直交・retain の空間内',
          ortho < 1e-5 and perp < 1e-5 and inside < 1e-5,
          f'直交性 {ortho:.1e} / remove 直交 {perp:.1e} / retain 内 {inside:.1e}')

    # --- 3. DualGPM 更新の性質 -----------------------------------------------
    mem = new_empty_memory()
    ok, info = True, []
    for t in range(3):
        covs = {'l1': rand_cov(decay=0.8), 'l2': rand_cov(200)}
        update_dual_gpm(covs, mem, 0.95 + 0.05 * t / 3)
        for n, F in mem['feature'].items():
            if F.shape[1] == 0:
                # retain 型が縮退した正当な状態（等方データのストレス経路）。
                # design_A 側のガードで止まることは項目 2 の範囲外なのでここでは許容。
                continue
            g = np.abs(F.T @ F - np.eye(F.shape[1])).max()
            if g > 1e-4:
                ok = False
                info.append(f'{n}: 非直交 {g:.1e}')
            if mem['ptype'][n] == 'retain' and F.shape[1] > F.shape[0] / 2:
                ok = False
                info.append(f'{n}: retain 超過')
    sizes = {n: (F.shape[1], mem['ptype'][n]) for n, F in mem['feature'].items()}
    check(3, 'DualGPM: 基底が正規直交・retain が半分以下・3 タスク一貫', ok,
          f'{sizes}' + (f' / {info[:2]}' if info else ''))


def run_gpu_checks():
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner.checkpoint import load_checkpoint
    from mmengine.dataset import pseudo_collate
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import DATASETS, MODELS
    from projects.lora_cl.inflora import InfLoRALinear
    from projects.lora_cl.dual_gpm import load_memory

    init_default_scope('mmdet')
    base = Config.fromfile(TASK_CFG)
    import_modules_from_strings(**base['custom_imports'])
    dev = 'cuda:0'
    tmpdir = tempfile.mkdtemp(prefix='inflora_check_')
    cfg_path = os.path.join(tmpdir, 'inflora_task_cfg.py')
    make_inflora_cfg().dump(cfg_path)

    # --- 1. 挿入と学習対象 ---------------------------------------------------
    model = MODELS.build(copy.deepcopy(make_inflora_cfg().model))
    layers = model.inflora_layers()
    tr = [n for n, p in model.named_parameters() if p.requires_grad]
    ok_tr = all(n.endswith('lora_B') for n in tr) and len(tr) == len(layers)
    scal = {float(m.lora_scaling) for _, m in layers}
    check(1, '232 層に挿入・学習対象は lora_B のみ・scaling=1',
          len(layers) == 232 and ok_tr and scal == {1.0},
          f'挿入 {len(layers)} 層 / trainable {len(tr)} / scaling {scal}')

    # --- 4. 収集が forward を変えない ----------------------------------------
    from projects.lora_cl.inflora_collect import loss_mode_dropout_off
    load_checkpoint(model, base.load_from, map_location='cpu')
    model = model.to(dev)
    loss_mode_dropout_off(model)
    dataset = DATASETS.build(base.train_dataloader['dataset'])
    batch = pseudo_collate([dataset[0], dataset[1]])
    data = model.data_preprocessor(batch, True)

    def total_loss():
        with torch.no_grad():
            losses = model.loss(copy.deepcopy(data['inputs']),
                                copy.deepcopy(data['data_samples']))
        return float(sum(v.sum() for v in losses.values()
                         if isinstance(v, torch.Tensor)))

    l_off = total_loss()
    model.set_input_collection(True)
    l_on = total_loss()
    model.set_input_collection(False)
    covs = model.pop_covariances()
    sym = max(float((c - c.t()).abs().max()) for c in covs.values())
    check(4, '共分散収集が forward を変えない（loss 一致・共分散は対称）',
          abs(l_on - l_off) < 1e-4 and sym < 1e-3,
          f'loss off={l_off:.4f} on={l_on:.4f} / 非対称最大 {sym:.1e}')
    del model
    torch.cuda.empty_cache()

    # --- 5. 実データパイプライン --------------------------------------------
    design1 = os.path.join(tmpdir, 'design_t1.pth')
    mem1 = os.path.join(tmpdir, 'memory_t1.pth')
    design2 = os.path.join(tmpdir, 'design_t2.pth')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get(
        'CUDA_VISIBLE_DEVICES', '0'))

    def run(script, *extra):
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, script), cfg_path,
             base.load_from, '--max-samples', '4', '--seed', '0',
             '--device', 'cuda:0', *extra],
            capture_output=True, text=True, cwd=ROOT, env=env)
        return r

    r1 = run('inflora_prepare.py', '--out', design1)
    ok1 = r1.returncode == 0 and os.path.exists(design1)
    r2 = run('inflora_update_memory.py', '--out', mem1,
             '--task-index', '0', '--total', '13')
    ok2 = r2.returncode == 0 and os.path.exists(mem1)
    r3 = run('inflora_prepare.py', '--out', design2, '--memory', mem1)
    ok3 = r3.returncode == 0 and os.path.exists(design2)
    detail = ''
    if not (ok1 and ok2 and ok3):
        bad = r1 if not ok1 else (r2 if not ok2 else r3)
        detail = (bad.stderr or bad.stdout)[-300:]
    ortho_real = float('nan')
    if ok3:
        mem = load_memory(mem1)
        d2 = torch.load(design2, map_location='cpu')['lora_A']
        vals = []
        for n, A in d2.items():
            if mem['ptype'].get(n) == 'remove':
                vals.append(float(np.abs(np.asarray(A) @ mem['feature'][n]).max()))
        ortho_real = max(vals) if vals else float('nan')
        n_rm = sum(1 for t in mem['ptype'].values() if t == 'remove')
        detail = (f'remove {n_rm}/{len(mem["ptype"])} 層 / '
                  f'設計 A とメモリの直交性 max {ortho_real:.2e}')
    check(5, 'prepare → メモリ更新 → メモリ有り設計で A がメモリと直交（実データ）',
          ok1 and ok2 and ok3 and (np.isnan(ortho_real) or ortho_real < 1e-4),
          detail)

    # --- 6. init_weights で設計が保存 / 学習 1 step ---------------------------
    if ok1:
        cfg2 = make_inflora_cfg(design_path=design1)
        model2 = MODELS.build(copy.deepcopy(cfg2.model))
        d1 = torch.load(design1, map_location='cpu')['lora_A']
        model2.init_weights()      # DINO の xavier 一括初期化を通す
        ok_keep, ok_frozen = True, True
        for n, m in model2.inflora_layers():
            if not torch.allclose(m.lora_A.detach(),
                                  torch.as_tensor(np.asarray(d1[n]))):
                ok_keep = False
            if m.lora_A.requires_grad or not m.lora_B.requires_grad \
                    or m.lora_B.abs().max() > 0:
                ok_frozen = False
        check(6, 'init_weights 後も設計 A が保存され、A 凍結・B=0 学習可',
              ok_keep and ok_frozen, f'A 保存={ok_keep} 凍結/B0={ok_frozen}')

        load_checkpoint(model2, base.load_from, map_location='cpu')
        model2 = model2.to(dev).train()
        a_before = {n: m.lora_A.detach().clone()
                    for n, m in model2.inflora_layers()}
        opt = torch.optim.AdamW(
            [p for p in model2.parameters() if p.requires_grad], lr=1e-3)
        data = model2.data_preprocessor(batch, True)
        losses = model2.loss(data['inputs'], data['data_samples'])
        total = sum(v.sum() for v in losses.values()
                    if isinstance(v, torch.Tensor) and v.requires_grad)
        total.backward()
        opt.step()
        a_moved = [n for n, m in model2.inflora_layers()
                   if not torch.equal(a_before[n], m.lora_A.detach())]
        b_moved = sum(1 for _, m in model2.inflora_layers()
                      if m.lora_B.abs().max() > 0)
        check('6b', '実データ 1 step: loss 有限・A 不変・B のみ更新',
              torch.isfinite(total) and not a_moved and b_moved > 0,
              f'loss={float(total):.2f} / A 変動 {len(a_moved)} / '
              f'B 非零 {b_moved}/{len(a_before)} 層')

        # --- 7. 層単位のマージ等価 -------------------------------------------
        name, m = model2.inflora_layers()[0]
        x = torch.randn(5, m.in_features, device=dev)
        y_branch = m(x)
        y_manual = torch.nn.functional.linear(
            x, m.weight + m.lora_B @ m.lora_A, m.bias)
        diff = float((y_branch - y_manual).abs().max())
        check(7, '層単位で (W + B·A)x が分岐 forward と一致', diff < 1e-4,
              f'{name} max|Δ|={diff:.2e}')
        del model2
    torch.cuda.empty_cache()


if __name__ == '__main__':
    check_pure_tensor()
    if '--gpu' in sys.argv:
        run_gpu_checks()
    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
