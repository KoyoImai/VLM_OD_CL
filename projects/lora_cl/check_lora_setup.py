"""projects/lora_cl の実装検証（実データで model.loss() まで通す）。

使い方:
    python projects/lora_cl/check_lora_setup.py [config]

既定 config は experiments/exp_008/configs/underwater_lora.py。
全項目 OK でなければ学習を始めないこと。

検証項目
  1. LoRA 層の置換が config の指定どおりに行われている
  2. init_weights() の後も B=0 が保たれる（ΔW=0）
     -> DINO.init_weights が encoder/decoder の dim>1 パラメータを一括 xavier
        初期化するため（mmdet/models/detectors/dino.py:72-75）、
        GroundingDINOLoRA.init_weights で復元し直している
  3. θ0 ロード後も ΔW=0（lora_* は ckpt に無いので上書きされない）
  4. LoRA を除いた base の state_dict が plain GroundingDINO と一致
  5. requires_grad=True は lora_A / lora_B のみ
  6. optimizer に登録されるのは LoRA のみ（TrainableParamsConstructor）
  7. 実データ 1 step の学習で base 重みが完全不変
  8. 1 step 目は全 LoRA に grad があり、非零勾配は B のみ
     -> B=0 初期化なので dL/dA = scaling·Bᵀ(dL/dy)xᵀ = 0 が理論値
  9. 2 step 目は A にも勾配が流れ、A/B とも更新される
 10. merge() が forward と数値一致（W += (α/r)BA の等価性）
"""
import sys

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from mmengine.utils import import_modules_from_strings

from mmdet.registry import MODELS

CKPT = ('/root/.cache/torch/hub/checkpoints/grounding_dino_swin-t_pretrain_'
        'obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth')

results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


if __name__ == '__main__':
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else \
        'experiments/exp_008/configs/underwater_lora.py'
    init_default_scope('mmdet')
    cfg = Config.fromfile(cfg_path)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg['custom_imports'])
    from projects.lora_cl.lora_layers import LoRALinear

    print(f'config: {cfg_path}')
    print(f'lora: {cfg.model.lora}\n')

    model = MODELS.build(cfg.model)

    # --- 1. 置換 ---------------------------------------------------------
    lora_mods = {n: m for n, m in model.named_modules()
                 if isinstance(m, LoRALinear)}
    inc = cfg.model.lora.get('include')
    outside = [n for n in lora_mods
               if inc is not None and n.split('.')[0] not in inc]
    check(1, 'LoRA 層の置換', len(lora_mods) > 0 and not outside,
          f'{len(lora_mods)} 層 / include 外の混入 {len(outside)}')

    # --- 2. init_weights 後の B=0 ---------------------------------------
    model.init_weights()
    d_max = max(float((float(m.lora_scaling) * (m.lora_B @ m.lora_A)).abs().max())
                for m in lora_mods.values())
    nb0 = sum(1 for m in lora_mods.values() if float(m.lora_B.abs().max()) == 0)
    check(2, 'init_weights 後も B=0（ΔW=0）', d_max == 0.0,
          f'max|ΔW|={d_max:.3e}, B=0 の層 {nb0}/{len(lora_mods)}')

    # --- 3. θ0 ロード後の ΔW=0 -------------------------------------------
    ck = _load_checkpoint(CKPT, map_location='cpu')
    _load_checkpoint_to_model(model, ck, strict=False)
    d_max2 = max(float((float(m.lora_scaling) * (m.lora_B @ m.lora_A)).abs().max())
                 for m in lora_mods.values())
    check(3, 'θ0 ロード後も ΔW=0', d_max2 == 0.0, f'max|ΔW|={d_max2:.3e}')

    # --- 4. base が plain GroundingDINO と一致 ---------------------------
    sd = model.state_dict()
    src = ck['state_dict'] if 'state_dict' in ck else ck
    bad = []
    for k, v in src.items():
        if k in sd and sd[k].shape == v.shape:
            if not torch.equal(sd[k].cpu(), v.cpu()):
                bad.append(k)
    check(4, 'base 重みが θ0 と一致', not bad,
          f'照合 {sum(1 for k in src if k in sd)} キー / 不一致 {len(bad)}')

    # --- 5. requires_grad --------------------------------------------------
    tr = [n for n, p in model.named_parameters() if p.requires_grad]
    bad_tr = [n for n in tr
              if not (n.endswith('lora_A') or n.endswith('lora_B'))]
    check(5, 'trainable は LoRA のみ', not bad_tr and len(tr) == 2 * len(lora_mods),
          f'{len(tr)} 個（LoRA 層 {len(lora_mods)} × 2）/ 想定外 {len(bad_tr)}')

    # --- 6. optimizer 登録 --------------------------------------------------
    from mmengine.optim import build_optim_wrapper
    ow = build_optim_wrapper(model, cfg.optim_wrapper)
    n_opt = sum(p.numel() for g in ow.optimizer.param_groups for p in g['params'])
    n_lora = sum(p.numel() for n, p in model.named_parameters()
                 if p.requires_grad)
    check(6, 'optimizer 登録は LoRA のみ', n_opt == n_lora,
          f'{n_opt:,} == {n_lora:,}')

    # --- 7/8. 実データ 1 step ----------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).train()
    loader = Runner.build_dataloader(cfg.train_dataloader)
    data = next(iter(loader))
    data = model.data_preprocessor(data, training=True)

    base_before = {n: p.detach().clone()
                   for n, p in model.named_parameters() if not p.requires_grad}
    lora_before = {n: p.detach().clone()
                   for n, p in model.named_parameters() if p.requires_grad}

    losses = model.loss(data['inputs'], data['data_samples'])
    total = sum(v.sum() for v in losses.values() if isinstance(v, torch.Tensor))
    total.backward()
    ow.optimizer.step()

    base_delta = max(float((p.detach() - base_before[n]).abs().max())
                     for n, p in model.named_parameters()
                     if n in base_before)
    check(7, '1 step 後も base 重みが不変', base_delta == 0.0,
          f'max|Δbase|={base_delta:.3e}, loss 項数={len(losses)}, '
          f'total={float(total):.4f}')

    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    gA = [n for n, p in model.named_parameters()
          if n.endswith('lora_A') and float(p.grad.abs().max()) > 0]
    gB = [n for n, p in model.named_parameters()
          if n.endswith('lora_B') and float(p.grad.abs().max()) > 0]
    # B=0 初期化のため 1 step 目の dL/dA = scaling·Bᵀ(dL/dy)xᵀ = 0 が理論値。
    # A が動き出すのは B が非零になった 2 step 目以降。
    check(8, '1 step 目: 全 LoRA に grad があり B のみ非零勾配',
          not no_grad and not gA and len(gB) > 0,
          f'grad=None {len(no_grad)} / 非零勾配 A={len(gA)} B={len(gB)}/{len(lora_mods)}')

    # --- 8b. 2 step 目で A も動く -----------------------------------------
    ow.optimizer.zero_grad()
    data2 = model.data_preprocessor(next(iter(loader)), training=True)
    losses2 = model.loss(data2['inputs'], data2['data_samples'])
    sum(v.sum() for v in losses2.values()
        if isinstance(v, torch.Tensor)).backward()
    gA2 = [n for n, p in model.named_parameters()
           if n.endswith('lora_A') and float(p.grad.abs().max()) > 0]
    ow.optimizer.step()
    movedA = sum(1 for n, p in model.named_parameters()
                 if n.endswith('lora_A')
                 and not torch.equal(p.detach(), lora_before[n]))
    movedB = sum(1 for n, p in model.named_parameters()
                 if n.endswith('lora_B')
                 and not torch.equal(p.detach(), lora_before[n]))
    check(9, '2 step 目: A にも勾配が流れ A/B とも更新される',
          len(gA2) > 0 and movedA > 0 and movedB > 0,
          f'非零勾配 A={len(gA2)}/{len(lora_mods)} / '
          f'更新された A={movedA} B={movedB}')

    # --- 9. merge 等価性 ---------------------------------------------------
    m = next(iter(lora_mods.values()))
    m = m.to(device)
    x = torch.randn(4, m.in_features, device=device)
    with torch.no_grad():
        y_lora = m(x)
        m.merge()
        y_merged = m(x)
    diff = float((y_lora - y_merged).abs().max())
    check(10, 'merge() が forward と数値一致', diff < 1e-4, f'max|Δy|={diff:.3e}')

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
