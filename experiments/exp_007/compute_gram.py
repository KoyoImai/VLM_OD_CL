"""exp_007 ステージ2: 活性化共分散 C_X = Σ h hᵀ を forward hook で計算・保存
各対象線形層 ℓ について、指定モデル(ckpt)に X のプローブを N 枚流し、入力活性化のグラム行列を累積。
これが分析C の「X の重要部分空間 S_X」の元になる（C_X の固有分解で S_X が得られる）。

使い方:
  python compute_gram.py <config> <ckpt> <out.pt> <N>
出力: {param_key: gram(d_in×d_in, cpu float64)}, 'n_tokens': {param_key:count} を out.pt に保存。
"""
import sys, os
import torch
import torch.nn as nn

REPO = '/workspace/kouyou/mmdetection'
sys.argv_backup = sys.argv

def is_target_key(name):
    if name.startswith('backbone.') or name.startswith('language_model.'):
        return False
    if 'label_embedding' in name:
        return False
    return True

def main(cfg_path, ckpt, out_path, N):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner, load_checkpoint
    from mmdet.registry import MODELS
    init_default_scope('mmdet')

    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt, map_location='cpu', strict=False)
    model = model.cuda().eval()

    # 対象モジュール（nn.Linear と 1x1 Conv2d）を param_key で特定しフック登録
    grams = {}     # param_key -> (d_in×d_in) cuda float64 acc
    counts = {}
    handles = []
    name_by_module = {}
    for mname, mod in model.named_modules():
        wkey = mname + '.weight'
        if not is_target_key(wkey):
            continue
        if isinstance(mod, nn.Linear):
            name_by_module[mod] = wkey
        elif isinstance(mod, nn.Conv2d) and mod.kernel_size == (1, 1):
            name_by_module[mod] = wkey

    def pre_hook(mod, args):
        key = name_by_module[mod]
        x = args[0]
        if isinstance(mod, nn.Linear):
            h = x.reshape(-1, x.shape[-1])               # (M, d_in)
        else:  # Conv2d 1x1: (N,C,H,W) -> (N*H*W, C)
            h = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        h = h.detach().double()
        g = h.t() @ h                                    # (d_in, d_in)
        if key in grams:
            grams[key] += g; counts[key] += h.shape[0]
        else:
            grams[key] = g; counts[key] = h.shape[0]

    for mod in name_by_module:
        handles.append(mod.register_forward_pre_hook(pre_hook))

    loader = Runner.build_dataloader(cfg.test_dataloader)
    seen = 0
    with torch.no_grad():
        for batch in loader:
            model.test_step(batch)
            seen += len(batch['data_samples'])
            if seen >= N:
                break
    for h in handles:
        h.remove()

    out = {'gram': {k: v.cpu() for k, v in grams.items()},
           'n_tokens': counts, 'n_images': seen, 'cfg': cfg_path, 'ckpt': ckpt}
    torch.save(out, out_path)
    print(f"[done] {out_path}: layers={len(grams)} images={seen}")

if __name__ == '__main__':
    cfg_path, ckpt, out_path, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    main(cfg_path, ckpt, out_path, N)
