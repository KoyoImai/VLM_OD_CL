"""exp_007 ステージ3: 干渉指標(分析C) と 忘却との相関(分析D) — COCO忘却(単独FT, 6点, base座標系)
干渉(i→COCO) = Σ_ℓ trace(ΔW_i^ℓ C_COCO^ℓ ΔW_i^ℓᵀ) = i の更新が COCO 層出力を乱す量。
これを exp_005/004 の実測 COCO 忘却量と相関させる。
"""
import os, json, math
import torch

REPO = '/workspace/kouyou/mmdetection'
G = f'{REPO}/experiments/exp_007/gram'
BASE = '/root/.cache/torch/hub/checkpoints/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
FT = {
    'underwater':      f'{REPO}/experiments/exp_004/underwater_work_dir/best_coco_bbox_mAP_epoch_20.pth',
    'aerial':          f'{REPO}/experiments/exp_005/aerial_work_dir/best_coco_bbox_mAP_epoch_17.pth',
    'microscopic':     f'{REPO}/experiments/exp_005/microscopic_work_dir/best_coco_bbox_mAP_epoch_20.pth',
    'videogames':      f'{REPO}/experiments/exp_005/videogames_work_dir/best_coco_bbox_mAP_epoch_20.pth',
    'documents':       f'{REPO}/experiments/exp_005/documents_work_dir/best_coco_bbox_mAP_epoch_19.pth',
    'electromagnetic': f'{REPO}/experiments/exp_005/electromagnetic_work_dir/best_coco_bbox_mAP_epoch_18.pth',
}
# 実測 COCO 忘却 F_COCO = 0.504 - COCO_after  (exp_005/004)
COCO_AFTER = dict(underwater=0.414, aerial=0.318, microscopic=0.327,
                  videogames=0.324, documents=0.275, electromagnetic=0.331)
F_COCO = {d: round(0.504 - v, 3) for d, v in COCO_AFTER.items()}

def load_sd(p):
    sd = torch.load(p, map_location='cpu'); return sd.get('state_dict', sd)

def as2d(v):
    if v.ndim == 2: return v
    if v.ndim == 4 and v.shape[2] == 1 and v.shape[3] == 1: return v[:, :, 0, 0]
    return None

def topk_proj_ratio(dW, C, energy=0.90):
    # ‖dW S‖²/‖dW‖²  (S=C の上位固有ベクトル, 累積energy)
    evals, evecs = torch.linalg.eigh(C)             # 昇順
    evals = evals.flip(0); evecs = evecs.flip(1)
    evals = evals.clamp(min=0)
    c = torch.cumsum(evals, 0) / (evals.sum() + 1e-12)
    k = int((c < energy).sum().item()) + 1
    S = evecs[:, :k]
    num = (dW @ S).pow(2).sum()
    den = dW.pow(2).sum() + 1e-12
    return float(num / den), k

def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    sx = sum((a-mx)**2 for a in x); sy = sum((b-my)**2 for b in y)
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    return sxy / (math.sqrt(sx*sy) + 1e-12)

def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0]*len(v)
        for rk, i in enumerate(order): r[i] = rk
        return r
    return pearson(rank(x), rank(y))

def main():
    base = load_sd(BASE)
    coco_g = torch.load(f'{G}/coco_base.pt', map_location='cpu')['gram']
    rows = []
    per_layer_dump = {}
    for dom, p in FT.items():
        ft = load_sd(p)
        I_abs = 0.0; dW_norm2 = 0.0; rel_list = []; dir_list = []
        contrib = {}    # 層別 絶対干渉
        for key, C in coco_g.items():
            if key not in base or key not in ft: continue
            Wb = as2d(base[key]); Wf = as2d(ft[key])
            if Wb is None or Wf is None or Wb.shape != Wf.shape: continue
            if C.shape[0] != Wb.shape[1]: continue
            dW = (Wf - Wb).double(); Wb = Wb.double(); C = C.double()
            i_abs = float(((dW @ C) * dW).sum())             # trace(dW C dWᵀ)
            base_out = float(((Wb @ C) * Wb).sum()) + 1e-12  # trace(Wb C Wbᵀ)
            I_abs += i_abs
            dW_norm2 += float(dW.pow(2).sum())
            rel_list.append(i_abs / base_out)
            r, _ = topk_proj_ratio(dW, C); dir_list.append(r)
            contrib[key] = i_abs
        I_rel = sum(rel_list)/len(rel_list)
        I_dir = sum(dir_list)/len(dir_list)
        rows.append(dict(domain=dom, F_COCO=F_COCO[dom],
                         I_abs=I_abs, I_rel=round(I_rel, 4), I_dir=round(I_dir, 4),
                         dW_norm2=round(dW_norm2, 3)))
        # 層別 top寄与
        top = sorted(contrib.items(), key=lambda kv: -kv[1])[:8]
        per_layer_dump[dom] = [(k, round(v, 4)) for k, v in top]

    # 相関
    doms = [r['domain'] for r in rows]
    F = [r['F_COCO'] for r in rows]
    def corr(metric):
        x = [r[metric] for r in rows]
        return round(pearson(x, F), 3), round(spearman(x, F), 3)
    print(f"\n{'domain':16s} {'F_COCO':>7s} {'I_abs':>12s} {'I_rel':>8s} {'I_dir':>7s} {'dW_norm2':>9s}")
    for r in sorted(rows, key=lambda r: r['F_COCO']):
        print(f"{r['domain']:16s} {r['F_COCO']:7.3f} {r['I_abs']:12.1f} {r['I_rel']:8.4f} {r['I_dir']:7.4f} {r['dW_norm2']:9.2f}")
    print("\n=== 干渉 vs COCO忘却 の相関 (Pearson, Spearman) ===")
    for m in ['I_abs', 'I_rel', 'I_dir', 'dW_norm2']:
        p_, s_ = corr(m)
        print(f"  {m:10s} Pearson={p_:+.3f}  Spearman={s_:+.3f}")

    out = dict(rows=rows, F_COCO=F_COCO,
               corr={m: dict(zip(['pearson','spearman'], corr(m))) for m in ['I_abs','I_rel','I_dir','dW_norm2']},
               top_layers=per_layer_dump)
    with open(f'{REPO}/experiments/exp_007/results/interference_vs_forgetting_coco.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(f"\n保存: results/interference_vs_forgetting_coco.json")

if __name__ == '__main__':
    main()
