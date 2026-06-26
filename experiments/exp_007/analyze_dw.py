"""exp_007 ステージ1: ΔW ベース分析（forward 不要）
- 分析B: 各線形層の重み変化 ΔW = W_ft - W_base の低ランク性（特異値スペクトル/実効ランク）
- ΔW の大きさ ‖ΔW‖_F（後段の干渉の交絡統制・正規化に使用）
対象: 学習される共有線形層（encoder/decoder/bbox_head の Linear, neck の 1x1 conv）。
      backbone/language は凍結=対象外、label_embedding はクラス依存=除外。
"""
import os, json, collections
import torch

REPO = '/workspace/kouyou/mmdetection'
BASE = '/root/.cache/torch/hub/checkpoints/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
# 単独FT(凍結, exp_005/004)の best ckpt
FT = {
    'underwater':      f'{REPO}/experiments/exp_004/underwater_work_dir/best_coco_bbox_mAP_epoch_20.pth',
    'aerial':          f'{REPO}/experiments/exp_005/aerial_work_dir/best_coco_bbox_mAP_epoch_17.pth',
    'microscopic':     f'{REPO}/experiments/exp_005/microscopic_work_dir/best_coco_bbox_mAP_epoch_20.pth',
    'videogames':      f'{REPO}/experiments/exp_005/videogames_work_dir/best_coco_bbox_mAP_epoch_20.pth',
    'documents':       f'{REPO}/experiments/exp_005/documents_work_dir/best_coco_bbox_mAP_epoch_19.pth',
    'electromagnetic': f'{REPO}/experiments/exp_005/electromagnetic_work_dir/best_coco_bbox_mAP_epoch_18.pth',
}

def load_sd(p):
    sd = torch.load(p, map_location='cpu')
    return sd.get('state_dict', sd)

def is_target(k, v):
    """学習される共有線形層の weight か判定し、2D(linear)行列を返す。それ以外は None。"""
    if not k.endswith('.weight'):
        return None
    if k.startswith('backbone.') or k.startswith('language_model.'):
        return None
    if 'label_embedding' in k:   # クラス依存=除外
        return None
    if v.ndim == 2:
        return v
    if v.ndim == 4 and v.shape[2] == 1 and v.shape[3] == 1:  # 1x1 conv = チャネル線形
        return v[:, :, 0, 0]
    return None

def module_group(k):
    return k.split('.')[0]

def eff_rank(s):  # 実効ランク（特異値分布のエントロピー exp）
    p = s / (s.sum() + 1e-12)
    H = -(p * (p + 1e-12).log()).sum()
    return float(H.exp())

def energy_in_topr(s, r):
    e = (s**2)
    return float(e[:r].sum() / (e.sum() + 1e-12))

def main():
    base = load_sd(BASE)
    rows = []           # 層別
    group_acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for dom, p in FT.items():
        if not os.path.exists(p):
            print(f"[skip] {dom}: {p} なし"); continue
        ft = load_sd(p)
        for k, vb in base.items():
            mb = is_target(k, vb)
            if mb is None or k not in ft:
                continue
            mf = is_target(k, ft[k])
            if mf is None or mf.shape != mb.shape:
                continue
            dW = (mf - mb).float()
            if dW.abs().max() == 0:
                continue
            s = torch.linalg.svdvals(dW)
            full = min(dW.shape)
            er = eff_rank(s)
            row = dict(domain=dom, layer=k, group=module_group(k),
                       out=dW.shape[0], inn=dW.shape[1], full_rank=full,
                       eff_rank=round(er, 2), eff_rank_ratio=round(er/full, 3),
                       e_top8=round(energy_in_topr(s, 8), 3),
                       e_top16=round(energy_in_topr(s, 16), 3),
                       e_top32=round(energy_in_topr(s, 32), 3),
                       frob=round(float(dW.norm()), 4))
            rows.append(row)
            g = group_acc[dom][module_group(k)]
            g.append((er/full, energy_in_topr(s, 16), float(dW.norm())))

    os.makedirs(f'{REPO}/experiments/exp_007/results', exist_ok=True)
    with open(f'{REPO}/experiments/exp_007/results/dw_lowrank_perlayer.json', 'w') as f:
        json.dump(rows, f, indent=1)

    # ドメイン×モジュール群の集約サマリ
    print(f"\n{'domain':16s} {'group':12s} {'#layers':>7s} {'eff_rank_ratio':>14s} {'E_top16':>8s} {'mean‖ΔW‖':>9s}")
    summary = []
    for dom in FT:
        for g, lst in group_acc.get(dom, {}).items():
            if not lst: continue
            import statistics as st
            err = st.mean(x[0] for x in lst)
            e16 = st.mean(x[1] for x in lst)
            fr = st.mean(x[2] for x in lst)
            print(f"{dom:16s} {g:12s} {len(lst):7d} {err:14.3f} {e16:8.3f} {fr:9.3f}")
            summary.append(dict(domain=dom, group=g, n=len(lst),
                                eff_rank_ratio=round(err,3), E_top16=round(e16,3), mean_frob=round(fr,4)))
    with open(f'{REPO}/experiments/exp_007/results/dw_lowrank_summary.json', 'w') as f:
        json.dump(summary, f, indent=1)
    print(f"\n対象層数(全ドメイン計): {len(rows)}  → results/dw_lowrank_*.json に保存")

if __name__ == '__main__':
    main()
