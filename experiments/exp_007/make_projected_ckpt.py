"""exp_007 分析E: ΔW から COCO 部分空間成分を直交除去したチェックポイントを作る。
W' = W_base + ΔW (I - S_COCO S_COCOᵀ)   （= W_ft - ΔW・P_COCO）
S_COCO = 各層の COCO 活性化共分散 C_COCO の上位固有ベクトル（累積energy）。
非対象パラメータ(norm/bias/backbone/language/label_embedding 等)は ft のまま。

使い方: python make_projected_ckpt.py <domain> <energy> <out.pth>
"""
import sys, os
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

def load(p):
    d = torch.load(p, map_location='cpu'); return d if 'state_dict' in d else {'state_dict': d}

def proj_complement(C, energy):
    evals, evecs = torch.linalg.eigh(C.double())
    evals = evals.flip(0).clamp(min=0); evecs = evecs.flip(1)
    c = torch.cumsum(evals, 0) / (evals.sum() + 1e-12)
    k = int((c < energy).sum().item()) + 1
    S = evecs[:, :k]
    I = torch.eye(C.shape[0], dtype=torch.float64)
    return I - S @ S.t(), k   # 直交補への射影 (I - SSᵀ)

def main(domain, energy, out):
    base = load(BASE)['state_dict']
    ftfull = load(FT[domain]); ft = ftfull['state_dict']
    coco_g = torch.load(f'{G}/coco_base.pt', map_location='cpu')['gram']
    new = {k: v.clone() for k, v in ft.items()}
    n_mod, removed_energy, total_energy = 0, 0.0, 0.0
    ks = []
    for key, C in coco_g.items():
        if key not in base or key not in ft: continue
        Wb, Wf = base[key], ft[key]
        is_conv = (Wb.ndim == 4 and Wb.shape[2:] == (1, 1))
        Wb2 = Wb[:, :, 0, 0] if is_conv else Wb
        Wf2 = Wf[:, :, 0, 0] if is_conv else Wf
        if Wb2.ndim != 2 or Wb2.shape != Wf2.shape or C.shape[0] != Wb2.shape[1]:
            continue
        dW = (Wf2 - Wb2).double()
        Pc, k = proj_complement(C, energy)
        dW_orth = dW @ Pc                          # 直交補成分（COCO方向を除去）
        removed = float((dW.pow(2).sum() - dW_orth.pow(2).sum()))
        removed_energy += removed; total_energy += float(dW.pow(2).sum()); ks.append(k)
        Wnew = (Wb2.double() + dW_orth).to(Wf.dtype)
        new[key] = Wnew.view_as(Wf) if is_conv else Wnew
        n_mod += 1
    ftfull['state_dict'] = new
    torch.save(ftfull, out)
    import statistics as st
    print(f"[{domain}] energy={energy} 改変層={n_mod} 平均k={st.mean(ks):.0f} "
          f"除去された更新エネルギー割合={removed_energy/(total_energy+1e-12):.3f}")
    print(f"  -> {out}")

if __name__ == '__main__':
    domain, energy, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    main(domain, energy, out)
