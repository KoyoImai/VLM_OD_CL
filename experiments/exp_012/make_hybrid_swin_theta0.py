#!/usr/bin/env python3
# =============================================================================
# exp_012 第1 swap: unfrozen モデル θ1_d^U の Swin backbone(`backbone.*`) だけを
#                    事前学習 θ0 の値に戻したハイブリッド H_d を生成する。
#   - 学習は一切行わない。既存の学習済み state_dict を後付けで合成するのみ。
#   - BERT(`language_model.*`), neck/encoder/decoder/bbox_head は unfrozen のまま。
# 出力: experiments/exp_012/hybrids/{domain}_swin_theta0.pth
# 評価は別途 eval_base_coco.py で ZCOCO を測る（本スクリプトは生成のみ）。
# 参照: experiments/exp_012/design.md
# =============================================================================
import os
import copy
import torch

REPO = '/workspace/kouyou/mmdetection'
THETA0 = os.path.join(
    REPO,
    'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth')

# unfrozen(θ1_d^U) best ckpt（帰属対象）
THETA1_U = {
    'underwater':      'experiments/exp_010/underwater_unfrozen_work_dir/best_coco_bbox_mAP_epoch_18.pth',
    'aerial':          'experiments/exp_011/aerial_unfrozen_work_dir/best_coco_bbox_mAP_epoch_19.pth',
    'microscopic':     'experiments/exp_011/microscopic_unfrozen_work_dir/best_coco_bbox_mAP_epoch_18.pth',
    'videogames':      'experiments/exp_011/videogames_unfrozen_work_dir/best_coco_bbox_mAP_epoch_20.pth',
    'documents':       'experiments/exp_011/documents_unfrozen_work_dir/best_coco_bbox_mAP_epoch_16.pth',
    'electromagnetic': 'experiments/exp_011/electromagnetic_unfrozen_work_dir/best_coco_bbox_mAP_epoch_19.pth',
}

SWAP_PREFIX = 'backbone.'          # Swin 画像バックボーン（BERT=language_model. は触らない）
OUT_DIR = os.path.join(REPO, 'experiments/exp_012/hybrids')


def get_sd(ckpt):
    """mmengine ckpt から state_dict を取り出す。"""
    return ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'[load] θ0: {THETA0}')
    sd0 = get_sd(torch.load(THETA0, map_location='cpu'))
    bb_keys0 = {k for k in sd0 if k.startswith(SWAP_PREFIX)}
    print(f'  θ0 の {SWAP_PREFIX}* パラメータ数: {len(bb_keys0)}')

    for d, rel in THETA1_U.items():
        path_u = os.path.join(REPO, rel)
        print(f'\n[domain={d}]')
        print(f'  [load] θ1_d^U: {rel}')
        ckpt_u = torch.load(path_u, map_location='cpu')
        sd_u = get_sd(ckpt_u)

        bb_keys_u = {k for k in sd_u if k.startswith(SWAP_PREFIX)}
        # --- サニティ1: backbone キー集合が θ0 と一致（同一アーキ） ---
        assert bb_keys_u == bb_keys0, (
            f'{d}: backbone キー集合が θ0 と不一致 '
            f'(only_in_u={sorted(bb_keys_u - bb_keys0)[:3]}, '
            f'only_in_0={sorted(bb_keys0 - bb_keys_u)[:3]})')

        # --- swap: backbone.* のみ θ0 に置換、他は据え置き ---
        new_sd = copy.deepcopy(sd_u)
        replaced = 0
        for k in bb_keys0:
            assert new_sd[k].shape == sd0[k].shape, f'{d}: shape不一致 {k}'
            new_sd[k] = sd0[k].clone()
            replaced += 1

        # --- サニティ2: backbone.* は θ0 と完全一致、非backboneは θ1_d^U と一致 ---
        for k in bb_keys0:
            assert torch.equal(new_sd[k], sd0[k]), f'{d}: backbone {k} が θ0 と不一致'
        sample_non_bb = next(k for k in sd_u if not k.startswith(SWAP_PREFIX))
        assert torch.equal(new_sd[sample_non_bb], sd_u[sample_non_bb]), \
            f'{d}: 非backbone {sample_non_bb} が θ1_d^U と不一致'
        print(f'  [swap] {SWAP_PREFIX}* を θ0 に置換: {replaced} params '
              f'(非backbone {len(sd_u) - len(bb_keys0)} params は unfrozen のまま)')

        # --- 保存: unfrozen ckpt の構造を維持し state_dict だけ差し替え ---
        out_ckpt = copy.deepcopy(ckpt_u)
        if isinstance(out_ckpt, dict) and 'state_dict' in out_ckpt:
            out_ckpt['state_dict'] = new_sd
            out_ckpt.pop('optimizer', None)   # 評価に不要・サイズ削減
            out_ckpt.pop('ema_state_dict', None)
        else:
            out_ckpt = new_sd
        out_path = os.path.join(OUT_DIR, f'{d}_swin_theta0.pth')
        torch.save(out_ckpt, out_path)
        print(f'  [save] {out_path}')

    print('\n[done] 全ドメインのハイブリッド生成完了。')


if __name__ == '__main__':
    main()
