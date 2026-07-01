#!/usr/bin/env python3
# =============================================================================
# exp_014: Feature Enhancer(= encoder) を θ0 に戻したハイブリッドを生成する。
#   - 学習は一切行わない。既存 state_dict を後付けで合成するのみ。
#   - enc    : encoder.*               全体（276）を θ0 へ
#   - fusion : encoder.fusion_layers.* 画像↔テキスト双方向 cross-attn（108）
#   - text   : encoder.text_layers.*   テキスト self-attn（72）
#   - image  : encoder.layers.*        画像 deformable self-attn（96）
#   - backbone/neck/language_model/text_feat_map/decoder/bbox_head は unfrozen のまま。
#   - 整合: enc = fusion ∪ text ∪ image（276 = 108+72+96）。定数バッファ無し。
# 出力: experiments/exp_014/hybrids/{domain}_{group}_theta0.pth（24本）
# 参照: experiments/exp_014/design.md
# =============================================================================
import os
import copy
import torch

REPO = '/workspace/kouyou/mmdetection'
THETA0 = os.path.join(
    REPO,
    'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth')

THETA1_U = {
    'underwater':      'experiments/exp_010/underwater_unfrozen_work_dir/best_coco_bbox_mAP_epoch_18.pth',
    'aerial':          'experiments/exp_011/aerial_unfrozen_work_dir/best_coco_bbox_mAP_epoch_19.pth',
    'microscopic':     'experiments/exp_011/microscopic_unfrozen_work_dir/best_coco_bbox_mAP_epoch_18.pth',
    'videogames':      'experiments/exp_011/videogames_unfrozen_work_dir/best_coco_bbox_mAP_epoch_20.pth',
    'documents':       'experiments/exp_011/documents_unfrozen_work_dir/best_coco_bbox_mAP_epoch_16.pth',
    'electromagnetic': 'experiments/exp_011/electromagnetic_unfrozen_work_dir/best_coco_bbox_mAP_epoch_19.pth',
}

# ロールバックするモジュール接頭辞のグループ（注: image は encoder.layers. だが
# encoder.fusion_layers./encoder.text_layers. も startswith('encoder.layers') では
# ないため曖昧さは無い。enc は全体）
GROUPS = {
    'enc':    ('encoder.',),                  # Feature Enhancer 全体 276
    'fusion': ('encoder.fusion_layers.',),    # 双方向 cross-attn 108
    'text':   ('encoder.text_layers.',),      # テキスト self-attn 72
    'image':  ('encoder.layers.',),           # 画像 deformable self-attn 96
}
OUT_DIR = os.path.join(REPO, 'experiments/exp_014/hybrids')


def get_sd(ckpt):
    return ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt


def in_group(key, prefixes):
    return any(key.startswith(p) for p in prefixes)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'[load] θ0: {THETA0}')
    sd0 = get_sd(torch.load(THETA0, map_location='cpu'))

    for d, rel in THETA1_U.items():
        print(f'\n[domain={d}] load θ1_d^U: {rel}')
        ckpt_u = torch.load(os.path.join(REPO, rel), map_location='cpu')
        sd_u = get_sd(ckpt_u)

        for gname, prefixes in GROUPS.items():
            # 置換対象 = unfrozen 側に存在する対象接頭辞キー
            tgt = [k for k in sd_u if in_group(k, prefixes)]
            missing = [k for k in tgt if k not in sd0]
            assert not missing, f'{d}/{gname}: θ0 に無い対象キー {missing[:3]}'
            only0 = [k for k in sd0 if in_group(k, prefixes) and k not in sd_u]  # 定数バッファ想定（0のはず）

            new_sd = copy.deepcopy(sd_u)
            for k in tgt:
                assert new_sd[k].shape == sd0[k].shape, f'{d}/{gname}: shape不一致 {k}'
                new_sd[k] = sd0[k].clone()

            # サニティ: 対象は θ0 と一致、非対象は θ1_d^U と一致
            for k in tgt:
                assert torch.equal(new_sd[k], sd0[k]), f'{d}/{gname}: 対象 {k} が θ0 と不一致'
            non_tgt = next(k for k in sd_u if not in_group(k, prefixes))
            assert torch.equal(new_sd[non_tgt], sd_u[non_tgt]), \
                f'{d}/{gname}: 非対象 {non_tgt} が θ1_d^U と不一致'

            out_ckpt = copy.deepcopy(ckpt_u)
            if isinstance(out_ckpt, dict) and 'state_dict' in out_ckpt:
                out_ckpt['state_dict'] = new_sd
                out_ckpt.pop('optimizer', None)
                out_ckpt.pop('ema_state_dict', None)
            else:
                out_ckpt = new_sd
            out_path = os.path.join(OUT_DIR, f'{d}_{gname}_theta0.pth')
            torch.save(out_ckpt, out_path)
            print(f'  [{gname}] 置換 {len(tgt)} params (θ0のみ={len(only0)}) -> {out_path}')

    print('\n[done] 全24ハイブリッド生成完了。')


if __name__ == '__main__':
    main()
