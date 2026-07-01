#!/usr/bin/env python3
# =============================================================================
# exp_012 第2 swap: unfrozen モデル θ1_d^U の Text Backbone(BERT, `language_model.*`)
#                    だけを事前学習 θ0 の値に戻したハイブリッド H^BERT_d を生成する。
#   - 学習は一切行わない。既存の学習済み state_dict を後付けで合成するのみ。
#   - Swin(`backbone.*`) は unfrozen のまま（ドメイン適応済み）。neck/enc/dec/head も unfrozen。
#   - θ0 は language_model.* が 198、unfrozen は 197（θ0のみ ...embeddings.position_ids＝定数バッファ）。
#     両者共通の 197 キーを θ0 値に置換（position_ids は定数のため省略で無害）。
# 出力: experiments/exp_012/hybrids/{domain}_bert_theta0.pth
# 参照: experiments/exp_012/design.md
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

SWAP_PREFIX = 'language_model.'      # Text バックボーン(BERT)。Swin(backbone.) は触らない。
OUT_DIR = os.path.join(REPO, 'experiments/exp_012/hybrids')


def get_sd(ckpt):
    return ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'[load] θ0: {THETA0}')
    sd0 = get_sd(torch.load(THETA0, map_location='cpu'))
    lm_keys0 = {k for k in sd0 if k.startswith(SWAP_PREFIX)}
    print(f'  θ0 の {SWAP_PREFIX}* パラメータ数: {len(lm_keys0)}')

    for d, rel in THETA1_U.items():
        path_u = os.path.join(REPO, rel)
        print(f'\n[domain={d}]')
        print(f'  [load] θ1_d^U: {rel}')
        ckpt_u = torch.load(path_u, map_location='cpu')
        sd_u = get_sd(ckpt_u)

        lm_keys_u = {k for k in sd_u if k.startswith(SWAP_PREFIX)}
        # --- サニティ1: unfrozen の language_model キーは全て θ0 に存在（置換可能） ---
        missing = lm_keys_u - lm_keys0
        assert not missing, f'{d}: θ0 に無い language_model キー {sorted(missing)[:3]}'
        only0 = lm_keys0 - lm_keys_u  # θ0 のみ（position_ids 等の定数バッファ想定）
        print(f'  unfrozen の {SWAP_PREFIX}* = {len(lm_keys_u)} / θ0 のみ存在 = {sorted(only0)}')

        # --- swap: language_model.*（共通キー）のみ θ0 に置換、他は据え置き ---
        new_sd = copy.deepcopy(sd_u)
        replaced = 0
        for k in lm_keys_u:
            assert new_sd[k].shape == sd0[k].shape, f'{d}: shape不一致 {k}'
            new_sd[k] = sd0[k].clone()
            replaced += 1

        # --- サニティ2: language_model.* は θ0 と一致、backbone(Swin)/非対象は θ1_d^U と一致 ---
        for k in lm_keys_u:
            assert torch.equal(new_sd[k], sd0[k]), f'{d}: language_model {k} が θ0 と不一致'
        sample_bb = next(k for k in sd_u if k.startswith('backbone.'))
        assert torch.equal(new_sd[sample_bb], sd_u[sample_bb]), \
            f'{d}: Swin {sample_bb} が θ1_d^U と不一致（Swinは据え置きのはず）'
        sample_other = next(k for k in sd_u
                            if not k.startswith(SWAP_PREFIX) and not k.startswith('backbone.'))
        assert torch.equal(new_sd[sample_other], sd_u[sample_other]), \
            f'{d}: 非対象 {sample_other} が θ1_d^U と不一致'
        print(f'  [swap] {SWAP_PREFIX}* を θ0 に置換: {replaced} params '
              f'(Swin・neck/enc/dec/head は unfrozen のまま)')

        # --- 保存 ---
        out_ckpt = copy.deepcopy(ckpt_u)
        if isinstance(out_ckpt, dict) and 'state_dict' in out_ckpt:
            out_ckpt['state_dict'] = new_sd
            out_ckpt.pop('optimizer', None)
            out_ckpt.pop('ema_state_dict', None)
        else:
            out_ckpt = new_sd
        out_path = os.path.join(OUT_DIR, f'{d}_bert_theta0.pth')
        torch.save(out_ckpt, out_path)
        print(f'  [save] {out_path}')

    print('\n[done] 全ドメインの BERT ハイブリッド生成完了。')


if __name__ == '__main__':
    main()
