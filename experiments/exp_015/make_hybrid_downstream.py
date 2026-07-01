#!/usr/bin/env python3
# =============================================================================
# exp_015: 下流（Query Selection＋Decoder＋Head）を θ0 に戻したハイブリッドを生成する。
#   - 学習は一切行わない。既存 state_dict を後付けで合成するのみ。
#   - decoder : decoder.*                Cross-Modality Decoder（174）
#   - cls     : bbox_head.cls_branches.* Contrastive classification head（7）
#   - reg     : bbox_head.reg_branches.* Box regression head（42）
#   - qsel    : memory_trans_fc/norm + level_embed + query_embedding
#               Language-guided Query Selection（推論経路, 6）
#   - whole   : 上記4グループの union（229）
#   - 上流（backbone/neck/language_model/text_feat_map/encoder）は unfrozen のまま。
#
# 除外: dn_query_generator.label_embedding.weight
#   θ0(256,256) vs domain(num_classes,256) で shape 不一致 → 置換不可。かつ denoising は
#   学習時のみ使用・評価では forward に不登場のため θ0/θ1 いずれでも評価結果は不変。
#   → 全グループから除外し θ1_d^U のまま据え置く（position_ids と同じ無害な扱い）。
# 出力: experiments/exp_015/hybrids/{domain}_{group}_theta0.pth（30本）
# 参照: experiments/exp_015/design.md
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

# 各機能ブロックの接頭辞。whole は 4 ブロックの union。
DECODER = ('decoder.',)
CLS     = ('bbox_head.cls_branches.',)
REG     = ('bbox_head.reg_branches.',)
QSEL    = ('memory_trans_fc.', 'memory_trans_norm.', 'level_embed', 'query_embedding.')
GROUPS = {
    'decoder': DECODER,
    'cls':     CLS,
    'reg':     REG,
    'qsel':    QSEL,
    'whole':   DECODER + CLS + REG + QSEL,
}
# 評価不使用・shape不一致で除外するキー（明示）
EXCLUDE = ('dn_query_generator.',)
OUT_DIR = os.path.join(REPO, 'experiments/exp_015/hybrids')


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
            # 置換対象 = 対象接頭辞に該当し、除外対象でないキー
            tgt = [k for k in sd_u if in_group(k, prefixes) and not in_group(k, EXCLUDE)]
            missing = [k for k in tgt if k not in sd0]
            assert not missing, f'{d}/{gname}: θ0 に無い対象キー {missing[:3]}'
            # shape 不一致が残っていないことを保証（EXCLUDE 済みなので 0 のはず）
            badshape = [k for k in tgt if sd_u[k].shape != sd0[k].shape]
            assert not badshape, f'{d}/{gname}: shape不一致が除外後も残存 {badshape}'

            new_sd = copy.deepcopy(sd_u)
            for k in tgt:
                new_sd[k] = sd0[k].clone()

            # サニティ: 対象は θ0 と一致、非対象は θ1_d^U と一致
            for k in tgt:
                assert torch.equal(new_sd[k], sd0[k]), f'{d}/{gname}: 対象 {k} が θ0 と不一致'
            non_tgt = next(k for k in sd_u if not (in_group(k, prefixes) and not in_group(k, EXCLUDE)))
            assert torch.equal(new_sd[non_tgt], sd_u[non_tgt]), \
                f'{d}/{gname}: 非対象 {non_tgt} が θ1_d^U と不一致'
            # label_embedding は据え置き（θ1_d^U のまま）
            le = 'dn_query_generator.label_embedding.weight'
            assert torch.equal(new_sd[le], sd_u[le]), f'{d}/{gname}: label_embedding が据え置かれていない'

            out_ckpt = copy.deepcopy(ckpt_u)
            if isinstance(out_ckpt, dict) and 'state_dict' in out_ckpt:
                out_ckpt['state_dict'] = new_sd
                out_ckpt.pop('optimizer', None)
                out_ckpt.pop('ema_state_dict', None)
            else:
                out_ckpt = new_sd
            out_path = os.path.join(OUT_DIR, f'{d}_{gname}_theta0.pth')
            torch.save(out_ckpt, out_path)
            print(f'  [{gname}] 置換 {len(tgt)} params -> {out_path}')

    print('\n[done] 全30ハイブリッド生成完了。')


if __name__ == '__main__':
    main()
