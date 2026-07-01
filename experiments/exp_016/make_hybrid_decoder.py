#!/usr/bin/env python3
# =============================================================================
# exp_016: Cross-Modality Decoder の内部処理ごとに θ0 に戻したハイブリッドを生成する。
#   - 学習は一切行わない。既存 state_dict を後付けで合成するのみ。
#   - 各処理＋直後の post-norm を同梱（forward: block -> norms[i]）。全6層合計:
#       dec_selfattn       : *.self_attn.*        + *.norms.0.*   (36)  クエリ↔クエリ self-attn
#       dec_crossattn_text : *.cross_attn_text.*  + *.norms.1.*   (36)  クエリ→テキスト cross-attn
#       dec_crossattn_img  : *.cross_attn.*(text除)+ *.norms.2.*  (60)  クエリ→画像 deformable cross-attn
#       dec_ffn            : *.ffn.*              + *.norms.3.*   (36)  FFN
#   - 4グループ = decoder.layers.* 全体（168）。ref_point_head(4)/decoder.norm(2) は unfrozen 据え置き。
#   - 上流(backbone/neck/lm/tfm/encoder)・下流(head/qsel) も unfrozen のまま。
# 出力: experiments/exp_016/hybrids/{domain}_{group}_theta0.pth（24本）
# 参照: experiments/exp_016/design.md
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

# 対象は decoder.layers.* のみ。各処理＋直後の post-norm を同梱するマッチャ。
def _self(k):  return '.self_attn.' in k or '.norms.0.' in k
def _text(k):  return '.cross_attn_text.' in k or '.norms.1.' in k
def _img(k):   return ('.cross_attn.' in k and '.cross_attn_text.' not in k) or '.norms.2.' in k
def _ffn(k):   return '.ffn.' in k or '.norms.3.' in k

GROUPS = {
    'dec_selfattn':       _self,
    'dec_crossattn_text': _text,
    'dec_crossattn_img':  _img,
    'dec_ffn':            _ffn,
}
LAYER_PREFIX = 'decoder.layers.'
OUT_DIR = os.path.join(REPO, 'experiments/exp_016/hybrids')

EXPECT = {'dec_selfattn': 36, 'dec_crossattn_text': 36, 'dec_crossattn_img': 60, 'dec_ffn': 36}


def get_sd(ckpt):
    return ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'[load] θ0: {THETA0}')
    sd0 = get_sd(torch.load(THETA0, map_location='cpu'))

    for d, rel in THETA1_U.items():
        print(f'\n[domain={d}] load θ1_d^U: {rel}')
        ckpt_u = torch.load(os.path.join(REPO, rel), map_location='cpu')
        sd_u = get_sd(ckpt_u)

        for gname, fn in GROUPS.items():
            # 対象 = decoder.layers.* かつ当該処理にマッチするキー
            tgt = [k for k in sd_u if k.startswith(LAYER_PREFIX) and fn(k)]
            assert len(tgt) == EXPECT[gname], f'{d}/{gname}: キー数 {len(tgt)} != 期待 {EXPECT[gname]}'
            missing = [k for k in tgt if k not in sd0]
            assert not missing, f'{d}/{gname}: θ0 に無い対象キー {missing[:3]}'
            badshape = [k for k in tgt if sd_u[k].shape != sd0[k].shape]
            assert not badshape, f'{d}/{gname}: shape不一致 {badshape[:3]}'

            new_sd = copy.deepcopy(sd_u)
            for k in tgt:
                new_sd[k] = sd0[k].clone()

            # サニティ: 対象は θ0 と一致、非対象は θ1_d^U と一致
            for k in tgt:
                assert torch.equal(new_sd[k], sd0[k]), f'{d}/{gname}: 対象 {k} が θ0 と不一致'
            non_tgt = next(k for k in sd_u if not (k.startswith(LAYER_PREFIX) and fn(k)))
            assert torch.equal(new_sd[non_tgt], sd_u[non_tgt]), \
                f'{d}/{gname}: 非対象 {non_tgt} が θ1_d^U と不一致'
            # ref_point_head/final_norm が据え置きか（代表1つ）
            assert torch.equal(new_sd['decoder.norm.weight'], sd_u['decoder.norm.weight']), \
                f'{d}/{gname}: decoder.norm が据え置かれていない'

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

    print('\n[done] 全24ハイブリッド生成完了。')


if __name__ == '__main__':
    main()
