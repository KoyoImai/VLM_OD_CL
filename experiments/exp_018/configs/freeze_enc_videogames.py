# exp_018 条件A: Feature Enhancer(encoder) "だけ" 凍結。それ以外は全て学習。
#   = unfrozen(全学習) から encoder のみ外した構成。lr スキームは unfrozen(exp_010/011) と一致。
#   backbone/language_model = lr_mult=0.1(実効1e-5), 他は既定1.0(1e-4), encoder のみ 0.0。
# 参照: experiments/exp_018/design.md
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_videogames.py'

randomness = dict(deterministic=False, seed=0)   # [[always-fix-seed-0]] unfrozen と同一

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),   # unfrozen と同一
            backbone=dict(lr_mult=0.1),                # base の 0.0 を解凍(unfrozen と同じ 0.1×)
            language_model=dict(lr_mult=0.1),          # 同上
            encoder=dict(lr_mult=0.0),                 # ★ Feature Enhancer だけ凍結
            # neck / text_feat_map / decoder / bbox_head / qsel群 は既定 lr_mult=1.0 で学習
        )))
