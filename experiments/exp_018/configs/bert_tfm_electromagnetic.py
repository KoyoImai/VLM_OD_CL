# exp_018 条件B: Text Backbone(BERT=language_model) + text_feat_map のみ学習。それ以外は全て凍結(lr_mult=0.0)。
#   BERT は unfrozen 準拠の 0.1×(実効1e-5)、text_feat_map は既定1.0(1e-4)。
# 参照: experiments/exp_018/design.md
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_electromagnetic.py'

randomness = dict(deterministic=False, seed=0)   # [[always-fix-seed-0]] unfrozen と同一

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            # base の language_model=0.0 を 0.1(unfrozen準拠) に。text_feat_map は指定なし=既定1.0で学習。
            language_model=dict(lr_mult=0.1),
            # それ以外は全て凍結。
            backbone=dict(lr_mult=0.0),
            neck=dict(lr_mult=0.0),
            encoder=dict(lr_mult=0.0),
            decoder=dict(lr_mult=0.0),
            bbox_head=dict(lr_mult=0.0),
            memory_trans_fc=dict(lr_mult=0.0),
            memory_trans_norm=dict(lr_mult=0.0),
            query_embedding=dict(lr_mult=0.0),
            level_embed=dict(lr_mult=0.0),
            dn_query_generator=dict(lr_mult=0.0),
        )))
