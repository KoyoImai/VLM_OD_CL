# exp_017: 射影アダプタ(neck + text_feat_map)のみ学習。それ以外は全て凍結(lr_mult=0.0)。
# 参照: experiments/exp_017/design.md
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_microscopic.py'

randomness = dict(seed=0)   # [[always-fix-seed-0]] 標準configは未設定のため明示

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            # backbone/language_model は base で既に 0.0。以降を追加凍結。
            'backbone': dict(lr_mult=0.0),
            'language_model': dict(lr_mult=0.0),
            'encoder': dict(lr_mult=0.0),
            'decoder': dict(lr_mult=0.0),
            'bbox_head': dict(lr_mult=0.0),
            'memory_trans_fc': dict(lr_mult=0.0),
            'memory_trans_norm': dict(lr_mult=0.0),
            'query_embedding': dict(lr_mult=0.0),
            'level_embed': dict(lr_mult=0.0),
            'dn_query_generator': dict(lr_mult=0.0),
            # neck / text_feat_map は指定なし = 既定 lr_mult=1.0 で学習
        }))
