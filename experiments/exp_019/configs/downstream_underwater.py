# exp_019 条件B: 下流のみ学習（decoder/bbox_head/memory_trans_fc・norm/query_embedding/dn_query_generator）。
#   抽出・融合部（Swin/BERT/neck/text_feat_map/encoder/level_embed）は全て凍結(lr_mult=0.0)。
#   下流は未指定＝既定 lr_mult 1.0（実効 1e-4）で学習。
# 参照: experiments/exp_019/design.md
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py'

randomness = dict(deterministic=False, seed=0)   # [[always-fix-seed-0]] unfrozen と同一

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            # 抽出・融合部を全凍結（backbone/language_model は base 既定 0.0 を明示）
            backbone=dict(lr_mult=0.0),
            language_model=dict(lr_mult=0.0),
            neck=dict(lr_mult=0.0),
            text_feat_map=dict(lr_mult=0.0),
            encoder=dict(lr_mult=0.0),
            level_embed=dict(lr_mult=0.0),
            # decoder / bbox_head / memory_trans_fc / memory_trans_norm /
            # query_embedding / dn_query_generator は未指定＝既定 1.0 で学習
        )))
