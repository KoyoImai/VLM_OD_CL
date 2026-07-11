# exp_019 条件A: 抽出・融合部のみ学習（Swin/BERT/neck/text_feat_map/encoder/level_embed）。
#   下流（decoder/bbox_head/query選択系/dn_query_generator）は全て凍結(lr_mult=0.0)。
#   backbone/language_model は unfrozen 基準線に準拠し 0.1（実効 1e-5）。
#   level_embed は encoder 入力への加算（deformable_detr.py:197）のため抽出・融合側＝学習。
# 参照: experiments/exp_019/design.md
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_videogames.py'

randomness = dict(deterministic=False, seed=0)   # [[always-fix-seed-0]] unfrozen と同一

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            # base の 0.0 を unfrozen 準拠の 0.1 に（学習する）
            backbone=dict(lr_mult=0.1),
            language_model=dict(lr_mult=0.1),
            # neck / text_feat_map / encoder / level_embed は未指定＝既定 1.0 で学習
            # 下流を全て凍結
            decoder=dict(lr_mult=0.0),
            bbox_head=dict(lr_mult=0.0),
            memory_trans_fc=dict(lr_mult=0.0),
            memory_trans_norm=dict(lr_mult=0.0),
            query_embedding=dict(lr_mult=0.0),
            dn_query_generator=dict(lr_mult=0.0),
        )))
