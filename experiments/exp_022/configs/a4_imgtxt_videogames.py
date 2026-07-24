# exp_022 条件A4: 画像+テキスト特徴抽出のみ学習（Swin + neck + BERT + text_feat_map）。
#   それ以外の特徴抽出・融合部および下流は全て凍結(lr_mult=0.0)。
#   学習するモジュールの lr_mult は条件3(unfrozen)と同一
#   (Swin/BERT 0.1=実効1e-5, neck/text_feat_map/encoder/level_embed 1.0=実効1e-4)。
#   level_embed は encoder 入力への加算のため融合側 (exp_019 と同じ裁定)。
# 参照: experiments/exp_022/design.md
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_videogames.py'

randomness = dict(deterministic=False, seed=0)   # [[always-fix-seed-0]] unfrozen と同一

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            # 学習する
            backbone=dict(lr_mult=0.1),
            language_model=dict(lr_mult=0.1),
            neck=dict(lr_mult=1.0),
            text_feat_map=dict(lr_mult=1.0),
            # 凍結する
            encoder=dict(lr_mult=0.0),
            level_embed=dict(lr_mult=0.0),
            decoder=dict(lr_mult=0.0),
            bbox_head=dict(lr_mult=0.0),
            memory_trans_fc=dict(lr_mult=0.0),
            memory_trans_norm=dict(lr_mult=0.0),
            query_embedding=dict(lr_mult=0.0),
            dn_query_generator=dict(lr_mult=0.0),
        )))
