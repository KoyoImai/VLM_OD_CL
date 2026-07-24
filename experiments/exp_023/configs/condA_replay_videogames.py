# =============================================================================
# exp_023 比較手法: 条件A + リプレイ / videogames
#   条件A = 特徴抽出+融合のみ学習（Swin backbone + neck + BERT + text_feat_map +
#   feature enhancer(encoder) + level_embed）、下流（decoder + bbox_head + query/
#   dn/memory_trans）は凍結。定義は experiment_notes/note02.md。
#   データ・混合・スケジュール（20ep/lr1e-4/batch6）は full FT+replay と共通なので
#   fullft_replay_videogames.py を継承し、optim_wrapper の paramwise（lr_mult）だけ差し替える。
#   学習モジュールの lr_mult は full FT と同一（Swin/BERT=0.1、他=1.0）。
#   評価は素の GroundingDINO 用 eval_videogames.py をそのまま使える（custom detector 不要）。
#   既存コード無変更。
# =============================================================================
_base_ = './fullft_replay_videogames.py'

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            # 学習（特徴抽出＋融合）
            backbone=dict(lr_mult=0.1),
            language_model=dict(lr_mult=0.1),
            neck=dict(lr_mult=1.0),
            text_feat_map=dict(lr_mult=1.0),
            encoder=dict(lr_mult=1.0),
            level_embed=dict(lr_mult=1.0),
            # 凍結（下流）
            decoder=dict(lr_mult=0.0),
            bbox_head=dict(lr_mult=0.0),
            memory_trans_fc=dict(lr_mult=0.0),
            memory_trans_norm=dict(lr_mult=0.0),
            query_embedding=dict(lr_mult=0.0),
            dn_query_generator=dict(lr_mult=0.0),
        )))
