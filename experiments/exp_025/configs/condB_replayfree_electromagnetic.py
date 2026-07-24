# =============================================================================
# exp_025: リプレイ無し逐次FT（条件B＝特徴抽出+融合）/ electromagnetic
#   条件B = 画像-テキスト特徴抽出部＋特徴融合部のみ学習
#     学習: backbone(Swin) + neck + language_model(BERT) + text_feat_map
#           + encoder(feature enhancer) + level_embed
#     凍結: decoder + bbox_head + memory_trans + query_embedding + dn_query_generator
#   定義は experiment_notes/note04.md（条件A=全モジュール / 条件B=特徴抽出+融合）。
#
#   データ・スケジュール・batch(4)・seed・init_cfg 等は exp_024（条件A）と完全に同一に
#   するため、**exp_024 のドメイン config をそのまま継承**し、optim_wrapper の
#   paramwise（lr_mult）だけを差し替える。これにより両実験の差分が学習範囲だけに閉じる。
#   学習モジュールの lr_mult は条件A と同一値（Swin/BERT=0.1、他=1.0）。
#
#   注: 凍結は requires_grad=False ではなく lr_mult=0.0 で実現する（exp_023 の条件B 実装と
#   同一方式）。勾配自体は計算されるが実効学習率が 0 になるため重みは更新されない。
#   既存ファイルは無変更。
# =============================================================================
_base_ = '../../exp_024/configs/fullft_replayfree_electromagnetic.py'

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
