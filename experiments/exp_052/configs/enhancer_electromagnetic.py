# =============================================================================
# exp_052: 蒸留E × 学習可能モジュール = Feature Enhancer（encoder） / electromagnetic（逐次 t=2）
#
# 【自動生成】experiments/exp_052/gen_configs.py。直接編集しない。
#
# ベース = exp_035 kdE（旧バッチ [4,1,1]×6・λ=10・バッファ由来 2 枚限定）。
# 上書きは paramwise の凍結（lr_mult 0.0）と ckpt 保存方針だけ（design.md §2.2）。
# 教師 θ_{t-1}（t=1 は θ0 の実パス）はドライバが model.teacher_ckpt で渡す。
# =============================================================================
_base_ = '../../exp_035/configs/kdE_condA_l2w100_electromagnetic.py'

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.0),  # 凍結
            language_model=dict(lr_mult=0.0),  # 凍結
            neck=dict(lr_mult=0.0),  # 凍結
            text_feat_map=dict(lr_mult=0.0),  # 凍結
            encoder=dict(lr_mult=1.0),  # 学習
            decoder=dict(lr_mult=0.0),  # 凍結
            bbox_head=dict(lr_mult=0.0),  # 凍結
            memory_trans_fc=dict(lr_mult=0.0),  # 凍結
            memory_trans_norm=dict(lr_mult=0.0),  # 凍結
            query_embedding=dict(lr_mult=0.0),  # 凍結
            level_embed=dict(lr_mult=0.0),  # 凍結
            dn_query_generator=dict(lr_mult=0.0),  # 凍結
        )))

# ckpt は last のみ・optimizer 状態なし（design.md §5）
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=-1,
                    save_optimizer=False, save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=50))
