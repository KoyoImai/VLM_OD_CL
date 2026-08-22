# =============================================================================
# exp_050: DGS stage1 / t=6 documents（RF100 逐次）
# 【自動生成】experiments/exp_050/gen_configs.py。直接編集しない。
# データ・スケジュールは ../../exp_040/configs/replayfree_documents.py を継承し、
# モデルだけ DGS に差し替え（design.md §2）。ドライバが渡すのは load_from のみ。
# =============================================================================
_base_ = '../../exp_040/configs/replayfree_documents.py'

custom_imports = dict(
    imports=['exp023_np_compat', 'projects.dgs_cl'],
    allow_failed_imports=False)

# IGA（公式値。exp_049 と同一）
moe_cfg = dict(
    type='moe_adaptive_expand_lora', experts_num=1, top_k=1,
    r=16, alpha=32.0, dropout=0.0,
    group_cfg=dict(type='rep', merge_method='ema', lambda_A=0.2, lambda_B=0.2),
    replace_layer_type=['enc_ffn_img', 'enc_ffn_text'],
    replace_enc_layer_ids=[0, 1, 2, 3, 4, 5], replace_dec_layer_ids=[])

model = dict(
    type='GroundingDINO_DGS_Base',
    num_tasks=6,
    task_id=5,
    seen_tasks='underwater,electromagnetic,videogames,aerial,microscopic,documents',
    moe_cfg=moe_cfg,
    vis_cfg=dict(type='none', save_path=''),
    frozen_cfg=dict(
        backbone_frozen=True, language_model_frozen=True, neck_frozen=True,
        encoder_frozen=True, decoder_frozen=True, head_frozen=True,
        exclude_keywords=['lora_']),
    domain_predictor_cfg=dict(
        type='svd',
        feat_path='experiments/exp_050/feats/',
        stats_path='experiments/exp_050/stats/',
        task_id_mapping_path='experiments/exp_050/work_dirs/task_id_mapping.yaml',
        multilevel=False,
        expand_th=150,
        ood_th=500,
        min_eig_ratio=1e-3),
    bbox_head=dict(
        type='GroundingDINOHead_inc',
        setting='cur_text',
        trunc_class=[0, 256]),
)
# dn_cfg は継承（RF100 枠 = 有効。DGS stage1 も公式どおり dn 有効）

# ckpt は last のみ・optimizer 状態なし（2026-08-22 決定）
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=-1,
                    save_optimizer=False, save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=50))

custom_hooks = [
    dict(type='WeightsTransformHook',
         cfg=[dict(type='moe_lora'), dict(type='moe_group_init')]),
    dict(type='MergeHook', cfg=dict(type='ema')),
    dict(type='DomainPredictorHooK'),
]
