# =============================================================================
# exp_050: DGS 評価 / microscopic
# 【自動生成】experiments/exp_050/gen_eval_configs.py。直接編集しない。
# task_id / seen_tasks / mapping はドライバが --cfg-options で上書きする。
# =============================================================================
_base_ = '../../exp_026/configs/eval_microscopic.py'

custom_imports = dict(
    imports=['exp023_np_compat', 'projects.dgs_cl'],
    allow_failed_imports=False)

moe_cfg = dict(
    type='moe_adaptive_expand_lora', experts_num=1, top_k=1,
    r=16, alpha=32.0, dropout=0.0,
    group_cfg=dict(type='rep', merge_method='ema', lambda_A=0.2, lambda_B=0.2),
    replace_layer_type=['enc_ffn_img', 'enc_ffn_text'],
    replace_enc_layer_ids=[0, 1, 2, 3, 4, 5], replace_dec_layer_ids=[])

model = dict(
    type='GroundingDINO_DGS_Base',
    num_tasks=6,
    task_id=5,                      # ドライバが上書き
    seen_tasks='underwater,electromagnetic,videogames,aerial,microscopic,documents',        # ドライバが上書き
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

custom_hooks = [
    dict(type='WeightsTransformHook', cfg=[dict(type='moe_lora')]),
    dict(type='DomainPredictorHooK'),
]
