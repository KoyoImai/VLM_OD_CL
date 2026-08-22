#!/usr/bin/env python3
"""exp_050: DGS 用の RF100 評価 config を生成する。

既存の per-domain 評価 config（前半3 = exp_023、後半3 = exp_026。valid・
(800,1333)・batch1・num_classes 256・videogames のみ chunked_size=40）を _base_ に
取り、モデルだけ DGS（画像単位ルーティング付き）に差し替える。
seen_tasks / task_id / mapping はドライバが --cfg-options で渡す（空白を含まない）。

ZCOCO は eval_base_coco を _base_ に、ood_th=200（公式 ZCOCO.py どおり）。

使い方: python experiments/exp_050/gen_eval_configs.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, 'configs')

ORDER = ['underwater', 'electromagnetic', 'videogames',
         'aerial', 'microscopic', 'documents']

MODEL_BLOCK = '''
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
    seen_tasks='{all_seen}',        # ドライバが上書き
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
        ood_th={ood_th},
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
'''

HEADER = '''\
# =============================================================================
# exp_050: DGS 評価 / {name}
# 【自動生成】experiments/exp_050/gen_eval_configs.py。直接編集しない。
# task_id / seen_tasks / mapping はドライバが --cfg-options で上書きする。
# =============================================================================
_base_ = '{base}'
'''


def gen():
    os.makedirs(CFG_DIR, exist_ok=True)
    all_seen = ','.join(ORDER)
    for d in ORDER:
        base = (f'../../exp_023/configs/eval_{d}.py' if d in ORDER[:3]
                else f'../../exp_026/configs/eval_{d}.py')
        body = HEADER.format(name=d, base=base) + MODEL_BLOCK.format(
            all_seen=all_seen, ood_th=500)
        path = os.path.join(CFG_DIR, f'eval_dgs_{d}.py')
        open(path, 'w').write(body)
        print('generated', os.path.basename(path))
    body = HEADER.format(
        name='ZCOCO（ood_th=200 は公式 ZCOCO.py どおり）',
        base='../../../configs/mm_grounding_dino/eval_base_coco.py')
    body += MODEL_BLOCK.format(all_seen=all_seen, ood_th=200)
    path = os.path.join(CFG_DIR, 'eval_dgs_zcoco.py')
    open(path, 'w').write(body)
    print('generated', os.path.basename(path))


if __name__ == '__main__':
    gen()
