# =============================================================================
# exp_049: DGS stage2（既存グループ・IGC 制約付き学習）/ ODinW-13 IVLOD
#
# 公式 projects/DGS/configs/IVLOD/dgs_stage2.py に対応。stage1 との差分は
#   - 検出器 GroundingDINO_DGS / head GroundingDINOHead_inc_DGS（擬似ラベル＋蒸留）
#   - dn 無効（dn_cfg=None。移植では上流生成器を凍結・不使用にして実現）
#   - lr 5e-4
#   - dataset に distn_cfg（グループクラス空間・ori_text の供給）
# seen_tasks / task_id_mapping_path 等はドライバが --cfg-options で上書きする。
# =============================================================================
_base_ = './dgs_stage1.py'

distn_cfg = dict(
    type='distillation',
    future_class=False,
    task_id_mapping_path='experiments/exp_049/work_dirs/task_id_mapping.yaml',
    label_distn=dict(
        type='threshold_pseudo',
        mode='hardlabel',
        sigma=0.4, label_iou_th=0.7,
    ),
    feat_distn=dict(
        # 公開 config は 'inter-intra' だが、実装に存在する分岐は 'inter-class' のみで
        # そのままでは KD（トポロジー蒸留）が不活性になる。Appendix B.4-B.5 が
        # 全実験で KD 有効（γ1=3, γ2=5 = 下の loss_weight）と明記しているため、
        # リリース時の名称ズレと判断して 'inter-class' に訂正（2026-08-21 ユーザー決定・案A）。
        type='inter-class',
        subtype='opt1',
        img_loss=dict(type='L2Loss', loss_weight=3.0, reduction='mean'),
        text_loss=dict(type='L2Loss', loss_weight=5.0, reduction='mean'),
    ),
    query_distn=dict(
        type='seperate_queryinit',
        num_matching_query=900,
        num_aux_query=900,
    ),
)

model = dict(
    type='GroundingDINO_DGS',
    dn_cfg=None,
    distn_cfg=distn_cfg,
    bbox_head=dict(
        type='GroundingDINOHead_inc_DGS',
        distn_cfg=distn_cfg),
)

train_dataloader = dict(
    dataset=dict(
        distn_cfg=distn_cfg,
        seen_tasks='AerialMaritimeDrone'))  # ドライバが上書き

optim_wrapper = dict(optimizer=dict(lr=0.0005))
