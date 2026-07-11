# exp_020: ZiRa 再現 (microscopic)
# ベースのドメイン config を継承し、モデル type と neck type を ZiRa 版に差し替える。
# 追加モジュールの実体: mmdet/models/{layers/zira_layers,necks/zira_channel_mapper,detectors/zira_grounding_dino}.py
# 設計: experiments/exp_020/implementation_plan.md
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_microscopic.py'

custom_imports = dict(
    imports=[
        'mmdet.models.layers.zira_layers',
        'mmdet.models.necks.zira_channel_mapper',
        'mmdet.models.detectors.zira_grounding_dino',
    ],
    allow_failed_imports=False)

randomness = dict(deterministic=False, seed=0)

model = dict(
    type='ZiRaGroundingDINO',
    zil_loss_weight=0.1,          # λ (公式実装の既定)
    neck=dict(type='ZiRaChannelMapper'))

# η: LLRB のみ低学習率 (lr x 0.2)。本体の凍結は requires_grad=False で行うため
# custom_keys に凍結エントリは不要 (backbone/language_model の既存エントリは無効化
# された param に掛かるだけで無害)。
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={
        'llrb': dict(lr_mult=0.2),
    }))
