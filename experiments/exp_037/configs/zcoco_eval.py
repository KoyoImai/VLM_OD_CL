# =============================================================================
# exp_037: ZCOCO 評価（COCO2017-val ゼロショット）/ DitHub モデル
#   COCO の評価プロトコルは eval_base_coco.py（プロジェクト共通・exp_002.5 確定）を継承。
#   モデル設定は exp_037 の学習と同一（dn 無効、ライブラリ 43 スロット）。
#   COCO 80 クラスのうちライブラリにモジュールを持つものだけに ΔW が当たり、
#   他は凍結モデル（θ0）を通る。θ0 の基準値は 0.5040（exp_001 実測）。
#
# 【自動生成】experiments/exp_037/gen_configs.py。直接編集しない。
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/eval_base_coco.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.layers.dithub_layers',
        'mmdet.models.detectors.dithub_grounding_dino',
        'mmdet.models.dense_heads.nodn_grounding_dino_head',
    ],
    allow_failed_imports=False)

model = dict(
    type='DitHubGroundingDINO',
    use_dn=False,
    encoder_cp=0,
    dithub_classes=('aeroplane', 'Ambulance', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'chanterelle', 'Cottontail-Rabbit',
        'CoW', 'Crab', 'diningtable', 'dock', 'dog', 'fish', 'hand',
        'horse', 'jellyfish', 'jetski', 'lift', 'Lobster', 'motorbike',
        'Motorcycle', 'package', 'penguin', 'person', 'pistol', 'pothole',
        'pottedplant', 'puffin', 'raccoon', 'shark', 'sheep', 'Shrimp',
        'sofa', 'starfish', 'stingray', 'train', 'Truck', 'tvmonitor'),
    bbox_head=dict(type='NoDNGroundingDINOHead'))
