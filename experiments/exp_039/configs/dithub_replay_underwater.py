# =============================================================================
# exp_039: DitHub / リプレイ有り / underwater
#
# 【自動生成】experiments/exp_039/gen_configs.py。直接編集しない。
#
# データ・スケジュール・optimizer はリプレイ・蒸留と同一にするため、継承元を
# ../../exp_023/configs/fullft_replay_underwater.py に取る（design.md §2.1）。
#   20 epoch / MultiStepLR milestones=[15] / AdamW lr 1e-4 wd 1e-4 /
#   grad clip 0.1(L2) / batch 6 / dn 有効 / num_classes 256 / seed 0 /
#   ODVGDataset + RandomSamplingNegPos
# 本 config が上書きするのは**手法固有の設定だけ**（design.md §2.2）。
#
# 逐次学習では前ドメインの融合済み ckpt を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '../../exp_023/configs/fullft_replay_underwater.py'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.datasets.samplers.current_epoch_multi_source_sampler',
        'mmdet.models.layers.dithub_layers',
        'mmdet.models.detectors.dithub_grounding_dino',
        'mmdet.models.detectors.dithub_replay_grounding_dino',
        'mmdet.engine.hooks.dithub_phase_hook',
        'mmdet.engine.hooks.dithub_seq_phase_hook',
    ],
    allow_failed_imports=False)

# 手法固有: クラス別 LoRA（r=16 / alpha=8 は検出器の既定）。
# type は ODVG のキャプション文字列に対応した loss を持つ版。
model = dict(
    type='DitHubODVGGroundingDINO',
    encoder_cp=6,
    dithub_classes=('pipe', 'fish', 'jellyfish', 'penguin', 'puffin', 'shark',
        'starfish', 'stingray', 'peix', 'taca', 'echinus', 'holothurian',
        'scallop', 'waterweeds', 'Arborescent', 'Caespitose-a',
        'Caespitose-b', 'Columnar', 'Corymbose', 'Digitate',
        'Encrusting', 'Foliose', 'Massive-Faviidae',
        'Massive-Merulinidae', 'Massive-Mussidae', 'Massive-Poritidae',
        'Solitary', 'Tabular'))

# 手法固有: warmup -> specialization の切替は学習量の半分 = epoch 10
# （20 epoch の半分。design.md §2.2）。式3 の対象は trained_classes。
custom_hooks = [
    dict(
        type='DitHubSeqPhaseHook',
        warmup_epochs=10,
        trained_classes=[])
]

# specialization 中は未選択クラスの A が不使用になるため必須
find_unused_parameters = True
