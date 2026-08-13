# =============================================================================
# exp_039: DitHub / リプレイ無し / electromagnetic
#
# 【自動生成】experiments/exp_039/gen_configs.py。直接編集しない。
#
# データ・スケジュール・optimizer はリプレイ・蒸留と同一にするため、継承元を
# ../../exp_024/configs/fullft_replayfree_electromagnetic.py に取る（design.md §2.1）。
#   20 epoch / MultiStepLR milestones=[15] / AdamW lr 1e-4 wd 1e-4 /
#   grad clip 0.1(L2) / batch 4 / dn 有効 / num_classes 256 / seed 0 /
#   ODVGDataset + RandomSamplingNegPos
# 本 config が上書きするのは**手法固有の設定だけ**（design.md §2.2）。
#
# 逐次学習では前ドメインの融合済み ckpt を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '../../exp_024/configs/fullft_replayfree_electromagnetic.py'

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
#
# encoder=dict(num_cp=0) は必須。事前学習 config の既定 num_cp=6 は fairscale の
# checkpoint_wrapper（再入版）を encoder に掛けるが、これは backward が二重に走り
# DDP の ready マークが二度立って
#   RuntimeError: Expected to mark a variable ready only once
# で落ちる（2026-08-12 にクラスタで発生）。DitHubGroundingDINO は非再入版
# （use_reentrant=False）を encoder_cp で自前に掛けるので、fairscale 側は切る。
model = dict(
    type='DitHubODVGGroundingDINO',
    encoder=dict(num_cp=0),
    encoder_cp=6,
    dithub_classes=('dog', 'person', 'Cell', 'Cell-Multi', 'No-Anomaly', 'Shadowing',
        'Unclassified', 'stray', 'target', 'cheetah', 'human',
        'artefact', 'distal phalanges', 'fifth metacarpal bone',
        'first metacarpal bone', 'fourth metacarpal bone',
        'intermediate phalanges', 'proximal phalanges', 'radius',
        'second metacarpal bone', 'soft tissue calcination',
        'third metacarpal bone', 'ulna', 'acl', '0', 'negative',
        'positive', '6W', '7W', 'EH', 'label0', 'label1', 'label2',
        'angle', 'fracture', 'line', 'messed_up_angle', 'bicycle', 'car'))

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
