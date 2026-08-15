# =============================================================================
# exp_041: DitHub / リプレイ無し / aerial（逐次 t=4）
#
# 【自動生成】experiments/exp_041/gen_configs.py。直接編集しない。
#
# データ・スケジュール・optimizer は exp_040 と同一にするため、継承元を
# ../../exp_040/configs/replayfree_aerial.py に取る（design.md §2.2）。
#   20 epoch / MultiStepLR milestones=[15] / AdamW lr 1e-4 wd 1e-4 /
#   grad clip 0.1(L2) / batch 4 / dn 有効 / num_classes 256 / seed 0
# 本 config が上書きするのは**手法固有の設定だけ**（design.md §2.3）。
#
# 逐次学習では前タスクの融合済み ckpt を load_from に上書きする（ドライバが実施。
# t=4 は exp_039 の merged_after_t3_videogames.pth。design.md §2.1）。
# =============================================================================
_base_ = '../../exp_040/configs/replayfree_aerial.py'

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
# type はリプレイの有無で変える（exp_039 と同一。2026-08-13 修正）:
#   replayfree : DitHubODVGGroundingDINO
#   replay     : DitHubReplayGroundingDINO
#       リプレイのバッチには参照(Objects365)・過去ドメインの画像が入り、現ドメインの
#       dithub_classes に無いクラスを持つため、ライブラリ非登録キーに warmup_lora_a
#       ＋共有 B を通す差し替え版を使う（state_dict は同一）。
#
# encoder=dict(num_cp=0) は必須。fairscale の再入版 checkpointing は DDP と非互換
# （"Expected to mark a variable ready only once"。2026-08-12 にクラスタで発生）。
# DitHub は非再入版（use_reentrant=False）を encoder_cp で自前に掛ける。
model = dict(
    type='DitHubODVGGroundingDINO',
    encoder=dict(num_cp=0),
    encoder_cp=6,
    dithub_classes=('black-hat', 'bodysurface', 'bodyunder', 'umpire', 'white-hat',
        'chain', 'green_sphero', 'orange-sphero', 'orange_sphero',
        'purple_sphero', 'red_sphero', 'yellow_sphero', 'football',
        'player', 'referee', 'crop', 'weed', 'cow', 'Fish', 'Flower',
        'Gravel', 'Sugar'))

# 手法固有: warmup -> specialization の切替は学習量の半分 = epoch 10。
# 式3 の対象は trained_classes（過去ドメインで学習済みのクラス）。
custom_hooks = [
    dict(
        type='DitHubSeqPhaseHook',
        warmup_epochs=10,
        trained_classes=['Fish'])
]

# specialization 中は未選択クラスの A が不使用になるため必須
find_unused_parameters = True
