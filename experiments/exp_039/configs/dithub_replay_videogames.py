# =============================================================================
# exp_039: DitHub / リプレイ有り / videogames
#
# 【自動生成】experiments/exp_039/gen_configs.py。直接編集しない。
#
# データ・スケジュール・optimizer はリプレイ・蒸留と同一にするため、継承元を
# ../../exp_023/configs/fullft_replay_videogames.py に取る（design.md §2.1）。
#   20 epoch / MultiStepLR milestones=[15] / AdamW lr 1e-4 wd 1e-4 /
#   grad clip 0.1(L2) / batch 6 / dn 有効 / num_classes 256 / seed 0 /
#   ODVGDataset + RandomSamplingNegPos
# 本 config が上書きするのは**手法固有の設定だけ**（design.md §2.2）。
#
# 逐次学習では前ドメインの融合済み ckpt を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '../../exp_023/configs/fullft_replay_videogames.py'

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
# type はリプレイの有無で変える（exp_023 と同一。2026-08-13 修正）:
#   replayfree : DitHubODVGGroundingDINO
#       ODVG のキャプション文字列に対応した loss を持つ版。
#   replay     : DitHubReplayGroundingDINO
#       上記に加え DitHubLinear を DitHubReplayLinear へ差し替える。specialization 中は
#       画像ごとに GT クラスの A を引くが、リプレイのバッチには参照(Objects365)や過去
#       ドメインの画像が入り、現ドメインの dithub_classes に無いクラスを持つため、素の
#       _delta_per_sample は
#         AttributeError: 'ParameterDict' object has no attribute 'class_leather_shoes'
#       で落ちる（2026-08-13 にクラスタで発生）。差し替え版はライブラリ非登録キーに
#       warmup_lora_a（層に 1 本のクラス非依存 A）＋共有 B を通す（2026-07-22 の裁定）。
#       パラメータを増やさないメソッド専用サブクラスなので state_dict は同一。
#
# encoder=dict(num_cp=0) は必須。事前学習 config の既定 num_cp=6 は fairscale の
# checkpoint_wrapper（再入版）を encoder に掛けるが、これは backward が二重に走り
# DDP の ready マークが二度立って
#   RuntimeError: Expected to mark a variable ready only once
# で落ちる（2026-08-12 にクラスタで発生）。DitHubGroundingDINO は非再入版
# （use_reentrant=False）を encoder_cp で自前に掛けるので、fairscale 側は切る。
model = dict(
    type='DitHubReplayGroundingDINO',
    encoder=dict(num_cp=0),
    encoder_cp=6,
    dithub_classes=('avatar', 'object', 'assassin', 'atv', 'car', 'gun', 'gun menu',
        'healthbar', 'horse', 'hud', 'map', 'person', 'surroundings',
        'CT', 'T', 'Character', 'enemy', 'enemy-head', 'friendly',
        'friendly-head', 'Akali', 'Blitzcrank', 'Braum', 'Caitlyn',
        'Camille', 'Cho-Gath', 'Darius', 'Dr- Mundo', 'Ekko', 'Ezreal',
        'Fiora', 'Galio', 'Gankplank', 'Garen', 'Graves', 'Heimerdinger',
        'Illaoi', 'Janna', 'Jayce', 'Jhin', 'Jinx', 'Kai-Sa', 'Kassadin',
        'Katarina', 'Kog-Maw', 'Leona', 'Lissandra', 'Lulu', 'Lux',
        'Malzahar', 'Miss Fortune', 'Orianna', 'Poppy', 'Quinn',
        'Samira', 'Seraphine', 'Shaco', 'Singed', 'Sion', 'Swain',
        'Tahm Kench', 'Talon', 'Taric', 'Tristana', 'Trundle',
        'Twisted Fate', 'Twitch', 'Urgot', 'Veigar', 'Vex', 'Vi',
        'Viktor', 'Warwick', 'Yone', 'Yuumi', 'Zac', 'Ziggs', 'Zilean',
        'Zyra', 'armor', 'base', 'rune', 'rune-blue', 'rune-gray',
        'rune-grey', 'rune-red', 'watcher'))

# 手法固有: warmup -> specialization の切替は学習量の半分 = epoch 10
# （20 epoch の半分。design.md §2.2）。式3 の対象は trained_classes。
custom_hooks = [
    dict(
        type='DitHubSeqPhaseHook',
        warmup_epochs=10,
        trained_classes=['person', 'car'])
]

# specialization 中は未選択クラスの A が不使用になるため必須
find_unused_parameters = True
