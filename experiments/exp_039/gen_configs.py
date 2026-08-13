#!/usr/bin/env python3
"""exp_039: ZiRa / DitHub × リプレイ有無 の config を生成する。

design.md §2 のとおり、**データ・スケジュール・optimizer はリプレイ・蒸留と同一**にする。
そのため継承元を「リプレイ・蒸留側のドメイン config」に取り、手法（model / imports /
paramwise / hooks）だけを上書きする。

    リプレイ有り: experiments/exp_023/configs/fullft_replay_<domain>.py
                 （batch 6、CurrentEpochMultiSourceSampler、20 epoch、lr 1e-4）
    リプレイ無し: experiments/exp_024/configs/fullft_replayfree_<domain>.py
                 （batch 4、DefaultSampler、20 epoch、lr 1e-4）

exp_023 の zira_*/dithub_* を継承しない理由（design.md §6.2 の訂正）:
それらは iteration 基準（`InfiniteSampler` / `MultiSourceSampler` ＋ IterBasedTrainLoop）で
組まれており、epoch 基準に載せ替えるとサンプラごと差し替える必要がある。継承元をリプレイ・
蒸留側にすれば、データ経路が構造的に一致することを保証できる。

生成するもの（12 本）:
    configs/{zira,dithub}_{replay,replayfree}_{underwater,electromagnetic,videogames}.py

使い方: python experiments/exp_039/gen_configs.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, 'configs')

DOMAINS = ['underwater', 'electromagnetic', 'videogames']

# 継承元（リプレイ・蒸留と同一の土台）
BASE = {
    'replay': '../../exp_023/configs/fullft_replay_{d}.py',
    'replayfree': '../../exp_024/configs/fullft_replayfree_{d}.py',
}

# DitHub のクラス別 LoRA 用スロット。exp_023 のドメイン config から流用する
# （label_map の index 順。複製ではなく import で取り出す）。
DITHUB_CLASSES_SRC = '../../exp_023/configs/dithub_{r}_{d}.py'

# 式3 の対象（現ドメインのクラスのうち過去ドメインに出現したもの）。
# 逐次順 underwater -> electromagnetic -> videogames。exp_023 の設定を踏襲。
TRAINED_CLASSES = {
    'underwater': [],
    'electromagnetic': [],
    'videogames': ['person', 'car'],
}

HEADER = '''\
# =============================================================================
# exp_039: {method_ja} / {replay_ja} / {domain}
#
# 【自動生成】experiments/exp_039/gen_configs.py。直接編集しない。
#
# データ・スケジュール・optimizer はリプレイ・蒸留と同一にするため、継承元を
# {base} に取る（design.md §2.1）。
#   20 epoch / MultiStepLR milestones=[15] / AdamW lr 1e-4 wd 1e-4 /
#   grad clip 0.1(L2) / batch {batch} / dn 有効 / num_classes 256 / seed 0 /
#   ODVGDataset + RandomSamplingNegPos
# 本 config が上書きするのは**手法固有の設定だけ**（design.md §2.2）。
#
# 逐次学習では前ドメインの融合済み ckpt を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '{base}'
'''

ZIRA_BODY = '''
custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.datasets.samplers.current_epoch_multi_source_sampler',
        'mmdet.models.layers.zira_layers',
        'mmdet.models.necks.zira_channel_mapper',
        'mmdet.models.detectors.zira_grounding_dino',
    ],
    allow_failed_imports=False)

# 手法固有: ZiL の λ=0.1、neck を RDB 付きに差し替え
model = dict(
    type='ZiRaGroundingDINO',
    zil_loss_weight=0.1,
    neck=dict(type='ZiRaChannelMapper'))

# 手法固有: LLRB のみ lr×η（η=0.2）。他の custom_keys は継承元のものを保つ。
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1),
            language_model=dict(lr_mult=0.1),
            llrb=dict(lr_mult=0.2))))
'''

DITHUB_BODY = '''
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
    dithub_classes={classes})

# 手法固有: warmup -> specialization の切替は学習量の半分 = epoch 10
# （20 epoch の半分。design.md §2.2）。式3 の対象は trained_classes。
custom_hooks = [
    dict(
        type='DitHubSeqPhaseHook',
        warmup_epochs=10,
        trained_classes={trained})
]

# specialization 中は未選択クラスの A が不使用になるため必須
find_unused_parameters = True
'''


def fmt_classes(classes, indent=8):
    body = ', '.join(repr(c) for c in classes)
    if len(body) <= 66:
        return f'({body})'
    out, line = [], ''
    for c in classes:
        item = repr(c) + ', '
        if len(line) + len(item) > 66:
            out.append(line.rstrip())
            line = ''
        line += item
    out.append(line.rstrip().rstrip(','))
    return '(' + ('\n' + ' ' * indent).join(out) + ')'


def dithub_classes(replay, domain):
    """exp_023 のドメイン config から dithub_classes を取り出す（複製しない）。"""
    from mmengine.config import Config
    p = os.path.join(HERE, '..', 'exp_023', 'configs',
                     f'dithub_{replay}_{domain}.py')
    return tuple(Config.fromfile(p).model['dithub_classes'])


def gen():
    os.makedirs(CFG_DIR, exist_ok=True)
    for method in ('zira', 'dithub'):
        for replay in ('replay', 'replayfree'):
            for d in DOMAINS:
                base = BASE[replay].format(d=d)
                head = HEADER.format(
                    method_ja='ZiRa' if method == 'zira' else 'DitHub',
                    replay_ja='リプレイ有り' if replay == 'replay' else 'リプレイ無し',
                    domain=d, base=base,
                    batch=6 if replay == 'replay' else 4)
                if method == 'zira':
                    body = ZIRA_BODY
                else:
                    cls = dithub_classes(replay, d)
                    body = DITHUB_BODY.format(
                        classes=fmt_classes(cls),
                        trained=repr(TRAINED_CLASSES[d]))
                path = os.path.join(CFG_DIR, f'{method}_{replay}_{d}.py')
                with open(path, 'w') as f:
                    f.write(head + body)
                extra = ''
                if method == 'dithub':
                    extra = (f'  ({len(dithub_classes(replay, d))} クラス, '
                             f'式3対象={TRAINED_CLASSES[d] or "なし"})')
                print(f'generated {path}{extra}')


if __name__ == '__main__':
    gen()
