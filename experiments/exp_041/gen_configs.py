#!/usr/bin/env python3
"""exp_041: ZiRa / DitHub × リプレイ有無の後半3ドメイン config を生成する。

design.md §2 のとおり、データ・スケジュール・optimizer は exp_040（リプレイフリー・
リプレイ・蒸留E）と同一にする。そのため継承元を exp_040 のドメイン config に取り、
手法（model / imports / paramwise / hooks）だけを上書きする（exp_039 が exp_023/exp_024 の
ドメイン config を継承したのと同じ方式）。

    リプレイ有り: experiments/exp_040/configs/replay_<domain>.py
                 （batch 6、CurrentEpochMultiSourceSampler [4,1,1]、3ソース累積過去プール）
    リプレイ無し: experiments/exp_040/configs/replayfree_<domain>.py
                 （batch 4、DefaultSampler、現在ドメインのみ）

生成するもの（22 本）:
    学習 12 本 : configs/{zira,dithub}_{replay,replayfree}_{aerial,microscopic,documents}.py
    評価 10 本 : configs/zira_eval_{aerial,microscopic,documents}.py
                 configs/dithub_eval_{6ドメイン,zcoco}.py
                   … クラススロットを 6 ドメイン和集合（260 クラス）にした版。
                   既存の exp_023 dithub_eval_*（152 クラス）は前半3ドメインの和集合しか
                   持たず、t>=4 のライブラリの後半クラス A を捨ててしまう（design.md §4）。

使い方: python experiments/exp_041/gen_configs.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, 'configs')

RF100 = '/workspace/kouyou/datasets/rf100_domain/'

# 6ドメインの逐次順。前半3つは exp_039 で学習済み、後半3つを本実験で学習する。
FULL_ORDER = ['underwater', 'electromagnetic', 'videogames',
              'aerial', 'microscopic', 'documents']
LATER = FULL_ORDER[3:]

# 式3 の対象（現ドメインのクラスのうち過去ドメインで学習済みのもの）。
# DitHub はクラスの同一性を canonical_key（小文字化・非英数字→'_'。
# mmdet/models/layers/dithub_layers.py）で判定するため、突き合わせも canonical キーで
# 行う（2026-08-15 実測。生文字列比較だと aerial の 'Fish' を取りこぼす）。
#   aerial   : 'Fish' が underwater の 'fish' と同一キー（class_fish）
#   documents: 'object' が videogames の 'object' と同一キー
# なお aerial は 'orange-sphero' と 'orange_sphero' が同一キーに衝突するため、
# 22 クラスに対しモジュールは 21 個になる（DitHub 実装の仕様どおり）。
TRAINED_CLASSES = {
    'aerial': ['Fish'],            # underwater で学習済み（canonical キー一致）
    'microscopic': [],
    'documents': ['object'],       # videogames で学習済み
}

HEADER = '''\
# =============================================================================
# exp_041: {method_ja} / {replay_ja} / {domain}（逐次 t={t}）
#
# 【自動生成】experiments/exp_041/gen_configs.py。直接編集しない。
#
# データ・スケジュール・optimizer は exp_040 と同一にするため、継承元を
# {base} に取る（design.md §2.2）。
#   20 epoch / MultiStepLR milestones=[15] / AdamW lr 1e-4 wd 1e-4 /
#   grad clip 0.1(L2) / batch {batch} / dn 有効 / num_classes 256 / seed 0
# 本 config が上書きするのは**手法固有の設定だけ**（design.md §2.3）。
#
# 逐次学習では前タスクの融合済み ckpt を load_from に上書きする（ドライバが実施。
# t=4 は exp_039 の merged_after_t3_videogames.pth。design.md §2.1）。
# =============================================================================
_base_ = '{base}'
'''

# 手法ボディは exp_039 と同一（experiments/exp_039/gen_configs.py）。
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
    type='{detector}',
    encoder=dict(num_cp=0),
    encoder_cp=6,
    dithub_classes={classes})

# 手法固有: warmup -> specialization の切替は学習量の半分 = epoch 10。
# 式3 の対象は trained_classes（過去ドメインで学習済みのクラス）。
custom_hooks = [
    dict(
        type='DitHubSeqPhaseHook',
        warmup_epochs=10,
        trained_classes={trained})
]

# specialization 中は未選択クラスの A が不使用になるため必須
find_unused_parameters = True
'''

ZIRA_EVAL = '''\
# =============================================================================
# exp_041 ZiRa 評価用 / {domain} valid
#
# 【自動生成】experiments/exp_041/gen_configs.py。直接編集しない。
#
#   ZiRa は RDB を全入力に一律適用（クラス選択なし）＋ Rep+ で LLRB に過去累積するため、
#   単一の ZiRa モデル（全和 forward = base + s*HLRB + LLRB）で現在・過去ドメインを評価できる。
#   データ・num_classes は exp_026 の eval_{domain}.py を継承。既存コード無変更。
#   （前半3ドメインと ZCOCO は exp_023 の zira_eval_*.py をそのまま使う。design.md §4）
# =============================================================================
_base_ = '{base}'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.layers.zira_layers',
        'mmdet.models.necks.zira_channel_mapper',
        'mmdet.models.detectors.zira_grounding_dino',
    ],
    allow_failed_imports=False)

model = dict(
    type='ZiRaGroundingDINO',
    zil_loss_weight=0.1,
    neck=dict(type='ZiRaChannelMapper'))
'''

DITHUB_EVAL = '''\
# =============================================================================
# exp_041 DitHub 評価用（6ドメイン逐次ライブラリ対応）/ {target}
#
# 【自動生成】experiments/exp_041/gen_configs.py。直接編集しない。
#
#   逐次学習で育てたライブラリ ckpt（per_class_lora_A）を漏れなくロードするため、
#   dithub_classes を**6ドメイン全クラスの和集合（{n} クラス）**にしてスロットを全確保する。
#   既存の exp_023 dithub_eval_*（152 クラス）は前半3ドメインの和集合しか持たず、
#   t>=4 のライブラリの後半クラス A を捨ててしまう（design.md §4）。
#   評価時は DitHub のフィルタにより、プロンプト中のクラスのうちモジュールを持つもの
#   だけに A が当たるため、和集合の拡大は評価値を変えない（check_exp041_setup.py の
#   項目7で実測確認）。データ・num_classes は {base_ja} を継承。既存コード無変更。
# =============================================================================
_base_ = '{base}'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.layers.dithub_layers',
        'mmdet.models.detectors.dithub_grounding_dino',
    ],
    allow_failed_imports=False)

# 6ドメイン和集合（{n} クラス）。評価時 checkpointing は不要（encoder_cp=0）。
_dithub_all_classes = {classes}

model = dict(
    type='DitHubGroundingDINO',
    dithub_classes=_dithub_all_classes,
    encoder=dict(num_cp=0),
    encoder_cp=0)
'''

BASE = {
    'replay': '../../exp_040/configs/replay_{d}.py',
    'replayfree': '../../exp_040/configs/replayfree_{d}.py',
}

# 評価 config のデータ部の継承元
EVAL_BASE = {
    'underwater': '../../exp_023/configs/eval_underwater.py',
    'electromagnetic': '../../exp_023/configs/eval_electromagnetic.py',
    'videogames': '../../exp_023/configs/eval_videogames.py',
    'aerial': '../../exp_026/configs/eval_aerial.py',
    'microscopic': '../../exp_026/configs/eval_microscopic.py',
    'documents': '../../exp_026/configs/eval_documents.py',
    'zcoco': '../../../configs/mm_grounding_dino/eval_base_coco.py',
}


def label_map_classes(domain):
    """ラベルマップの index 順のクラス名（exp_023/exp_039 の dithub_classes と同じ規約）。"""
    with open(f'{RF100}{domain}/{domain}_label_map.json') as f:
        m = json.load(f)
    return [m[k] for k in sorted(m, key=int)]


def union_classes():
    """6ドメイン和集合。ドメイン順 × ラベルマップ順で重複は初出を残す。"""
    seen, out = set(), []
    for d in FULL_ORDER:
        for c in label_map_classes(d):
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


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


def gen():
    os.makedirs(CFG_DIR, exist_ok=True)

    # --- 学習 config 12 本 ---------------------------------------------------
    for method in ('zira', 'dithub'):
        for replay in ('replay', 'replayfree'):
            for i, d in enumerate(LATER):
                t = 4 + i
                base = BASE[replay].format(d=d)
                head = HEADER.format(
                    method_ja='ZiRa' if method == 'zira' else 'DitHub',
                    replay_ja='リプレイ有り' if replay == 'replay' else 'リプレイ無し',
                    domain=d, t=t, base=base,
                    batch=6 if replay == 'replay' else 4)
                if method == 'zira':
                    body = ZIRA_BODY
                else:
                    cls = label_map_classes(d)
                    body = DITHUB_BODY.format(
                        classes=fmt_classes(cls),
                        trained=repr(TRAINED_CLASSES[d]),
                        detector=('DitHubReplayGroundingDINO'
                                  if replay == 'replay'
                                  else 'DitHubODVGGroundingDINO'))
                path = os.path.join(CFG_DIR, f'{method}_{replay}_{d}.py')
                with open(path, 'w') as f:
                    f.write(head + body)
                extra = ''
                if method == 'dithub':
                    extra = (f'  ({len(label_map_classes(d))} クラス, '
                             f'式3対象={TRAINED_CLASSES[d] or "なし"})')
                print(f'generated {path}  (t={t}){extra}')

    # --- ZiRa 評価 config 3 本（後半ドメインのみ。前半は exp_023 を使う）------
    for d in LATER:
        path = os.path.join(CFG_DIR, f'zira_eval_{d}.py')
        with open(path, 'w') as f:
            f.write(ZIRA_EVAL.format(domain=d, base=EVAL_BASE[d]))
        print(f'generated {path}')

    # --- DitHub 評価 config 7 本（6ドメイン＋zcoco、260 クラス和集合）--------
    uni = union_classes()
    for tgt in FULL_ORDER + ['zcoco']:
        base = EVAL_BASE[tgt]
        base_ja = ('eval_base_coco.py' if tgt == 'zcoco'
                   else ('exp_023' if tgt in FULL_ORDER[:3] else 'exp_026')
                   + f' の eval_{tgt}.py')
        path = os.path.join(CFG_DIR, f'dithub_eval_{tgt}.py')
        with open(path, 'w') as f:
            f.write(DITHUB_EVAL.format(
                target=tgt, base=base, base_ja=base_ja,
                n=len(uni), classes=fmt_classes(uni, indent=8)))
        print(f'generated {path}  ({len(uni)} クラス)')


if __name__ == '__main__':
    gen()
