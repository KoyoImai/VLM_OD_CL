#!/usr/bin/env python3
"""exp_040: 後半3ドメイン（aerial -> microscopic -> documents）の config を生成する。

design.md §2 のとおり、学習設定は前半（exp_024 / exp_027 / exp_035）と完全に同一にする。
そのため base はそれぞれの共通ベースを継承し、本 config はデータだけを定義する。

    条件1 replayfree : experiments/exp_024/configs/fullft_replayfree_base.py
                       （batch 4、DefaultSampler、現在ドメインのみ）
    条件2 replay     : experiments/exp_023/configs/fullft_replay_base.py
                       （batch 6、CurrentEpochMultiSourceSampler [4,1,1]、3ソース）
    条件3 kdE        : 条件2 に KDGroundingDINO + 蒸留設定（λ=10）を重ねる

3ソース構成（design.md §2.4、2026-08-14 確定）:
    ソース0 現在ドメイン        4 枚/バッチ
    ソース1 参照バッファ        Objects365 1,000 枚（全タスク固定）  1 枚/バッチ
    ソース2 過去プール          学習済みドメイン各 500 枚を ConcatDataset で1ソースに  1 枚/バッチ
過去プールは累積する。Objects365 は過去プールに統合しない。

生成するもの（9 本）:
    configs/{replayfree,replay,kdE}_{aerial,microscopic,documents}.py

使い方: python experiments/exp_040/gen_configs.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, 'configs')

RF100 = '/workspace/kouyou/datasets/rf100_domain/'
O365 = '/workspace/kouyou/datasets/o365v1_stage/Objects365_v1/2019-08-02/'
BUF = '/workspace/kouyou/mmdetection/experiments/exp_023/buffer/'

# 6ドメインの逐次順。前半3つは exp_024 / exp_027 / exp_035 で学習済み。
FULL_ORDER = ['underwater', 'electromagnetic', 'videogames',
              'aerial', 'microscopic', 'documents']
LATER = FULL_ORDER[3:]          # 本実験で学習する後半3ドメイン

STATS = {'aerial': (6643, 35840, 22),
         'microscopic': (9576, 74208, 28),
         'documents': (17866, 126166, 59)}

HEADER = '''\
# =============================================================================
# exp_040: {cond_ja} / {domain}（逐次 t={t}）
#
# 【自動生成】experiments/exp_040/gen_configs.py。直接編集しない。
#
# 6ドメイン逐次 {order} の {t} 番目。
# 重みは前ドメインの last をドライバが load_from で渡す（t=4 は前半3タスク目の
# epoch_20.pth。design.md §2.1）。
#   画像 {n_img} 枚 / box {n_box} / クラス {n_cls}
#
# 学習設定は前半（exp_024 / exp_027 / exp_035）と完全に同一で、本 config が
# 定義するのはデータだけである（design.md §2.3）。
# =============================================================================
_base_ = '{base}'

train_pipeline = _base_.train_pipeline

_root = '{rf100}{domain}/'

# 現在ドメイン（OD-ODVG）
current_{domain} = dict(
    type='ODVGDataset',
    data_root=_root,
    ann_file='{domain}_train_od.json',
    label_map_file='{domain}_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)
'''

REPLAYFREE_TAIL = '''
train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=current_{domain})
'''

REPLAY_TAIL = '''
# 参照バッファ（Objects365v1 1,000。全タスク共通・固定。過去プールには統合しない）
reference_buffer = dict(
    type='ODVGDataset',
    data_root='{o365}',
    ann_file='{buf}reference_o365v1_1000.odvg.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)

# 過去プール（学習済み {n_past} ドメイン × 500 を入れ子 ConcatDataset で1 source に）
#   {past_list}
past_pool = dict(
    type='ConcatDataset',
    datasets=[
{past_entries}    ])

# 混合: [現在, 参照, 過去プール] の3ソース、source_ratio [4,1,1]、batch_size=6
train_dataloader = dict(
    _delete_=True,
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='CurrentEpochMultiSourceSampler', batch_size=6, source_ratio=[4, 1, 1]),
    dataset=dict(
        type='ConcatDataset',
        # 過去プールが入れ子 ConcatDataset で metainfo に cumulative_sizes を持つため、
        # トップの整合チェックで無視する（mmdet ConcatDataset の既存パラメータ）。
        ignore_keys=['cumulative_sizes'],
        datasets=[current_{domain}, reference_buffer, past_pool]))
'''

KD_TAIL = '''
# 手法固有: 蒸留E（画像・テキスト・融合の3特徴、L2）、λ=10.0（exp_035 と同一）
custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.datasets.samplers.current_epoch_multi_source_sampler',
        'mmdet.models.losses.feature_distill_loss',
        'mmdet.models.detectors.kd_grounding_dino',
    ],
    allow_failed_imports=False)

model = dict(
    type='KDGroundingDINO',
    # 教師 = θ_{{t-1}}。ドライバが --cfg-options で実パスに差し替える。
    teacher_ckpt=None,
    kd=dict(
        targets=['img', 'txt', 'fus'],
        loss=dict(type='FeatureDistillLoss', form='l2'),
        loss_weight=10.0,
        num_buffer_per_batch=2))
'''

PAST_ENTRY = '''\
        dict(
            type='ODVGDataset',
            data_root='{rf100}{d}/',
            ann_file='{buf}{d}_500.odvg.json',
            label_map_file='{d}_label_map.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=False),
            return_classes=True,
            pipeline=train_pipeline),
'''

BASE = {
    'replayfree': '../../exp_024/configs/fullft_replayfree_base.py',
    'replay': '../../exp_023/configs/fullft_replay_base.py',
    'kdE': '../../exp_023/configs/fullft_replay_base.py',
}
COND_JA = {
    'replayfree': 'リプレイ無し逐次FT（条件A＝全モジュール）',
    'replay': 'リプレイ有り逐次FT（条件A＝全モジュール）',
    'kdE': '蒸留E（3項・L2・λ=10）＋ リプレイ（条件A）',
}


def gen():
    os.makedirs(CFG_DIR, exist_ok=True)
    for cond in ('replayfree', 'replay', 'kdE'):
        for i, d in enumerate(LATER):
            t = 4 + i
            past = FULL_ORDER[:t - 1]          # 学習済みドメイン（累積）
            n_img, n_box, n_cls = STATS[d]
            body = HEADER.format(
                cond_ja=COND_JA[cond], domain=d, t=t,
                order=' -> '.join(FULL_ORDER), base=BASE[cond], rf100=RF100,
                n_img=f'{n_img:,}', n_box=f'{n_box:,}', n_cls=n_cls)
            if cond == 'replayfree':
                body += REPLAYFREE_TAIL.format(domain=d)
            else:
                entries = ''.join(
                    PAST_ENTRY.format(rf100=RF100, buf=BUF, d=p) for p in past)
                body += REPLAY_TAIL.format(
                    domain=d, o365=O365, buf=BUF, n_past=len(past),
                    past_list=' + '.join(f'{p} 500' for p in past),
                    past_entries=entries)
                if cond == 'kdE':
                    body += KD_TAIL
            path = os.path.join(CFG_DIR, f'{cond}_{d}.py')
            with open(path, 'w') as f:
                f.write(body)
            extra = '' if cond == 'replayfree' else f'  過去プール={len(past)} ドメイン'
            print(f'generated {path}  (t={t}){extra}')


if __name__ == '__main__':
    gen()
