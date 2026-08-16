#!/usr/bin/env python3
"""exp_043: リプレイと蒸留の別パターン（バッチ構成 [4,2,2]×8）の config を生成する。

design.md §2 のとおり、学習枠は既存 RF100 系列（exp_023 の fullft_replay_base）を継承し、
本 config はデータ（バッチ構成込み）と手法だけを定義する。

バッチ構成（note13。既存系列 [現在4, 参照1, 過去1]×6 との差分の核）:
    t=1 (underwater): ソース [現在, 汎用]、source_ratio [4, 4]、batch 8/GPU
    t>=2            : ソース [現在, 汎用, 過去プール]、source_ratio [4, 2, 2]、batch 8/GPU
    4 GPU で合計 32。現在ドメインは 16 枚/step で従来と同じ（iters/epoch 不変）。
過去プールは学習済みドメインの累積（各 500 枚、exp_023 バッファ = seed 0 ランダム選択）。

kdE 条件は KDGroundingDINO（蒸留E: img/txt/fus、L2、λ=10）を重ねる。
蒸留はバッチ末尾のバッファ由来 4 枚に限定（num_buffer_per_batch=4。design.md §2.4）。
教師 θ_{t-1}（t=1 は θ0）はドライバが --cfg-options model.teacher_ckpt= で渡す。

生成するもの（12 本）: configs/{replay,kdE}_{6ドメイン}.py

使い方: python experiments/exp_043/gen_configs.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, 'configs')

RF100 = '/workspace/kouyou/datasets/rf100_domain/'
O365 = '/workspace/kouyou/datasets/o365v1_stage/Objects365_v1/2019-08-02/'
BUF = '/workspace/kouyou/mmdetection/experiments/exp_023/buffer/'

FULL_ORDER = ['underwater', 'electromagnetic', 'videogames',
              'aerial', 'microscopic', 'documents']

# (画像数, クラス数)。画像数は *_train_od.json の行数（2026-08-16 実測）
STATS = {'underwater': (12633, 28), 'electromagnetic': (25398, 39),
         'videogames': (8233, 87), 'aerial': (6643, 22),
         'microscopic': (9576, 28), 'documents': (17866, 59)}

HEADER = '''\
# =============================================================================
# exp_043: {cond_ja} / {domain}（逐次 t={t}）
#
# 【自動生成】experiments/exp_043/gen_configs.py。直接編集しない。
#
# 6ドメイン逐次 {order} の {t} 番目。
# 学習枠は既存 RF100 系列と同一（20 epoch / lr 1e-4 / wd 1e-4。design.md §2.1）。
# バッチ構成が本実験の核（design.md §2.2）:
#   {batch_ja}
# 重みは前ドメインの last をドライバが load_from で渡す（t=1 は config の θ0）。
#   画像 {n_img} 枚 / クラス {n_cls}
# =============================================================================
_base_ = '../../exp_023/configs/fullft_replay_base.py'

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

# 汎用ドメイン（Objects365v1 1,000。seed 0 ランダム選択・全タスク共通）
reference_buffer = dict(
    type='ODVGDataset',
    data_root='{o365}',
    ann_file='{buf}reference_o365v1_1000.odvg.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True,
    pipeline=train_pipeline)
'''

T1_TAIL = '''
# t=1: [現在, 汎用] の 2 ソース、source_ratio [4, 4]、batch 8/GPU（合計 32）
train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='CurrentEpochMultiSourceSampler', batch_size=8, source_ratio=[4, 4]),
    dataset=dict(
        type='ConcatDataset',
        datasets=[current_{domain}, reference_buffer]))
'''

TN_TAIL = '''
# 過去プール（学習済み {n_past} ドメイン × 500 を入れ子 ConcatDataset で 1 ソースに。
# バッチにはここからランダムに 2 枚入る）
#   {past_list}
past_pool = dict(
    type='ConcatDataset',
    datasets=[
{past_entries}    ])

# t>=2: [現在, 汎用, 過去プール] の 3 ソース、source_ratio [4, 2, 2]、batch 8/GPU（合計 32）
train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='CurrentEpochMultiSourceSampler', batch_size=8, source_ratio=[4, 2, 2]),
    dataset=dict(
        type='ConcatDataset',
        # 過去プールが入れ子 ConcatDataset で metainfo に cumulative_sizes を持つため無視する
        ignore_keys=['cumulative_sizes'],
        datasets=[current_{domain}, reference_buffer, past_pool]))
'''

KD_TAIL = '''
# 手法固有: 蒸留E（画像・テキスト・融合の 3 特徴、L2）、λ=10（exp_035/040 と同一）。
# 蒸留はバッチ末尾のバッファ由来 4 枚（t=1: 汎用4 / t>=2: 汎用2+過去2）に限定する
# （num_buffer_per_batch=4。design.md §2.4。従来系列は 2 枚）。
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
    # 教師 = θ_{{t-1}}（t=1 は θ0）。ドライバが --cfg-options で実パスに差し替える。
    teacher_ckpt=None,
    kd=dict(
        targets=['img', 'txt', 'fus'],
        loss=dict(type='FeatureDistillLoss', form='l2'),
        loss_weight=10.0,
        num_buffer_per_batch=4))
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

COND_JA = {
    'replay': 'リプレイのみ（バッチ [4,2,2]×8）',
    'kdE': '蒸留E（λ=10・バッファ4枚限定）＋ リプレイ（バッチ [4,2,2]×8）',
}


def gen():
    os.makedirs(CFG_DIR, exist_ok=True)
    for cond in ('replay', 'kdE'):
        for i, d in enumerate(FULL_ORDER):
            t = i + 1
            past = FULL_ORDER[:i]
            n_img, n_cls = STATS[d]
            batch_ja = ('t=1: [現在4, 汎用4]（2 ソース）'
                        if t == 1 else
                        f't={t}: [現在4, 汎用2, 過去2]（3 ソース、過去 {len(past)} ドメイン累積）')
            body = HEADER.format(
                cond_ja=COND_JA[cond], domain=d, t=t,
                order=' -> '.join(FULL_ORDER), rf100=RF100, o365=O365,
                buf=BUF, batch_ja=batch_ja, n_img=f'{n_img:,}', n_cls=n_cls)
            if t == 1:
                body += T1_TAIL.format(domain=d)
            else:
                entries = ''.join(
                    PAST_ENTRY.format(rf100=RF100, buf=BUF, d=p) for p in past)
                body += TN_TAIL.format(
                    domain=d, n_past=len(past),
                    past_list=' + '.join(f'{p} 500' for p in past),
                    past_entries=entries)
            if cond == 'kdE':
                body += KD_TAIL
            path = os.path.join(CFG_DIR, f'{cond}_{d}.py')
            with open(path, 'w') as f:
                f.write(body)
            print(f'generated {path}  (t={t})')


if __name__ == '__main__':
    gen()
