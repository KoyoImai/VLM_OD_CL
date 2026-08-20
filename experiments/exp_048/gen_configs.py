#!/usr/bin/env python3
"""exp_048: リプレイ無し蒸留E（kdEonly）／全サンプル蒸留E（kdEall）の config を生成する。

design.md §2 のとおり、モデル以外は既存系列の config を継承し、モデルだけを
KDGroundingDINO（直接蒸留・projector 無し）に差し替える。

    条件1 kdEonly（§2.3）: リプレイ無し。継承元は replayfree
        前半3: experiments/exp_024/configs/fullft_replayfree_<domain>.py
        後半3: experiments/exp_040/configs/replayfree_<domain>.py
      蒸留はバッチ全 4 枚（= 現在ドメイン）。num_buffer_per_batch=4。

    条件2 kdEall（§2.4）: リプレイ [現在4, 汎用1, 過去1]×6（exp_035 と同一）。継承元は
        前半3: experiments/exp_023/configs/fullft_replay_<domain>.py
        後半3: experiments/exp_040/configs/replay_<domain>.py
      蒸留はバッチ全 6 枚（現在4＋バッファ2）。num_buffer_per_batch=6。

教師 θ_{t-1}（t=1 は θ0 の実パス）はドライバが --cfg-options model.teacher_ckpt= で渡す。
`num_buffer_per_batch >= bs` のとき末尾スライスが全体になり、蒸留が全サンプルに掛かる
（kd_proj_grounding_dino.py が使用済みの機構。バッファ位置検査は自動で無効化されるため、
全サンプル適用は check_exp048_setup.py の項目3で直接確認する）。

生成するもの（12 本）: configs/{kdEonly,kdEall}_{6ドメイン}.py

使い方: python experiments/exp_048/gen_configs.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, 'configs')

FULL_ORDER = ['underwater', 'electromagnetic', 'videogames',
              'aerial', 'microscopic', 'documents']


def base_for(cond, d):
    front = d in FULL_ORDER[:3]
    if cond == 'kdEonly':
        return (f'../../exp_024/configs/fullft_replayfree_{d}.py' if front
                else f'../../exp_040/configs/replayfree_{d}.py')
    return (f'../../exp_023/configs/fullft_replay_{d}.py' if front
            else f'../../exp_040/configs/replay_{d}.py')


HEADER = '''\
# =============================================================================
# exp_048: {cond_ja} / {domain}（逐次 t={t}）
#
# 【自動生成】experiments/exp_048/gen_configs.py。直接編集しない。
#
# データ・スケジュールは {base}
# を継承し、本 config が上書きするのはモデル（蒸留E）だけ（design.md §2）。
# 逐次学習では前タスクの epoch_20 を load_from に、教師 θ_{{t-1}}（t=1 は θ0）を
# model.teacher_ckpt に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '{base}'
'''

KDEONLY_BODY = '''
custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.models.losses.feature_distill_loss',
        'mmdet.models.detectors.kd_grounding_dino',
    ],
    allow_failed_imports=False)

# 条件1（design.md §2.3）: リプレイ無し・蒸留のみ。
# バッチは現在ドメイン 4 枚/GPU（DefaultSampler。継承元のまま）。
# num_buffer_per_batch=4 = バッチサイズ → 末尾スライスが全体になり、
# 蒸留E（L2・λ=10・教師 θ_{t-1}）が現在ドメインの全 4 枚に掛かる。
model = dict(
    type='KDGroundingDINO',
    teacher_ckpt=None,
    kd=dict(
        targets=['img', 'txt', 'fus'],
        loss=dict(type='FeatureDistillLoss', form='l2'),
        loss_weight=10.0,
        num_buffer_per_batch=4))
'''

KDEALL_BODY = '''
custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'mmdet.datasets.samplers.current_epoch_multi_source_sampler',
        'mmdet.models.losses.feature_distill_loss',
        'mmdet.models.detectors.kd_grounding_dino',
    ],
    allow_failed_imports=False)

# 条件2（design.md §2.4）: リプレイあり・全サンプル蒸留。
# バッチは [現在4, 汎用1, 過去1]×6/GPU（t=1 は [現在4, 汎用2]。継承元のまま）。
# num_buffer_per_batch=6 = バッチサイズ → 蒸留E（L2・λ=10・教師 θ_{t-1}）が
# 現在ドメイン 4 枚とバッファ 2 枚の全 6 枚に掛かる（exp_035 はバッファ 2 枚のみ）。
model = dict(
    type='KDGroundingDINO',
    teacher_ckpt=None,
    kd=dict(
        targets=['img', 'txt', 'fus'],
        loss=dict(type='FeatureDistillLoss', form='l2'),
        loss_weight=10.0,
        num_buffer_per_batch=6))
'''

COND_JA = {
    'kdEonly': '条件1 kdEonly（リプレイ無し・現在ドメイン全4枚に蒸留E）',
    'kdEall': '条件2 kdEall（リプレイ[4,1,1]×6・全6枚に蒸留E）',
}
BODY = {'kdEonly': KDEONLY_BODY, 'kdEall': KDEALL_BODY}


def gen():
    os.makedirs(CFG_DIR, exist_ok=True)
    for cond in ('kdEonly', 'kdEall'):
        for i, d in enumerate(FULL_ORDER):
            path = os.path.join(CFG_DIR, f'{cond}_{d}.py')
            with open(path, 'w') as f:
                f.write(HEADER.format(cond_ja=COND_JA[cond], domain=d, t=i + 1,
                                      base=base_for(cond, d)))
                f.write(BODY[cond])
            print(f'generated {path}  (t={i + 1})')


if __name__ == '__main__':
    gen()
