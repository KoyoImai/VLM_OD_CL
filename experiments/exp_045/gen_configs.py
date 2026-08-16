#!/usr/bin/env python3
"""exp_045: EWC / InfLoRA の RF100 6 ドメイン（バッファ不使用）config を生成する。

design.md §2 のとおり、データ・スケジュールはリプレイフリー系列と同一にする。
そのため継承元を replayfree のドメイン config に取り、モデルだけを差し替える。

    前半3ドメイン: experiments/exp_024/configs/fullft_replayfree_<domain>.py
    後半3ドメイン: experiments/exp_040/configs/replayfree_<domain>.py
    （いずれも現在ドメインのみ・batch 4/GPU・DefaultSampler・ODVG＋RandomSamplingNegPos・
      dn 有効・num_classes 256・seed 0）

    EWC     : EWCGroundingDINO（全モジュール対象。λ と状態はドライバが --cfg-options で
              毎タスク明示。t=1 は状態無し＝ペナルティ不活性）
    InfLoRA : GroundingDINOInfLoRA（232 層・r=16・alpha=16。design_path はドライバが渡す）

生成するもの（12 本）: configs/{ewc,inflora}_{6ドメイン}.py

使い方: python experiments/exp_045/gen_configs.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, 'configs')

FULL_ORDER = ['underwater', 'electromagnetic', 'videogames',
              'aerial', 'microscopic', 'documents']


def base_for(d):
    if d in FULL_ORDER[:3]:
        return f'../../exp_024/configs/fullft_replayfree_{d}.py'
    return f'../../exp_040/configs/replayfree_{d}.py'


HEADER = '''\
# =============================================================================
# exp_045: {method_ja} / {domain}（逐次 t={t}）
#
# 【自動生成】experiments/exp_045/gen_configs.py。直接編集しない。
#
# データ・スケジュール（バッファ不使用・batch 4/GPU・DefaultSampler・ODVG・dn 有効）は
# {base} を継承し、
# 本 config が上書きするのはモデルだけ（design.md §2.1）。
# 逐次学習では前タスクの θ_{{t-1}} を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '{base}'
'''

EWC_BODY = '''
custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'projects.lora_cl',
        'projects.ewc_cl',
    ],
    allow_failed_imports=False)

# EWC（design.md §2.2）: 学習対象 = EWC 対象 = 全モジュール。
# λ（本走 10^3）と EWC 状態はドライバが --cfg-options model.ewc.lam= /
# model.ewc.state_path= で毎タスク明示する（t=1 は状態無し = ペナルティ不活性）。
# dn は RF100 枠に従い有効のまま（use_dn を渡さない = 既定 True）。
model = dict(
    type='EWCGroundingDINO',
    ewc=dict(target_components='all', lam=0.0, state_path=None))
'''

INFLORA_BODY = '''
custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'projects.lora_cl',
    ],
    allow_failed_imports=False)

# InfLoRA（design.md §2.3）: 挿入 232 層（Swin・BERT・feature enhancer・text_feat_map）、
# r=16 / alpha=16（scaling=1 = 公式の ΔW=B·A）。encoder テキスト側 self-attention は
# MHA を q/k/v/o に分解して挿入する。design_path（設計済み A）はドライバが
# --cfg-options model.inflora.design_path= で渡す。dn は RF100 枠に従い有効のまま。
model = dict(
    type='GroundingDINOInfLoRA',
    lora=dict(
        r=16,
        alpha=16,
        exclude_components=[],
        decompose_mha=[r'^encoder\\.text_layers\\.\\d+\\.self_attn\\.attn$'],
        include=['backbone', 'language_model', 'text_feat_map', 'encoder']),
    inflora=dict(design_path=None))
'''

METHODS = {'ewc': ('EWC（全モジュール・λはドライバ指定）', EWC_BODY),
           'inflora': ('InfLoRA（232 層・r=16）', INFLORA_BODY)}


def gen():
    os.makedirs(CFG_DIR, exist_ok=True)
    for m, (ja, body) in METHODS.items():
        for i, d in enumerate(FULL_ORDER):
            path = os.path.join(CFG_DIR, f'{m}_{d}.py')
            with open(path, 'w') as f:
                f.write(HEADER.format(method_ja=ja, domain=d, t=i + 1,
                                      base=base_for(d)) + body)
            print(f'generated {path}  (t={i + 1})')


if __name__ == '__main__':
    gen()
