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
#
# encoder=dict(num_cp=0) は必須（2026-08-19 修正）。既定の num_cp=6 は fairscale の
# **再入型** checkpoint_wrapper を encoder に掛けるが、EWC ペナルティは θ を forward の
# 外で直接使うため勾配が 2 経路になり、再入型 checkpoint の入れ子 backward と DDP の
# 組で「Expected to mark a variable ready only once」で落ちる（t=2 でペナルティが
# 活性化した時点でクラスタで発生。最小再現で cp あり=エラー / cp なし=正常を実証済み）。
# t=1 はペナルティ不活性のため num_cp=6 でも通る。checkpointing の有無は勾配を
# 浮動小数点レベルで変えるため、対照（replayfree/InfLoRA は num_cp=6）との但し書きは
# design.md §6 に記録。
model = dict(
    type='EWCGroundingDINO',
    encoder=dict(num_cp=0),
    ewc=dict(target_components='all', lam=0.0, state_path=None))

# DDP の static_graph=True は必須（2026-08-20 修正）。上記 num_cp=0 は encoder の
# fairscale 再入型 checkpointing を切るだけで、**Swin backbone の with_cp=True**
# （事前学習 config、mmdet/models/backbones/swin.py:375 が use_reentrant 未指定＝再入型）
# は残るため、同じ機序（EWC ペナルティによる 2 経路目の勾配 × 再入型 cp の入れ子
# backward × DDP）で t=2 に
#   Expected to mark a variable ready only once ...
#   backbone.stages.3.blocks.1.ffn.layers.1.bias has been marked as ready twice
# が出る（2026-08-20 にクラスタで発生。本環境 2 GPU で同一パラメータ・同一 index で再現）。
# static_graph=True は再入型 checkpointing を DDP 下で使うための PyTorch 公式の構成で、
# 学習の計算自体（cp あり）は t=1・他条件（replayfree/InfLoRA/LoRA）と同じまま変わらない。
# 対照として backbone.with_cp=False も実測したが、batch 4/GPU で 40GB を使い切り
# iter 13 で OOM（2 回再現）したため採らない。検証は design.md §6。
model_wrapper_cfg = dict(type='MMDistributedDataParallel', static_graph=True)
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
