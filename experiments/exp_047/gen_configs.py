#!/usr/bin/env python3
"""exp_047: 素の LoRA の RF100 6 ドメイン（バッファ不使用）config を生成する。

design.md §2 のとおり、データ・スケジュールは exp_045 と同一（replayfree 継承）で、
モデルだけを GroundingDINOLoRA に差し替える。LoRA 設定は exp_045 の InfLoRA と
完全に同一（232 層・r=16・alpha=16 = scaling 1）。差分は「A をランダム学習するか
部分空間設計して固定するか」だけになる。

アダプタの学習率は RF100 枠の paramwise のまま（Swin/BERT 内は lr_mult=0.1 で 1e-5、
encoder・text_feat_map は 1e-4。2026-08-18 決定。design.md §2.2）。

生成するもの（6 本）: configs/lora_{6ドメイン}.py

使い方: python experiments/exp_047/gen_configs.py
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


BODY = '''\
# =============================================================================
# exp_047: 素の LoRA（232 層・r=16・alpha=16）/ {domain}（逐次 t={t}）
#
# 【自動生成】experiments/exp_047/gen_configs.py。直接編集しない。
#
# データ・スケジュール（バッファ不使用・batch 4/GPU・DefaultSampler・ODVG・dn 有効）は
# {base} を継承し、
# 本 config が上書きするのはモデルだけ（design.md §2.1）。LoRA 設定は exp_045 の
# InfLoRA と完全に同一（差分は A の扱いのみ）。
# 逐次学習では前タスクのマージ済み θ_{{t-1}} を load_from に上書きする（ドライバが実施。
# LoRA は毎タスク B=0 から引き直すので開始時点は θ_{{t-1}} と厳密に等価）。
# =============================================================================
_base_ = '{base}'

custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'projects.lora_cl',
    ],
    allow_failed_imports=False)

# 素の LoRA（design.md §2.2）: 挿入 232 層、r=16 / alpha=16（scaling=1）。
# encoder テキスト側 self-attention は MHA を q/k/v/o に分解して挿入する。
# dn は RF100 枠に従い有効のまま（use_dn を渡さない = 既定 True）。
model = dict(
    type='GroundingDINOLoRA',
    lora=dict(
        r=16,
        alpha=16,
        exclude_components=[],
        decompose_mha=[r'^encoder\\.text_layers\\.\\d+\\.self_attn\\.attn$'],
        include=['backbone', 'language_model', 'text_feat_map', 'encoder']))
'''


def gen():
    os.makedirs(CFG_DIR, exist_ok=True)
    for i, d in enumerate(FULL_ORDER):
        path = os.path.join(CFG_DIR, f'lora_{d}.py')
        with open(path, 'w') as f:
            f.write(BODY.format(domain=d, t=i + 1, base=base_for(d)))
        print(f'generated {path}  (t={i + 1})')


if __name__ == '__main__':
    gen()
