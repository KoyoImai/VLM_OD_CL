#!/usr/bin/env python3
"""exp_042: EWC / InfLoRA の ODinW-13 学習 config を生成する。

design.md §2 のとおり、共通枠は exp_034/037/038 の ODinW-13 枠で lr 1e-4 / wd 1e-4。
データ・スケジュール・パイプラインを構造的に一致させるため、継承元を
**exp_038 条件C のタスク config**（lora_odinw13_condC_<task>.py。データ定義と
3000 iter / batch 2 / dn 無効 / num_classes 256 / seed 0 を含む）に取り、
本 config は手法（model）と optimizer（lr / wd）だけを上書きする。

    EWC     : model から lora を除去し EWCGroundingDINO ＋ ewc 設定に差し替え
              （λ と state_path はドライバが --cfg-options で毎回明示する）
    InfLoRA : type を GroundingDINOInfLoRA に変え alpha=16（scaling=1 = 公式の
              ΔW=B·A）と inflora.design_path を追加（design_path はドライバが渡す）

評価は exp_038 の plain 評価 config（odinw13_plain_eval / _eval_after_t* / zcoco_eval）を
そのまま使う（複製しない）。EWC の ckpt は素の構造、InfLoRA はマージ後 plain なので
どちらもそのまま読める。

生成するもの（26 本）: configs/{ewc,inflora}_<task>.py × 13

使い方: python experiments/exp_042/gen_configs.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'exp_034'))
from odinw_official_tasks import task_order  # noqa: E402

CFG_DIR = os.path.join(HERE, 'configs')
SEED = 42
ORDER = task_order(SEED)

HEADER = '''\
# =============================================================================
# exp_042: {method_ja} 学習 config / t={t:02d} {task}
#
# 【自動生成】experiments/exp_042/gen_configs.py。直接編集しない。
#
# データ・スケジュール（3000 iter / batch 2 / dn 無効 / num_classes 256 / seed 0 /
# COCO 形式 pipeline）は exp_038 条件C のタスク config を継承し、本 config が
# 上書きするのは手法（model）と optimizer（lr 1e-4 / wd 1e-4。design.md §2.1）だけ。
#
# 逐次学習では前タスクの θ_{{t-1}} を load_from に上書きする（ドライバが実施）。
# =============================================================================
_base_ = '../../exp_038/configs/lora_odinw13_condC_{task}.py'
'''

EWC_BODY = '''
custom_imports = dict(
    imports=[
        'exp023_np_compat',
        'projects.lora_cl',
        'projects.ewc_cl',
        'mmdet.models.dense_heads.nodn_grounding_dino_head',
    ],
    allow_failed_imports=False)

# 手法: 継承した model から LoRA を除き、EWC 検出器に差し替える（design.md §2.2）。
# 学習対象 = EWC 対象 = 全モジュール（2026-08-15 確定）。
# λ と state_path は**ドライバが --cfg-options model.ewc.lam= / model.ewc.state_path=
# で毎回明示する**（t=1 は state_path=None でペナルティ不活性 = 素のフル FT）。
model = _base_.model
model.pop('lora')
model.update(dict(
    type='EWCGroundingDINO',
    ewc=dict(target_components='all', lam=0.0, state_path=None)))

# lr 1e-4 / wd 1e-4（exp_038 条件C は lr 1e-4 / wd 1e-2。design.md §2.1）
optim_wrapper = dict(optimizer=dict(lr=1e-4, weight_decay=1e-4))
'''

INFLORA_BODY = '''
# 手法: InfLoRA（design.md §2.3）。挿入箇所・r=16 は exp_038 条件A/C と同一の 232 層を
# 継承し、alpha=16（scaling=1 = 公式の ΔW=B·A）に変更。design_path はドライバが
# --cfg-options model.inflora.design_path= で渡す（inflora_prepare.py の出力）。
model = dict(
    type='GroundingDINOInfLoRA',
    lora=dict(alpha=16),
    inflora=dict(design_path=None))

# lr 1e-4 / wd 1e-4（exp_038 条件C は lr 1e-4 / wd 1e-2。design.md §2.1）
optim_wrapper = dict(optimizer=dict(lr=1e-4, weight_decay=1e-4))
'''


def gen():
    os.makedirs(CFG_DIR, exist_ok=True)
    for method, body, ja in (('ewc', EWC_BODY, 'EWC（全モジュール）'),
                             ('inflora', INFLORA_BODY, 'InfLoRA（232 層・r=16）')):
        for i, task in enumerate(ORDER):
            path = os.path.join(CFG_DIR, f'{method}_{task}.py')
            with open(path, 'w') as f:
                f.write(HEADER.format(method_ja=ja, t=i + 1, task=task) + body)
            print(f'generated {path}')


if __name__ == '__main__':
    gen()
