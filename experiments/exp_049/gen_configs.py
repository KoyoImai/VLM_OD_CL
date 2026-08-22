#!/usr/bin/env python3
"""exp_049: タスク別の学習 config（stage1/stage2 × 13 タスク）を生成する。

tools/dist_train.sh は引数を非引用で転送するため、空白を含むデータパス
（Aquarium / NorthAmericaMushrooms）を --cfg-options で渡せない。そこで
タスク固有の値（データパス・metainfo・task_id・seen_tasks・mapping パス）を
config に焼き込み、ドライバが渡すのは load_from だけにする。

stage の選択（stage1=新グループ / stage2=既存グループ）は実行時に DTG の結果で
決まるため、全タスクについて両方を生成する（t=1 は stage1 のみ）。

使い方: python experiments/exp_049/gen_configs.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'exp_034'))
from odinw_official_tasks import DATA_ROOT, ODINW13  # noqa: E402

CFG_DIR = os.path.join(HERE, 'configs')
MAPPING = 'experiments/exp_049/work_dirs/task_id_mapping.yaml'

OFFICIAL_ORDER = [
    'AerialMaritimeDrone', 'Aquarium', 'CottontailRabbits', 'EgoHands',
    'NorthAmericaMushroom', 'Packages', 'PascalVOC', 'pistols', 'pothole',
    'Raccoon', 'ShellfishOpenImages', 'thermalDogsAndPeople',
    'VehiclesOpenImages'
]
ALIAS = {'NorthAmericaMushroom': 'NorthAmericaMushrooms'}

TMPL = '''\
# =============================================================================
# exp_049: DGS {stage} / t={t:02d} {task}
# 【自動生成】experiments/exp_049/gen_configs.py。直接編集しない。
# ドライバが渡すのは load_from（θ_{{t-1}}）だけ。
# =============================================================================
_base_ = './dgs_{stage}.py'

model = dict(
    num_tasks=13,
    task_id={task_id},
    seen_tasks='{seen}',
    domain_predictor_cfg=dict(task_id_mapping_path='{mapping}'))

train_dataloader = dict(
    dataset=dict(
        metainfo='{task}',
        data_root='{root}',
        ann_file='{ann}',
        data_prefix=dict(img='{img}'){extra}))
'''

S2_EXTRA = ''',
        seen_tasks='{seen}',
        distn_cfg=dict(task_id_mapping_path='{mapping}')'''


def gen():
    os.makedirs(CFG_DIR, exist_ok=True)
    for i, task in enumerate(OFFICIAL_ORDER):
        t = i + 1
        spec = ODINW13[ALIAS.get(task, task)]
        seen = ','.join(OFFICIAL_ORDER[:t])
        for stage in (['stage1'] if t == 1 else ['stage1', 'stage2']):
            extra = S2_EXTRA.format(seen=seen, mapping=MAPPING) \
                if stage == 'stage2' else ''
            body = TMPL.format(
                stage=stage, t=t, task=task, task_id=i, seen=seen,
                mapping=MAPPING, root=DATA_ROOT + spec['root'],
                ann=spec['train_ann'], img=spec['train_img'], extra=extra)
            path = os.path.join(CFG_DIR, f'dgs_{stage}_t{t:02d}_{task}.py')
            open(path, 'w').write(body)
            print('generated', os.path.basename(path))


if __name__ == '__main__':
    gen()
