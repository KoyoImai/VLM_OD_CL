#!/usr/bin/env python3
"""exp_049: DTG 用の画像特徴を ODinW-13 の 13 タスク全てで事前抽出する。

公式 projects/DGS/configs/IVLOD/extract_odinw_features.py の本環境版。
θ0（凍結 backbone）の neck 出力 4 レベルを GAP した [4, 256] を画像ごとに
`experiments/exp_049/feats/<公式タスク名>/train/img_<id>.pt` に保存する。
backbone は全タスクを通じて凍結なので、抽出は最初に 1 回だけでよい。

データパスの正本は experiments/exp_034/odinw_official_tasks.py（ZiRa 形式）。
公式タスク名 'NorthAmericaMushroom' は本環境ディレクトリ 'NorthAmericaMushrooms'
に対応する（別名マップ）。

使い方:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_049/extract_feats_odinw13.py \
      [--theta0 <θ0.pth>] [--save-dir experiments/exp_049/feats]
"""
import argparse
import os
import sys

import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'exp_034'))

from odinw_official_tasks import DATA_ROOT, ODINW13  # noqa: E402

# 公式タスク名（DGS の順序・METAINFO キー） -> exp_034 テーブルのキー
OFFICIAL_ORDER = [
    'AerialMaritimeDrone', 'Aquarium', 'CottontailRabbits', 'EgoHands',
    'NorthAmericaMushroom', 'Packages', 'PascalVOC', 'pistols', 'pothole',
    'Raccoon', 'ShellfishOpenImages', 'thermalDogsAndPeople',
    'VehiclesOpenImages'
]
ALIAS = {'NorthAmericaMushroom': 'NorthAmericaMushrooms'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--theta0', default=os.path.join(
        ROOT, 'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det'
        '_20231204_095047-b448804b.pth'))
    p.add_argument('--save-dir', default=os.path.join(HERE, 'feats'))
    p.add_argument('--max-samples', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmdet.registry import MODELS, DATASETS
    init_default_scope('mmdet')

    # 素の MM-GDINO（extract_feat だけ使う。MoE 差し替えは不要）
    cfg = Config.fromfile(os.path.join(
        ROOT, 'configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'))
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.theta0, map_location='cpu')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    test_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None,
             imdecode_backend='pillow'),
        dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True,
             backend='pillow'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='PackDetInputs',
             meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                        'scale_factor', 'text', 'custom_entities')),
    ]

    for name in OFFICIAL_ORDER:
        spec = ODINW13[ALIAS.get(name, name)]
        save_dir = os.path.join(args.save_dir, name, 'train')
        os.makedirs(save_dir, exist_ok=True)
        ds_cfg = dict(
            type='CocoDataset',
            metainfo=dict(classes=spec['classes']),
            data_root=DATA_ROOT + spec['root'],
            ann_file=spec['train_ann'],
            data_prefix=dict(img=spec['train_img']),
            filter_cfg=dict(filter_empty_gt=False),
            pipeline=test_pipeline,
            test_mode=True,
            return_classes=True)
        dataset = DATASETS.build(ds_cfg)
        n = len(dataset) if args.max_samples is None else min(
            len(dataset), args.max_samples)
        done = len([f for f in os.listdir(save_dir) if f.endswith('.pt')])
        if done >= n:
            print(f'[skip] {name}: {done} 件 抽出済み')
            continue
        print(f'[extract] {name}: {n} 件 -> {save_dir}')
        with torch.no_grad():
            for i in tqdm(range(n)):
                item = dataset[i]
                data = model.data_preprocessor(
                    {'inputs': [item['inputs']],
                     'data_samples': [item['data_samples']]}, False)
                feats = model.extract_feat(data['inputs'])
                pooled = torch.stack([
                    torch.nn.functional.adaptive_avg_pool2d(f, 1).view(-1)
                    for f in feats], dim=0)  # [4, 256]
                img_id = item['data_samples'].img_id
                torch.save(pooled.cpu(),
                           os.path.join(save_dir, f'img_{img_id}.pt'))
    print('done')


if __name__ == '__main__':
    main()
