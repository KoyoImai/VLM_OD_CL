#!/usr/bin/env python3
"""exp_050: DTG 用の画像特徴を RF100 の 6 ドメイン全てで事前抽出する。

exp_049 の extract_feats_odinw13.py の RF100 版。θ0（凍結 backbone）の neck 出力
4 レベルを GAP した [4, 256] を画像ごとに
`experiments/exp_050/feats/<domain>/train/img_<id>.pt` に保存する（約 80,300 枚）。
決定的（θ0・拡張なし）なので、本環境とクラスタのどちらで実行しても同一。

使い方:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_050/extract_feats_rf100.py
"""
import argparse
import os
import sys

import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

ORDER = ['underwater', 'electromagnetic', 'videogames',
         'aerial', 'microscopic', 'documents']
DATA_ROOT = '/workspace/kouyou/datasets/rf100_domain/'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--theta0', default=None)
    p.add_argument('--save-dir', default=os.path.join(HERE, 'feats'))
    p.add_argument('--max-samples', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    theta0 = args.theta0
    if theta0 is None:
        for c in [os.path.join(ROOT, 'grounding_dino_swin-t_pretrain_obj365_goldg'
                               '_grit9m_v3det_20231204_095047-b448804b.pth'),
                  '/workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_'
                  'goldg_grit9m_v3det_20231204_095047-b448804b.pth']:
            if os.path.isfile(c):
                theta0 = c
                break
    assert theta0, 'θ0 が見つからない（--theta0 で指定）'

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmdet.registry import MODELS, DATASETS
    init_default_scope('mmdet')

    cfg = Config.fromfile(os.path.join(
        ROOT, 'configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'))
    model = MODELS.build(cfg.model)
    load_checkpoint(model, theta0, map_location='cpu')
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

    for name in ORDER:
        save_dir = os.path.join(args.save_dir, name, 'train')
        os.makedirs(save_dir, exist_ok=True)
        ds_cfg = dict(
            type='CocoDataset',
            data_root=DATA_ROOT + name + '/',
            ann_file='train/_annotations.coco.json',
            data_prefix=dict(img='train/'),
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
                    for f in feats], dim=0)
                img_id = item['data_samples'].img_id
                torch.save(pooled.cpu(),
                           os.path.join(save_dir, f'img_{img_id}.pt'))
    print('done')


if __name__ == '__main__':
    main()
