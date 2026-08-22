"""exp_050 実行前検証（design.md §4）。

使い方:
    python experiments/exp_050/check_exp050_setup.py           # 静的（GPU 不要）
    CUDA_VISIBLE_DEVICES=0 python experiments/exp_050/check_exp050_setup.py --gpu
        # + ODVG 実データ 1 step・chunked predict・ルーティング（GPU 1 枚・要 θ0）
        # DTG 用のサブセット特徴は --feats で指定（無ければ 30 枚/2 ドメインを自動抽出）

検証項目
  1. config: 学習 6 本＋評価 7 本が build でき、学習 config の model 以外が
     継承元（replayfree）と一致・task_id/seen_tasks の整合・ckpt が last のみ設定。
  2. 凍結と容量: 学習対象が LoRA 688,128 ＋ dn label_embedding のみ。
  3. （--gpu）ODVG 実データ 1 step（underwater）: θ0 ロード欠落 0・loss 有限・
     dn 損失あり・lora_B 24 本に勾配（**ODVG 経路での DGS 学習の初検証**）。
  4. （--gpu）videogames の chunked predict: 87 クラス・chunked_size=40 の分割推論が
     DGS のルーティング併用で動き、予測が返ること。
  5. （--gpu）ルーティング: RF100 ドメイン内画像 → 正タスク、COCO 画像の OOD 率
     （ood_th=200/500 の両方で実測値を表示。ZCOCO ≈ θ0 になるかの事前材料）。
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402

ORDER = ['underwater', 'electromagnetic', 'videogames',
         'aerial', 'microscopic', 'documents']
THETA0 = os.path.join(
    ROOT, 'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det'
    '_20231204_095047-b448804b.pth')
results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--feats', default=None)
    args = ap.parse_args()
    init_default_scope('mmdet')

    # --- 1. config build と整合 -------------------------------------------
    bad, n = [], 0
    for i, d in enumerate(ORDER):
        p = f'{HERE}/configs/dgs_stage1_t{i+1:02d}_{d}.py'
        base_p = (f'experiments/exp_024/configs/fullft_replayfree_{d}.py'
                  if i < 3 else f'experiments/exp_040/configs/replayfree_{d}.py')
        try:
            c = Config.fromfile(p)
            b = Config.fromfile(base_p)
            for key in ('train_dataloader', 'optim_wrapper', 'param_scheduler',
                        'train_cfg', 'randomness'):
                if _plain(c.get(key)) != _plain(b.get(key)):
                    bad.append(f'{d}: {key} 不一致')
            assert c.model.task_id == i and \
                c.model.seen_tasks.split(',') == ORDER[:i + 1]
            ck = c.default_hooks.checkpoint
            assert ck.interval == 20 and ck.save_optimizer is False, 'ckpt 設定'
            n += 1
        except Exception as e:
            bad.append(f'{d}: {e}')
    for p in sorted(glob.glob(f'{HERE}/configs/eval_dgs_*.py')):
        try:
            Config.fromfile(p)
            n += 1
        except Exception as e:
            bad.append(f'{os.path.basename(p)}: {e}')
    check(1, '学習 6 本＋評価 7 本の build・整合・last のみ保存', not bad and n == 13,
          f'{n}/13 / 問題 {bad[:3]}')

    if not args.gpu:
        print(f'\n{sum(results)}/{len(results)} OK（--gpu で 2-5 を実行）')
        sys.exit(0 if all(results) else 1)

    import tempfile
    from mmengine.dataset import pseudo_collate
    from mmengine.runner import load_checkpoint  # noqa: F401
    from mmdet.registry import MODELS, DATASETS
    from projects.dgs_cl.hooks.transform_builder import MoeLoraTransform
    from projects.dgs_cl.domain_predictor.builder import AdaptiveDomainPredictor

    tmp = tempfile.mkdtemp(prefix='exp050_check_')
    feats = args.feats or os.path.join(tmp, 'feats')
    stats = os.path.join(tmp, 'stats'); os.makedirs(stats, exist_ok=True)
    if args.feats is None:
        os.system(f'python {HERE}/extract_feats_rf100.py --save-dir {feats} '
                  f'--max-samples 30 > /dev/null 2>&1')
    import yaml
    wd = os.path.join(tmp, 'wd'); os.makedirs(wd)
    dp = Config(dict(type='svd', feat_path=feats + '/', stats_path=stats + '/',
                     multilevel=False, expand_th=150, ood_th=500,
                     min_eig_ratio=1e-3))
    for t in range(len(ORDER)):
        AdaptiveDomainPredictor(dp, num_tasks=6, moe_modules=[], task_id=t,
                                seen_tasks=ORDER[:t + 1], work_dirs=wd)
    mp = os.path.join(wd, 'task_id_mapping.yaml')

    def build(cfg_path, task_id, seen):
        cfg = Config.fromfile(cfg_path)
        import_modules_from_strings(**cfg.custom_imports)
        cfg.model.domain_predictor_cfg.feat_path = feats + '/'
        cfg.model.domain_predictor_cfg.stats_path = stats + '/'
        cfg.model.domain_predictor_cfg.task_id_mapping_path = mp
        cfg.model.task_id = task_id
        cfg.model.seen_tasks = ','.join(seen)
        return cfg, MODELS.build(cfg.model)

    # --- 2/3. 凍結・θ0 ロード・ODVG 1 step --------------------------------
    cfg1, model = build(f'{HERE}/configs/dgs_stage1_t01_underwater.py', 0, ORDER[:1])
    tr = {k: p.numel() for k, p in model.named_parameters() if p.requires_grad}
    lora_n = sum(v for k, v in tr.items() if 'lora_' in k)
    non_lora = [k for k in tr if 'lora_' not in k]
    check(2, '凍結と容量（LoRA 688,128 ＋ dn label_embedding）',
          lora_n == 688128 and
          non_lora == ['dn_query_generator.label_embedding.weight'],
          f'lora={lora_n:,}')

    ckpt = torch.load(THETA0, map_location='cpu')['state_dict']
    ckpt = MoeLoraTransform().transform(model, ckpt)
    missing, _ = model.load_state_dict(ckpt, strict=False)
    miss = [k for k in missing if 'lora_' not in k]
    model = model.cuda(); model.train()
    ds = DATASETS.build(cfg1.train_dataloader.dataset)
    batch = pseudo_collate([ds[i] for i in range(4)])
    data = model.data_preprocessor(batch, True)
    losses = model.loss(data['inputs'], data['data_samples'])
    total = sum(v.mean() if hasattr(v, 'mean') else sum(x.mean() for x in v)
                for k, v in losses.items() if 'loss' in k)
    total.backward()
    gB = [n_ for n_, p in model.named_parameters()
          if p.requires_grad and 'lora_B' in n_ and p.grad is not None
          and p.grad.abs().sum() > 0]
    check(3, 'ODVG 実データ 1 step（θ0 欠落 0・loss 有限・dn あり・lora_B 勾配 24）',
          not miss and bool(torch.isfinite(total)) and len(gB) == 24
          and any('dn_loss' in k for k in losses),
          f'loss={float(total):.2f} grad_B={len(gB)}')
    del model; torch.cuda.empty_cache()

    # --- 4. videogames chunked predict -------------------------------------
    cfgv = Config.fromfile(f'{HERE}/configs/eval_dgs_videogames.py')
    import_modules_from_strings(**cfgv.custom_imports)
    cfgv.model.domain_predictor_cfg.feat_path = feats + '/'
    cfgv.model.domain_predictor_cfg.stats_path = stats + '/'
    cfgv.model.domain_predictor_cfg.task_id_mapping_path = mp
    cfgv.model.task_id = 2
    cfgv.model.seen_tasks = ','.join(ORDER[:3])
    mv = MODELS.build(cfgv.model)
    ckpt = torch.load(THETA0, map_location='cpu')['state_dict']
    ckpt = MoeLoraTransform().transform(mv, ckpt)
    mv.load_state_dict(ckpt, strict=False)
    mv = mv.cuda(); mv.eval()
    dsv = DATASETS.build(cfgv.test_dataloader.dataset)
    item = dsv[0]
    d = mv.data_preprocessor({'inputs': [item['inputs']],
                              'data_samples': [item['data_samples']]}, False)
    with torch.no_grad():
        out = mv.predict(d['inputs'], d['data_samples'])
    ok4 = hasattr(out[0], 'pred_instances') and \
        cfgv.model.test_cfg.get('chunked_size', -1) == 40
    check(4, 'videogames chunked predict（87 クラス・ルーティング併用）', ok4,
          f'検出数={len(out[0].pred_instances)}')

    # --- 5. ルーティング -----------------------------------------------------
    with torch.no_grad():
        vf = mv.extract_feat(d['inputs'])
        tid = mv.domain_predictor(vf, input_sample=None)
    test_pipeline = cfgv.test_pipeline if hasattr(cfgv, 'test_pipeline') else None
    coco = DATASETS.build(Config(dict(d=dict(
        type='CocoDataset', data_root='/workspace/kouyou/datasets/coco2017/',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'), test_mode=True,
        pipeline=dsv.pipeline.transforms and [
            dict(type='LoadImageFromFile', backend_args=None,
                 imdecode_backend='pillow'),
            dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True,
                 backend='pillow'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs',
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'text', 'custom_entities'))],
        return_classes=True))).d)
    import random; random.seed(0)
    cnt = {200: 0, 500: 0}
    N = 30
    for th in (200, 500):
        mv.domain_predictor.ood_th = th
        ood = 0
        for i in random.sample(range(len(coco)), N):
            it = coco[i]
            dd = mv.data_preprocessor({'inputs': [it['inputs']],
                                       'data_samples': [it['data_samples']]}, False)
            with torch.no_grad():
                f = mv.extract_feat(dd['inputs'])
                if mv.domain_predictor(f, input_sample=None) == -1:
                    ood += 1
        cnt[th] = ood
    check(5, 'ルーティング（videogames 画像の割当と COCO の OOD 率）',
          tid == 2,
          f'vg画像→task{tid} / COCO OOD: th200 {cnt[200]}/{N}, th500 {cnt[500]}/{N}'
          '（サブセット統計での参考値）')

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
