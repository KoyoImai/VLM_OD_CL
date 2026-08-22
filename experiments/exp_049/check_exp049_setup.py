"""exp_049 実装検証（implementation_plan §4 / design.md §3-1）。

使い方:
    python experiments/exp_049/check_exp049_setup.py           # 静的検証（GPU 不要）
    CUDA_VISIBLE_DEVICES=0 python experiments/exp_049/check_exp049_setup.py --gpu
        # + 実データ 1 step（stage1/stage2）・ルーティング（GPU 1 枚・要 θ0・
        #   要サブセット特徴: --feats で指定、無ければ 20 枚/2 タスクを自動抽出）

検証項目
  1. config: 学習 config（stage1/2 ベース＋タスク別 25 本）と評価 config（14 本）が
     build できること。タスク別 config の task_id / seen_tasks / データパスの整合。
  2. 凍結と容量: 学習対象が LoRA（688,128 = 6 層 × [img FFN 73,728 + text FFN 40,960]）
     ＋ dn label_embedding（stage1 のみ。公式も凍結しない）だけであること。
  3. マージの数値検証: LoraLinearRep.__rep__ が 新 ← λ·base + (1−λ)·新（λ=0.2）を
     計算して base に昇格し、新キーを削除すること（純テンソル）。
  4. （--gpu）θ0 ロード: moe_lora 変換後に lora 以外の欠落キーが 0。
  5. （--gpu）stage1 1 step: loss 有限・lora_B に勾配。
  6. （--gpu）stage2 1 step: 擬似ラベル経路（3 dict の head.loss）を通り loss 有限・
     dn 損失が無く、**トポロジー蒸留（inter_text_loss / inter_query_loss）が乗る**こと。
     feat_distn.type は 'inter-class'（公開 config の 'inter-intra' は実装分岐に無く
     KD が不活性になる名称ズレ。Appendix B.4-B.5 に基づき訂正 = 案A、2026-08-21）。
     検証では擬似ラベルを確実に立てるため σ を 0.02 に下げる（本走は 0.4。
     σ=0.4 でバッチ内に旧クラス物体が無ければ KD 項が出ないのは正しい挙動）。
     loss_ld（logits 蒸留）が無いのは論文の最終式(13) どおり。
  7. （--gpu）DTG とルーティング: 統計推定 → 割当（閾値の両側で分岐）→
     ドメイン内画像が正グループ・閾値を絞ると -1（zero-shot フォールバック）。
"""
import argparse
import copy
import glob
import math
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

THETA0 = os.path.join(
    ROOT, 'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det'
    '_20231204_095047-b448804b.pth')
ORDER = ['AerialMaritimeDrone', 'Aquarium', 'CottontailRabbits', 'EgoHands',
         'NorthAmericaMushroom', 'Packages', 'PascalVOC', 'pistols', 'pothole',
         'Raccoon', 'ShellfishOpenImages', 'thermalDogsAndPeople',
         'VehiclesOpenImages']
results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


def build_model(cfg_path, feat_path, stats_path, mapping, task_id, seen,
                sigma=None):
    cfg = Config.fromfile(cfg_path)
    import_modules_from_strings(**cfg.custom_imports)
    cfg.model.domain_predictor_cfg.feat_path = feat_path
    cfg.model.domain_predictor_cfg.stats_path = stats_path
    cfg.model.domain_predictor_cfg.task_id_mapping_path = mapping
    cfg.model.task_id = task_id
    cfg.model.seen_tasks = ','.join(seen)
    if 'distn_cfg' in cfg.model:
        cfg.model.distn_cfg.task_id_mapping_path = mapping
        cfg.train_dataloader.dataset.distn_cfg.task_id_mapping_path = mapping
        if sigma is not None:  # head は init 時に閾値を取り込むため build 前に設定する
            cfg.model.distn_cfg.label_distn.sigma = sigma
            cfg.model.bbox_head.distn_cfg.label_distn.sigma = sigma
    from mmdet.registry import MODELS
    return cfg, MODELS.build(cfg.model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--feats', default=None,
                    help='サブセット特徴のディレクトリ（無ければ自動抽出）')
    args = ap.parse_args()
    init_default_scope('mmdet')

    # --- 1. config build と整合 ------------------------------------------
    bad = []
    n = 0
    for p in sorted(glob.glob(f'{HERE}/configs/dgs_stage[12]_t*.py')):
        try:
            c = Config.fromfile(p)
            base = os.path.basename(p)
            t = int(base.split('_t')[1][:2])
            task = base.split('_', 3)[3][:-3]
            assert c.model.task_id == t - 1, 'task_id 不一致'
            assert c.model.seen_tasks.split(',') == ORDER[:t], 'seen 不一致'
            assert c.model.seen_tasks.split(',')[-1] == task
            assert os.path.isfile(os.path.join(
                c.train_dataloader.dataset.data_root,
                c.train_dataloader.dataset.ann_file)), 'ann 不在'
            n += 1
        except Exception as e:
            bad.append(f'{os.path.basename(p)}: {e}')
    for p in sorted(glob.glob(f'{HERE}/configs/eval_*.py')):
        try:
            Config.fromfile(p)
            n += 1
        except Exception as e:
            bad.append(f'{os.path.basename(p)}: {e}')
    check(1, '学習 25 本＋評価 14 本の build・整合', not bad and n == 39,
          f'{n}/39 / 問題 {bad[:2]}')

    # --- 3. マージの数値検証（純テンソル） --------------------------------
    from projects.dgs_cl.layers.modules.group_lora import LoraLinearRep
    layer = LoraLinearRep(
        in_features=8, out_features=8, r=4, lora_alpha=8, task_ids=[0, 1],
        cur_task_id=1,
        group_cfg=dict(type='rep', merge_method='ema', lambda_A=0.2, lambda_B=0.2))
    with torch.no_grad():
        for d in (layer.lora_A, layer.lora_B):
            for k in d:
                d[k].normal_()
    A0, A1 = layer.lora_A['0'].clone(), layer.lora_A['1'].clone()
    B0, B1 = layer.lora_B['0'].clone(), layer.lora_B['1'].clone()
    layer.__rep__(delete=True)
    ok = (torch.allclose(layer.lora_A['0'], 0.2 * A0 + 0.8 * A1)
          and torch.allclose(layer.lora_B['0'], 0.2 * B0 + 0.8 * B1)
          and '1' not in layer.lora_A)
    check(3, 'EMA マージ（新←0.2·base+0.8·新→base 昇格・新削除）', ok)

    if not args.gpu:
        print(f'\n{sum(results)}/{len(results)} OK（--gpu で 2,4-7 を実行）')
        sys.exit(0 if all(results) else 1)

    import tempfile
    import yaml
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import DATASETS
    from projects.dgs_cl.hooks.transform_builder import MoeLoraTransform

    tmp = tempfile.mkdtemp(prefix='exp049_check_')
    feats = args.feats or os.path.join(tmp, 'feats')
    stats = os.path.join(tmp, 'stats')
    os.makedirs(stats, exist_ok=True)

    if args.feats is None:
        # 2 タスク × 20 枚だけ抽出（機構検証用。閾値の挙動は本統計と異なる）
        os.system(f'python {HERE}/extract_feats_odinw13.py '
                  f'--save-dir {feats} --max-samples 20 > /dev/null 2>&1')

    # --- 7a. DTG: 統計推定と割当の両分岐 ----------------------------------
    from projects.dgs_cl.domain_predictor.builder import AdaptiveDomainPredictor
    wd1 = os.path.join(tmp, 'wd_expand'); os.makedirs(wd1)
    dp_cfg = Config(dict(type='svd', feat_path=feats + '/', stats_path=stats + '/',
                         multilevel=False, expand_th=150, ood_th=500,
                         min_eig_ratio=1e-3))
    for t in range(2):
        AdaptiveDomainPredictor(dp_cfg, num_tasks=13, moe_modules=[], task_id=t,
                                seen_tasks=ORDER[:t + 1], work_dirs=wd1)
    m1 = yaml.safe_load(open(os.path.join(wd1, 'task_id_mapping.yaml')))
    wd2 = os.path.join(tmp, 'wd_merge'); os.makedirs(wd2)
    dp_cfg2 = copy.deepcopy(dp_cfg); dp_cfg2.expand_th = 1e12
    for t in range(2):
        AdaptiveDomainPredictor(dp_cfg2, num_tasks=13, moe_modules=[], task_id=t,
                                seen_tasks=ORDER[:t + 1], work_dirs=wd2)
    m2 = yaml.safe_load(open(os.path.join(wd2, 'task_id_mapping.yaml')))
    check(7, 'DTG: 閾値の両側で 新グループ/併合 に分岐',
          m1 == {0: 0, 1: 1} and m2 == {0: 0, 1: 0},
          f'expand: {m1} / merge: {m2}')

    # --- 2/4/5. stage1 build・θ0 ロード・1 step ---------------------------
    mp1 = os.path.join(wd1, 'task_id_mapping.yaml')
    cfg1, model = build_model(f'{HERE}/configs/dgs_stage1_t01_AerialMaritimeDrone.py',
                              feats + '/', stats + '/', mp1, 0, ORDER[:1])
    tr = {k: p.numel() for k, p in model.named_parameters() if p.requires_grad}
    lora_n = sum(v for k, v in tr.items() if 'lora_' in k)
    non_lora = [k for k in tr if 'lora_' not in k]
    check(2, '凍結と容量（LoRA 688,128 ＋ dn label_embedding のみ）',
          lora_n == 688128 and non_lora == ['dn_query_generator.label_embedding.weight'],
          f'lora={lora_n:,} / 非lora={non_lora}')

    ckpt = torch.load(THETA0, map_location='cpu')['state_dict']
    ckpt = MoeLoraTransform().transform(model, ckpt)
    missing, _ = model.load_state_dict(ckpt, strict=False)
    miss = [k for k in missing if 'lora_' not in k]
    check(4, 'θ0 ロード（moe_lora 変換後の欠落 = lora のみ）', not miss, str(miss[:3]))

    model = model.cuda(); model.train()
    ds = DATASETS.build(cfg1.train_dataloader.dataset)
    batch = pseudo_collate([ds[i] for i in range(2)])
    data = model.data_preprocessor(batch, True)
    losses = model.loss(data['inputs'], data['data_samples'])
    total = sum(v.mean() if hasattr(v, 'mean') else sum(x.mean() for x in v)
                for k, v in losses.items() if 'loss' in k)
    fin = bool(torch.isfinite(total))
    total.backward()
    gB = [n_ for n_, p in model.named_parameters()
          if p.requires_grad and 'lora_B' in n_ and p.grad is not None
          and p.grad.abs().sum() > 0]
    check(5, 'stage1 1 step（loss 有限・lora_B 24 本に勾配・dn 損失あり）',
          fin and len(gB) == 24 and any('dn_loss' in k for k in losses),
          f'loss={float(total):.2f} grad_B={len(gB)}')
    del model; torch.cuda.empty_cache()

    # --- 6. stage2 1 step --------------------------------------------------
    mp2 = os.path.join(wd2, 'task_id_mapping.yaml')
    # 検証用に σ=0.02 で build（擬似ラベル → KD 経路を確実に踏ませる。docstring 参照）
    cfg2, model = build_model(f'{HERE}/configs/dgs_stage2_t02_Aquarium.py',
                              feats + '/', stats + '/', mp2, 1, ORDER[:2],
                              sigma=0.02)
    assert cfg2.model.distn_cfg.feat_distn.type == 'inter-class'
    ckpt = torch.load(THETA0, map_location='cpu')['state_dict']
    ckpt = MoeLoraTransform().transform(model, ckpt)
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda(); model.train()
    trace = dict(old=None)
    _o = model.bbox_head.loss_by_feat_old
    def _wo(*a, **k):
        out = _o(*a, **k); trace['old'] = list(out.keys()); return out
    model.bbox_head.loss_by_feat_old = _wo
    ds2cfg = copy.deepcopy(cfg2.train_dataloader.dataset)
    ds2 = DATASETS.build(ds2cfg)
    batch = pseudo_collate([ds2[i] for i in range(2)])
    data = model.data_preprocessor(batch, True)
    losses = model.loss(data['inputs'], data['data_samples'])
    total = sum(v.mean() if hasattr(v, 'mean') else sum(x.mean() for x in v)
                for k, v in losses.items() if 'loss' in k)
    check(6, 'stage2 1 step（3dict 経路・loss 有限・dn 無し・トポロジー蒸留が乗る）',
          bool(torch.isfinite(total))
          and sorted(trace['old']) == ['inter_query_loss', 'inter_text_loss']
          and not any('dn_loss' in k for k in losses),
          f'loss={float(total):.2f} old_keys={trace["old"]}')

    # --- 7b. ルーティング（ドメイン内 → タスク 0、閾値を絞ると -1） --------
    model.eval()
    item = ds[0]
    d = model.data_preprocessor({'inputs': [item['inputs']],
                                 'data_samples': [item['data_samples']]}, False)
    vf = model.extract_feat(d['inputs'])
    with torch.no_grad():
        tid = model.domain_predictor(vf, input_sample=None)
        model.domain_predictor.ood_th = 1e-6
        tid_ood = model.domain_predictor(vf, input_sample=None)
    check(7, 'ルーティング（AerialMaritimeDrone→0 / ood 強制→-1）',
          tid == 0 and tid_ood == -1, f'tid={tid} ood={tid_ood}')

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
