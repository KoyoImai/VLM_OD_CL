"""蒸留重み λ の校正（exp_028/029/030/031 共通）.

design: experiments/exp_028/design.md §4.5

    λ = ρ * L_det_hat / L_KD_hat,   ρ = 0.1

L_det_hat / L_KD_hat は、**リプレイのみの実験で実際に生じたドリフト**を基準に測る。
学生に θ^ref（exp_027 の該当ドメイン last）、教師に θ^T（前ドメインの last。t=1 は θ0）を
置き、ドメイン t のバッファ構成と同じミニバッチを K 本流して平均する。

測定は **train モード**で行う（design §4.5「確率的な床」）。学習中に実際に加わる損失と
同じ条件で測ることで、蒸留対象・形式の間で実効強度が揃う。
比較のため eval モード（＝真のドリフト）も併せて記録する。

使い方:
    python experiments/exp_028/calibrate_lambda.py \
        --config experiments/exp_028/configs/kdA_condA_l2_underwater.py \
        --student <θ^ref のパス> --teacher <θ^T のパス> \
        --out experiments/exp_028/calibration/kdA_condA_l2_underwater.json

学習を伴わず前向き計算のみ。GPU 1枚・数分で終わる。
"""
import argparse
import json
import os
import os.path as osp

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

from mmdet.registry import MODELS

RHO = 0.1
ALL_TARGETS = ['img', 'txt', 'fus']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--student', required=True, help='θ^ref（exp_027 の last）')
    p.add_argument('--teacher', required=True, help='θ^T（前ドメイン last / θ0）')
    p.add_argument('--out', required=True)
    p.add_argument('--num-batches', type=int, default=20, help='K')
    p.add_argument('--rho', type=float, default=RHO)
    return p.parse_args()


def main():
    args = parse_args()
    init_default_scope('mmdet')

    cfg = Config.fromfile(args.config)
    cfg.model.backbone.init_cfg = None      # ドライバと同じ扱い
    cfg.model.teacher_ckpt = args.teacher
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    targets_cfg = list(cfg.model.kd.targets)
    form = cfg.model.kd.loss.form

    model = MODELS.build(cfg.model).cuda()
    model.init_weights()                     # 教師を構築（θ^T をロード）
    load_checkpoint(model, args.student, map_location='cpu', logger='current')

    # seed を明示しないと mmengine が sync_random_seed() にフォールバックし、
    # 同じ ckpt でも λ が再現しない（design §4.5 手順2 の「seed 固定」）。
    loader = Runner.build_dataloader(
        cfg.train_dataloader,
        seed=cfg.get('randomness', {}).get('seed', 0))
    it = iter(loader)

    # プロンプト文字列を捕捉する（教師にも同じ文字列を渡すため）
    cap = {}
    model.language_model.register_forward_hook(
        lambda mod, a, o: cap.__setitem__('prompts', a[0]))

    def student_feats(m):
        td = m.language_model(cap['prompts'])
        if m.text_feat_map is not None:
            td['embedded'] = m.text_feat_map(td['embedded'])
        vf = m.extract_feat(inputs)
        enc_in, _ = m.pre_transformer(vf, samples)
        enc = m.forward_encoder(**enc_in, text_dict=td)
        return dict(
            img=vf, txt=td['embedded'], txt_mask=td['text_token_mask'],
            mem=enc['memory'], mem_mask=enc['memory_mask'],
            mem_txt=enc['memory_text'])

    acc = {m: {t: 0.0 for t in ALL_TARGETS} for m in ('train', 'eval')}
    det_acc = 0.0
    n = 0
    for _ in range(args.num_batches):
        batch = next(it)
        data = model.data_preprocessor(batch, training=True)
        inputs, samples = data['inputs'], data['data_samples']

        # 検出損失（train モード。学習時と同じ）
        model.train()
        object.__setattr__(model, '_kd_capture', True)
        object.__setattr__(model, '_kd_cache', {})
        with torch.no_grad():
            model.kd_targets = targets_cfg
            losses = model.loss(inputs, samples)
        object.__setattr__(model, '_kd_capture', False)
        # mmengine の parse_losses と同じ集計（キーに 'loss' を含む全項目の和）。
        # 補助デコーダ層（d0.loss_* など）と DN 損失も検出損失に含まれる。
        det = 0.0
        for k, v in losses.items():
            if 'loss' not in k or k == 'loss_kd':
                continue
            if isinstance(v, torch.Tensor):
                det += float(v.mean())
            elif isinstance(v, (list, tuple)):
                det += float(sum(x.mean() for x in v))
        det_acc += det

        # 蒸留損失（全対象・train と eval の両方）
        model.kd_targets = ALL_TARGETS
        for mode in ('train', 'eval'):
            model.train(mode == 'train')
            with torch.no_grad():
                s = student_feats(model)
                t = model._teacher_features(inputs, cap['prompts'], samples)
            im = model._mlvl_valid_masks(s['img'], samples)
            bs = inputs.shape[0]
            nb = min(model.num_buffer_per_batch, bs)
            buf = slice(bs - nb, bs)
            for tgt in ALL_TARGETS:
                model.kd_targets = [tgt]
                acc[mode][tgt] += float(model._kd_losses(s, t, im, buf))
            model.kd_targets = ALL_TARGETS
        n += 1

    det_hat = det_acc / n
    res = {
        'config': args.config,
        'student_ref': args.student,
        'teacher': args.teacher,
        'num_batches': n,
        'rho': args.rho,
        'form': form,
        'targets_of_this_config': targets_cfg,
        'det_hat_train': det_hat,
        'kd_hat_train': {t: acc['train'][t] / n for t in ALL_TARGETS},
        'kd_hat_eval': {t: acc['eval'][t] / n for t in ALL_TARGETS},
    }
    # 蒸留E は3項の平均（design §4.4）
    res['kd_hat_train']['E'] = sum(
        res['kd_hat_train'][t] for t in ALL_TARGETS) / 3
    res['kd_hat_eval']['E'] = sum(
        res['kd_hat_eval'][t] for t in ALL_TARGETS) / 3
    # λ は train モードの値で決める（design §4.5）
    res['lambda'] = {
        k: args.rho * det_hat / v if v > 0 else None
        for k, v in res['kd_hat_train'].items()
    }
    # この config が使う λ
    key = 'E' if len(targets_cfg) == 3 else targets_cfg[0]
    res['lambda_for_this_config'] = res['lambda'][key]

    os.makedirs(osp.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f'\nλ（この config 用, key={key}） = '
          f'{res["lambda_for_this_config"]:.6g}')


if __name__ == '__main__':
    main()
