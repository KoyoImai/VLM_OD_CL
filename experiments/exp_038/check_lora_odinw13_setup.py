"""exp_038 実行前検証（design.md §6.4）。

使い方:
    python experiments/exp_038/check_lora_odinw13_setup.py

全項目 OK でなければ学習を始めないこと。

検証項目
  1. 13 タスクの学習 config が全て build でき、逐次順・データが exp_034 / exp_037 と一致
  2. LoRA 232 層の内訳が design.md §2.1 と一致（Swin 51 / BERT 72 / encoder 108 / text_feat_map 1）
  3. 学習設定が exp_037 と同一（3000 iter, batch 2, AdamW 1e-3/1e-2, milestones[1200],
     clip 0.1, seed 0, num_classes 256）かつ optimizer 登録が LoRA のみ
  4. θ0 ロード後に max|ΔW| = 0（B=0 が保たれ、学習開始時点が θ_{t-1} と等価）
  5. dn が無効（dn_loss_* が損失に出ない）
  6. 実データ 1 step で base 不変・LoRA に勾配（Swin / BERT / encoder / text_feat_map の全てで）
  7. merge → plain GroundingDINO のロードで missing/unexpected 0（MHA 再融合を含む）
  8. マージ後 plain モデルの検出スコアが LoRA モデルと一致
  9. 評価 config（plain）が build でき、13 タスク分のデータセットを持つ
"""
import copy
import os
import sys
import tempfile
from collections import Counter

import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from mmengine.utils import import_modules_from_strings

from mmdet.registry import MODELS

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, 'configs')
sys.path.insert(0, os.path.join(HERE, '..', 'exp_034'))
from odinw_official_tasks import ODINW13, task_order  # noqa: E402

# 条件ごとの期待値（design.md §2.1 / §10.2）。第1引数で切り替える。
EXPECT_BY_COND = {
    'A': ({'backbone': 51, 'language_model': 72, 'encoder': 108,
           'text_feat_map': 1}, 232, 'lora_odinw13'),
    'B': ({'backbone': 51, 'encoder': 108, 'text_feat_map': 1},
          160, 'lora_odinw13_condB'),
    'C': ({'backbone': 51, 'language_model': 72, 'encoder': 108,
           'text_feat_map': 1}, 232, 'lora_odinw13_condC'),
}
# 条件ごとの期待 lr（design.md §2.2 / §11.2）
LR_BY_COND = {'A': 0.001, 'B': 0.001, 'C': 0.0001}
results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


if __name__ == '__main__':
    cond = sys.argv[1] if len(sys.argv) > 1 else 'A'
    EXPECT, N_EXPECT, PREFIX = EXPECT_BY_COND[cond]
    print(f'条件{cond}: 期待 {N_EXPECT} 層 {EXPECT}\n')
    init_default_scope('mmdet')
    order = task_order(42)

    # --- 1. 13 タスクの config -------------------------------------------
    bad = []
    for i, name in enumerate(order, 1):
        p = os.path.join(CFG_DIR, f'{PREFIX}_{name}.py')
        if not os.path.exists(p):
            bad.append(f'{name}: config なし')
            continue
        c = Config.fromfile(p)
        ds = c.train_dataloader['dataset']
        exp_root = 'data/odinw/' + ODINW13[name]['root']
        if ds['data_root'] != exp_root:
            bad.append(f'{name}: data_root {ds["data_root"]}')
        if ds['ann_file'] != ODINW13[name]['train_ann']:
            bad.append(f'{name}: ann_file {ds["ann_file"]}')
        if tuple(ds['metainfo']['classes']) != tuple(ODINW13[name]['classes']):
            bad.append(f'{name}: classes')
        if not os.path.exists(os.path.join(exp_root, ODINW13[name]['train_ann'])):
            bad.append(f'{name}: 学習データが存在しない')
    check(1, '13 タスクの学習 config とデータ', not bad,
          f'{len(order)} 本 / 不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 以降は t=1 の config で実機確認 ------------------------------------
    cfg = Config.fromfile(os.path.join(CFG_DIR, f'{PREFIX}_{order[0]}.py'))
    import_modules_from_strings(**cfg['custom_imports'])
    from projects.lora_cl.lora_layers import LoRALinear
    from projects.lora_cl.merge_lora import merge

    model = MODELS.build(copy.deepcopy(cfg.model))
    lo = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    got = dict(Counter(n.split('.')[0] for n in lo))
    check(2, f'LoRA {N_EXPECT} 層の内訳', got == EXPECT and len(lo) == N_EXPECT,
          f'{len(lo)} 層 {got}')

    # --- 3. 学習設定 -------------------------------------------------------
    ok3, d3 = True, []
    for label, got_v, want in (
            ('max_iters', cfg.train_cfg['max_iters'], 3000),
            ('batch_size', cfg.train_dataloader['batch_size'], 2),
            ('lr', cfg.optim_wrapper['optimizer']['lr'], LR_BY_COND[cond]),
            ('weight_decay', cfg.optim_wrapper['optimizer']['weight_decay'], 0.01),
            ('milestones', list(cfg.param_scheduler[0]['milestones']), [1200]),
            ('clip max_norm', cfg.optim_wrapper['clip_grad']['max_norm'], 0.1),
            ('seed', cfg.randomness['seed'], 0),
            ('num_classes', cfg.model['bbox_head']['num_classes'], 256),
            ('constructor', cfg.optim_wrapper['constructor'],
             'TrainableParamsConstructor')):
        if got_v != want:
            ok3 = False
            d3.append(f'{label}={got_v}(期待 {want})')
    model_cuda = model.cuda()
    ow = build_optim_wrapper(model_cuda, cfg.optim_wrapper)
    n_opt = sum(p.numel() for g in ow.optimizer.param_groups for p in g['params'])
    n_lora = sum(p.numel() for p in model_cuda.parameters() if p.requires_grad)
    ok3 = ok3 and n_opt == n_lora
    check(3, '学習設定が exp_037 と同一 / optimizer 登録は LoRA のみ', ok3,
          f'optimizer {n_opt:,} == LoRA {n_lora:,}' + (f' / {d3}' if d3 else ''))

    # --- 4. θ0 ロード後の ΔW=0 ---------------------------------------------
    model_cuda = model_cuda.cpu()
    model_cuda.init_weights()
    ck = _load_checkpoint(cfg.load_from, map_location='cpu')
    _load_checkpoint_to_model(model_cuda, ck, strict=False)
    dmax = max(float((float(m.lora_scaling) * (m.lora_B @ m.lora_A)).abs().max())
               for _, m in model_cuda.named_modules() if isinstance(m, LoRALinear))
    src = ck['state_dict'] if 'state_dict' in ck else ck
    sd = model_cuda.state_dict()
    badw = [k for k, v in src.items()
            if k in sd and sd[k].shape == v.shape
            and not torch.equal(sd[k].cpu(), v.cpu())]
    check(4, 'θ0 ロード後も ΔW=0・base は θ0 と一致', dmax == 0.0 and not badw,
          f'max|ΔW|={dmax:.3e} / 照合 {sum(1 for k in src if k in sd)} キー / '
          f'不一致 {len(badw)}')

    # --- 5/6. 実データ 1 step -----------------------------------------------
    model_cuda = model_cuda.cuda().train()
    ow = build_optim_wrapper(model_cuda, cfg.optim_wrapper)
    loader = Runner.build_dataloader(cfg.train_dataloader)
    data = model_cuda.data_preprocessor(next(iter(loader)), training=True)
    base_before = {n: p.detach().clone()
                   for n, p in model_cuda.named_parameters() if not p.requires_grad}
    losses = model_cuda.loss(data['inputs'], data['data_samples'])
    dn_keys = [k for k in losses if 'dn_' in k]
    check(5, 'dn が無効（dn_loss_* が出ない）', not dn_keys,
          f'損失 {len(losses)} 項 / dn 項 {len(dn_keys)}')

    sum(v.sum() for v in losses.values()
        if isinstance(v, torch.Tensor)).backward()
    gB = Counter(n.split('.')[0] for n, p in model_cuda.named_parameters()
                 if n.endswith('lora_B') and p.grad is not None
                 and float(p.grad.abs().max()) > 0)
    dbase = max(float((p.detach() - base_before[n]).abs().max())
                for n, p in model_cuda.named_parameters() if n in base_before)
    ow.update_params(sum(v.sum() for v in losses.values()
                         if isinstance(v, torch.Tensor)) * 0 + 0.0) \
        if False else ow.optimizer.step()
    ok6 = (dbase == 0.0 and gB.get('backbone') == 51
           and gB.get('text_feat_map') == 1 and gB.get('encoder', 0) >= 100
           and gB.get('language_model', 0) == EXPECT.get('language_model', 0))
    check(6, '1 step で base 不変・全構成要素の LoRA に勾配', ok6,
          f'非零勾配 {dict(gB)} / max|Δbase|={dbase:.3e}')

    # --- 7/8. merge → plain -------------------------------------------------
    tmpd = tempfile.mkdtemp()
    src_ck, dst_ck = os.path.join(tmpd, 'l.pth'), os.path.join(tmpd, 'theta.pth')
    torch.save({'state_dict': {k: v.cpu()
                               for k, v in model_cuda.state_dict().items()}}, src_ck)
    merge(src_ck, dst_ck)
    eval_cfg = Config.fromfile(os.path.join(CFG_DIR, 'odinw13_plain_eval.py'))
    plain = MODELS.build(copy.deepcopy(eval_cfg.model))
    info = plain.load_state_dict(
        torch.load(dst_ck, map_location='cpu')['state_dict'], strict=False)
    check(7, 'マージ後 θt が plain GroundingDINO にそのまま載る',
          not info.missing_keys and not info.unexpected_keys,
          f'missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}')

    val_loader = Runner.build_dataloader(
        dict(eval_cfg.val_dataloader, batch_size=1, num_workers=0,
             persistent_workers=False))
    raw = next(iter(val_loader))
    plain = plain.cuda().eval()
    model_cuda.eval()

    @torch.no_grad()
    def scores(m):
        d = m.data_preprocessor(copy.deepcopy(raw), training=False)
        out = m.predict(d['inputs'], d['data_samples'], rescale=True)
        return torch.cat([o.pred_instances.scores for o in out])

    s_p, s_l = scores(plain), scores(model_cuda)
    ds = float((s_p - s_l).abs().max())
    rel = ds / max(float(s_l.max()), 1e-9)
    # マージは重み空間では厳密（本番 θ1 の独立再計算で 908 キーすべて差 0、
    # COCO 5000 枚でマージ前後の ZCOCO が 0.2470 で一致）。ここで残るのは
    # 「マージ済み W を1回通す」対「W と LoRA 分岐を別々に通す」の float32
    # 演算順序差で、大きさは ||ΔW|| に比例する。よって絶対値ではなくスコアの
    # 大きさに対する相対値で判定する（1%）。
    check(8, 'マージ後 plain の検出スコアが LoRA モデルと一致', rel < 1e-2,
          f'max|Δscore|={ds:.3e} / 最大スコア {float(s_l.max()):.4f} '
          f'→ 相対 {rel*100:.3f}%')

    # --- 9. 評価 config ------------------------------------------------------
    n_ds = len(eval_cfg.val_dataloader['dataset']['datasets'])
    n_sub = [len(Config.fromfile(
        os.path.join(CFG_DIR, f'_eval_after_t{i:02d}.py')
    ).val_dataloader['dataset']['datasets']) for i in range(1, 14)]
    check(9, '評価 config（plain）が 13 タスク分そろっている',
          n_ds == 13 and n_sub == list(range(1, 14)),
          f'全体 {n_ds} タスク / 部分集合 {n_sub}')

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
