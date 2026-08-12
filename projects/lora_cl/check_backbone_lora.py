"""image backbone(Swin-T) と text backbone(BERT) への LoRA 挿入を検証する。

使い方:
    python projects/lora_cl/check_backbone_lora.py [config]

`lora.exclude_components` で既定除外（backbone / language_model）を上書きできる
ことと、その状態でも学習・マージ・評価の経路が壊れないことを確認する。

検証項目
  1. 既定（exclude_components 未指定）の挙動が従来どおり
  2. exclude_components=[] で Swin 51 層・BERT 72 層が対象に入る
  3. exclude_components=['language_model'] で Swin だけ入る
  4. include に除外中の構成要素を書いたら build 時に落ちる
  5. base 重みが θ0 と一致（Swin/BERT を LoRALinear に置換しても壊れない）
  6. init_weights + θ0 ロード後も ΔW=0
  7. 検出器全体の推論が plain GroundingDINO と一致（θ0 等価）
  8. 実データ 1 step で Swin/BERT の LoRA に勾配が流れ、base は不変
  9. merge 後の ckpt が plain GroundingDINO に missing/unexpected 0 で載る
 10. マージ後 plain モデルの検出スコアが LoRA モデルと一致
"""
import copy
import os
import sys
import tempfile

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from mmengine.utils import import_modules_from_strings

from mmdet.registry import MODELS

CKPT = ('/root/.cache/torch/hub/checkpoints/grounding_dino_swin-t_pretrain_'
        'obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth')

results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


def build(cfg, lora=None, load=True):
    m_cfg = copy.deepcopy(cfg.model)
    if lora is None:
        m_cfg.pop('lora', None)
        m_cfg['type'] = 'GroundingDINO'
    else:
        m_cfg['type'] = 'GroundingDINOLoRA'
        m_cfg['lora'] = lora
    model = MODELS.build(m_cfg)
    model.init_weights()
    if load:
        _load_checkpoint_to_model(
            model, _load_checkpoint(CKPT, map_location='cpu'), strict=False)
    return model


@torch.no_grad()
def predict_scores(model, data):
    model.eval()
    out = model.predict(data['inputs'], data['data_samples'], rescale=True)
    return torch.cat([o.pred_instances.scores for o in out])


if __name__ == '__main__':
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else \
        'experiments/exp_008/configs/underwater_lora.py'
    init_default_scope('mmdet')
    cfg = Config.fromfile(cfg_path)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg['custom_imports'])
    from projects.lora_cl.grounding_dino_lora import GroundingDINOLoRA
    from projects.lora_cl.lora_layers import LoRALinear
    from projects.lora_cl.merge_lora import merge

    from collections import Counter

    def count(lora):
        """モデルを組まずに _match だけで数える（安価）。"""
        import torch.nn as nn
        hit = []
        for n, m in probe.named_modules():
            if not GroundingDINOLoRA._match(n, m, lora):
                continue
            if isinstance(dict(probe.named_modules()).get(
                    n.rsplit('.', 1)[0]), nn.MultiheadAttention):
                continue
            hit.append(n)
        return Counter(n.split('.')[0] for n in hit)

    probe = build(cfg, None, load=False)

    # --- 1. 既定の挙動 ------------------------------------------------------
    base_cfg = dict(r=16, alpha=16,
                    include=['encoder', 'decoder', 'bbox_head'])
    c = count(base_cfg)
    check(1, '既定（exclude_components 未指定）は従来どおり',
          sum(c.values()) == 143 and 'backbone' not in c
          and 'language_model' not in c, f'{sum(c.values())} 層 {dict(c)}')

    # --- 2. 両方を対象に -----------------------------------------------------
    all_cfg = dict(r=16, alpha=16, exclude_components=[],
                   include=['backbone', 'language_model'])
    c2 = count(all_cfg)
    check(2, 'exclude_components=[] で Swin と BERT が対象に入る',
          c2.get('backbone') == 51 and c2.get('language_model') == 72,
          f'backbone={c2.get("backbone")} language_model={c2.get("language_model")}')

    # --- 3. Swin だけ --------------------------------------------------------
    c3 = count(dict(r=16, alpha=16, exclude_components=['language_model'],
                    include=['backbone']))
    check(3, "exclude_components=['language_model'] で Swin だけ入る",
          c3.get('backbone') == 51 and 'language_model' not in c3,
          f'{dict(c3)}')

    # --- 4. 取り違えを build 時に落とす --------------------------------------
    try:
        build(cfg, dict(r=16, alpha=16, include=['backbone']), load=False)
        ok, msg = False, '例外が出なかった'
    except AssertionError as e:
        ok, msg = True, str(e).split('。')[0]
    check(4, 'include と exclude_components の矛盾を build 時に検出', ok, msg)

    # --- 5〜10. Swin + BERT を対象にした実機確認 -----------------------------
    LORA = dict(r=16, alpha=16, exclude_components=[],
                include=['backbone', 'language_model'])
    print(f'\n--- 実機確認: lora={LORA} ---')
    model = build(cfg, LORA)
    lo = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    ck = _load_checkpoint(CKPT, map_location='cpu')
    src = ck['state_dict'] if 'state_dict' in ck else ck
    sd = model.state_dict()
    bad = [k for k, v in src.items()
           if k in sd and sd[k].shape == v.shape
           and not torch.equal(sd[k].cpu(), v.cpu())]
    check(5, 'base 重みが θ0 と一致', not bad,
          f'LoRA {len(lo)} 層 / 照合 {sum(1 for k in src if k in sd)} キー / 不一致 {len(bad)}')

    dmax = max(float((float(m.lora_scaling) * (m.lora_B @ m.lora_A)).abs().max())
               for _, m in model.named_modules() if isinstance(m, LoRALinear))
    check(6, 'init_weights + θ0 ロード後も ΔW=0', dmax == 0.0, f'max|ΔW|={dmax:.3e}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plain = build(cfg, None).to(device)
    model = model.to(device)
    d_val = model.data_preprocessor(
        next(iter(Runner.build_dataloader(cfg.val_dataloader))), training=False)
    s_l = predict_scores(model, d_val)
    s_p = predict_scores(plain, d_val)
    ds = float((s_l - s_p).abs().max())
    check(7, '検出器全体の推論が plain GroundingDINO と一致', ds == 0.0,
          f'max|Δscore|={ds:.3e} / 検出数={len(s_l)}')

    from mmengine.optim import build_optim_wrapper
    ow = build_optim_wrapper(model, cfg.optim_wrapper)
    model.train()
    d_tr = model.data_preprocessor(
        next(iter(Runner.build_dataloader(cfg.train_dataloader))), training=True)
    base_before = {n: p.detach().clone()
                   for n, p in model.named_parameters() if not p.requires_grad}
    losses = model.loss(d_tr['inputs'], d_tr['data_samples'])
    sum(v.sum() for v in losses.values()
        if isinstance(v, torch.Tensor)).backward()
    gB = {c: sum(1 for n, p in model.named_parameters()
                 if n.startswith(c) and n.endswith('lora_B')
                 and p.grad is not None and float(p.grad.abs().max()) > 0)
          for c in ('backbone', 'language_model')}
    dbase = max(float((p.detach() - base_before[n]).abs().max())
                for n, p in model.named_parameters() if n in base_before)
    ow.optimizer.step()
    check(8, 'Swin/BERT の LoRA に勾配が流れ base は不変',
          gB['backbone'] == 51 and gB['language_model'] == 72 and dbase == 0.0,
          f'非零勾配 backbone={gB["backbone"]}/51 '
          f'language_model={gB["language_model"]}/72, max|Δbase|={dbase:.3e}')

    tmpd = tempfile.mkdtemp()
    src_ck, dst_ck = os.path.join(tmpd, 'l.pth'), os.path.join(tmpd, 'm.pth')
    torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}},
               src_ck)
    merge(src_ck, dst_ck)
    p2_cfg = copy.deepcopy(cfg.model)
    p2_cfg.pop('lora', None)
    p2_cfg['type'] = 'GroundingDINO'
    plain2 = MODELS.build(p2_cfg)
    info = plain2.load_state_dict(
        torch.load(dst_ck, map_location='cpu')['state_dict'], strict=False)
    check(9, 'マージ後 ckpt が plain GroundingDINO にそのまま載る',
          not info.missing_keys and not info.unexpected_keys,
          f'missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}')

    plain2 = plain2.to(device)
    model.eval()
    ds2 = float((predict_scores(plain2, d_val)
                 - predict_scores(model, d_val)).abs().max())
    check(10, 'マージ後 plain モデルの検出スコアが LoRA モデルと一致',
          ds2 < 1e-3, f'max|Δscore|={ds2:.3e}')

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
