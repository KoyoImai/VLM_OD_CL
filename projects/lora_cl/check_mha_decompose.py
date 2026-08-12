"""Feature Enhancer のテキスト側 self-attention への LoRA 挿入を検証する。

使い方:
    python projects/lora_cl/check_mha_decompose.py [config]

既定 config は experiments/exp_008/configs/underwater_lora.py を土台に
`lora.decompose_mha` と encoder テキスト側の include_patterns を上書きしたもの。

検証項目
  1. 対象の MHA が DecomposedMHA へ差し替わっている
  2. テキスト self-attention の q/k/v/o に LoRA が入っている
  3. θ0 を無変換でロードできる（in_proj_weight の分割ロードが効く）
  4. DecomposedMHA 単体の出力が nn.MultiheadAttention と数値一致
  5. init_weights 後も B=0（ΔW=0）
  6. 検出器全体の推論結果が plain GroundingDINO と一致（θ0 等価）
  7. 実データ 1 step で分解層の LoRA に勾配が流れ、base は不変
  8. merge_lora + MHA 再融合した ckpt が plain GroundingDINO に
     missing/unexpected 0 でロードできる
  9. マージ後 plain モデルの推論結果が LoRA モデルと一致
"""
import copy
import os
import sys
import tempfile

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from mmengine.utils import import_modules_from_strings

from mmdet.registry import MODELS

CKPT = ('/root/.cache/torch/hub/checkpoints/grounding_dino_swin-t_pretrain_'
        'obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth')
TEXT_MHA = r'^encoder\.text_layers\.\d+\.self_attn\.attn$'

results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


def build(cfg, lora=None):
    # 検出器の __init__ が bbox_head 側へ share_pred_layer 等を書き込むため、
    # 同じ cfg.model を再利用すると 2 回目の build が落ちる。毎回 deepcopy する。
    m_cfg = copy.deepcopy(cfg.model)
    if lora is None:
        m_cfg.pop('lora', None)
        m_cfg['type'] = 'GroundingDINO'
    else:
        m_cfg['type'] = 'GroundingDINOLoRA'
        m_cfg['lora'] = lora
    model = MODELS.build(m_cfg)
    model.init_weights()
    _load_checkpoint_to_model(model, _load_checkpoint(CKPT, map_location='cpu'),
                              strict=False)
    return model


@torch.no_grad()
def predict_scores(model, data):
    model.eval()
    out = model.predict(data['inputs'], data['data_samples'], rescale=True)
    return (torch.cat([o.pred_instances.scores for o in out]),
            torch.cat([o.pred_instances.bboxes for o in out]))


if __name__ == '__main__':
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else \
        'experiments/exp_008/configs/underwater_lora.py'
    init_default_scope('mmdet')
    cfg = Config.fromfile(cfg_path)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg['custom_imports'])
    from mmdet.models.layers.dithub_layers import DecomposedMHA

    from projects.lora_cl.lora_layers import LoRALinear
    from projects.lora_cl.merge_lora import merge

    LORA = dict(r=16, alpha=16,
                decompose_mha=[TEXT_MHA],
                include_patterns=[
                    r'^encoder\.text_layers\.\d+\.self_attn\.attn\.linear_',
                    r'^encoder\.text_layers\.\d+\.ffn\.'])
    print(f'config: {cfg_path}\nlora: {LORA}\n')

    model = build(cfg, LORA)

    # --- 1. 差し替え -------------------------------------------------------
    dec = [n for n, m in model.named_modules() if isinstance(m, DecomposedMHA)]
    left = [n for n, m in model.named_modules()
            if isinstance(m, nn.MultiheadAttention) and 'text_layers' in n]
    check(1, 'テキスト self-attn が DecomposedMHA へ差し替わった',
          len(dec) == 6 and not left, f'分解 {len(dec)} 個 / 未分解 {len(left)} 個')

    # --- 2. LoRA 対象 -------------------------------------------------------
    lo = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    qkvo = [n for n in lo if '.self_attn.attn.linear_' in n]
    check(2, 'テキスト self-attn の q/k/v/o に LoRA が入った',
          len(qkvo) == 24 and len(lo) == 36,
          f'q/k/v/o {len(qkvo)} 層 / LoRA 全体 {len(lo)} 層')

    # --- 3. θ0 の無変換ロード ----------------------------------------------
    ck = _load_checkpoint(CKPT, map_location='cpu')
    src = ck['state_dict'] if 'state_dict' in ck else ck
    sd = model.state_dict()
    plain = build(cfg, None)
    psd = plain.state_dict()
    mismatch = []
    for i in range(6):
        p = f'encoder.text_layers.{i}.self_attn.attn.'
        q, k, v = torch.chunk(src[p + 'in_proj_weight'], 3, dim=0)
        for nm, ref in (('q', q), ('k', k), ('v', v)):
            if not torch.equal(sd[p + f'linear_{nm}.weight'], ref):
                mismatch.append(p + nm)
        if not torch.equal(sd[p + 'linear_o.weight'], src[p + 'out_proj.weight']):
            mismatch.append(p + 'o')
    check(3, 'θ0 の in_proj_weight が q/k/v へ正しく分割ロードされた',
          not mismatch, f'照合 24 テンソル / 不一致 {len(mismatch)}')

    # --- 4. DecomposedMHA 単体の等価性 --------------------------------------
    torch.manual_seed(0)
    x = torch.randn(11, 3, 256)          # (seq, batch, embed) = seq-first
    pad = torch.zeros(3, 11, dtype=torch.bool)
    pad[1, 8:] = True
    d_mha = model.encoder.text_layers[0].self_attn.attn
    o_mha = plain.encoder.text_layers[0].self_attn.attn
    d_mha.eval(); o_mha.eval()
    with torch.no_grad():
        yd = d_mha(query=x, key=x, value=x, attn_mask=None, key_padding_mask=pad)[0]
        yo = o_mha(query=x, key=x, value=x, attn_mask=None, key_padding_mask=pad)[0]
    diff = float((yd - yo).abs().max())
    check(4, 'DecomposedMHA が nn.MultiheadAttention と数値一致',
          diff < 1e-5, f'max|Δ|={diff:.3e}')

    # --- 5. init_weights 後の B=0 -------------------------------------------
    dmax = max(float((float(m.lora_scaling) * (m.lora_B @ m.lora_A)).abs().max())
               for _, m in model.named_modules() if isinstance(m, LoRALinear))
    check(5, 'init_weights + θ0 ロード後も ΔW=0', dmax == 0.0, f'max|ΔW|={dmax:.3e}')

    # --- 6. 検出器全体の θ0 等価性 ------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    val_loader = Runner.build_dataloader(cfg.val_dataloader)
    raw = next(iter(val_loader))
    model = model.to(device)
    plain = plain.to(device)
    d_val = model.data_preprocessor(raw, training=False)
    s_lora, b_lora = predict_scores(model, d_val)
    s_plain, b_plain = predict_scores(plain, d_val)
    ds = float((s_lora - s_plain).abs().max())
    db = float((b_lora - b_plain).abs().max())
    check(6, '検出器全体の推論が plain GroundingDINO と一致',
          ds < 1e-4 and db < 1e-3,
          f'max|Δscore|={ds:.3e}, max|Δbbox|={db:.3e}, 検出数={len(s_lora)}')

    # --- 7. 実データ 1 step --------------------------------------------------
    from mmengine.optim import build_optim_wrapper
    ow = build_optim_wrapper(model, cfg.optim_wrapper)
    model.train()
    train_loader = Runner.build_dataloader(cfg.train_dataloader)
    d_tr = model.data_preprocessor(next(iter(train_loader)), training=True)
    base_before = {n: p.detach().clone()
                   for n, p in model.named_parameters() if not p.requires_grad}
    losses = model.loss(d_tr['inputs'], d_tr['data_samples'])
    sum(v.sum() for v in losses.values()
        if isinstance(v, torch.Tensor)).backward()
    # q/k は attention logits 経由でしか損失に効かず、θ0 では勾配が v/o より
    # 2〜4 桁小さい（DitHub 実装でも同様: q~1e-4, k~1e-3 に対し v/o~1e-1）。
    # バッチによっては float32 で 0 に落ちるため、判定は「grad テンソルが
    # 存在すること」と「v/o が全層非零であること」に置く。
    gr = {n: p.grad for n, p in model.named_parameters()
          if '.self_attn.attn.linear_' in n and n.endswith('lora_B')}
    none_g = [n for n, g in gr.items() if g is None]
    vo = [n for n in gr if '_v.' in n or '_o.' in n]
    vo_zero = [n for n in vo if float(gr[n].abs().max()) == 0]
    qk_max = max(float(gr[n].abs().max()) for n in gr
                 if '_q.' in n or '_k.' in n)
    dbase = max(float((p.detach() - base_before[n]).abs().max())
                for n, p in model.named_parameters() if n in base_before)
    ow.optimizer.step()
    check(7, '分解層の LoRA に勾配が流れ base は不変',
          not none_g and not vo_zero and dbase == 0.0,
          f'grad=None {len(none_g)}/24, v/o の非零 {len(vo) - len(vo_zero)}/12, '
          f'q/k の最大勾配 {qk_max:.2e}, max|Δbase|={dbase:.3e}')

    # --- 8/9. マージ＋再融合 --------------------------------------------------
    tmpd = tempfile.mkdtemp()
    src_ck = os.path.join(tmpd, 'lora.pth')
    dst_ck = os.path.join(tmpd, 'merged.pth')
    torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}},
               src_ck)
    merge(src_ck, dst_ck)

    p2_cfg = copy.deepcopy(cfg.model)
    p2_cfg.pop('lora', None)
    p2_cfg['type'] = 'GroundingDINO'
    plain2 = MODELS.build(p2_cfg)
    info = plain2.load_state_dict(
        torch.load(dst_ck, map_location='cpu')['state_dict'], strict=False)
    check(8, 'マージ後 ckpt が plain GroundingDINO にそのまま載る',
          not info.missing_keys and not info.unexpected_keys,
          f'missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}')

    plain2 = plain2.to(device)
    model.eval()

    # 9. MHA 単位の等価性（学習後の LoRA 付き DecomposedMHA vs マージ再融合後の MHA）。
    #    学習済み LoRA が base へ正しく畳まれ、q/k/v/o が in_proj へ戻ったかを直接見る。
    torch.manual_seed(1)
    xx = torch.randn(11, 3, 256, device=device)
    pp = torch.zeros(3, 11, dtype=torch.bool, device=device)
    pp[1, 8:] = True
    worst = 0.0
    with torch.no_grad():
        for i in range(6):
            a = model.encoder.text_layers[i].self_attn.attn.eval()
            b = plain2.encoder.text_layers[i].self_attn.attn.eval()
            ya = a(query=xx, key=xx, value=xx, attn_mask=None,
                   key_padding_mask=pp)[0]
            yb = b(query=xx, key=xx, value=xx, attn_mask=None,
                   key_padding_mask=pp)[0]
            worst = max(worst, float((ya - yb).abs().max()))
    check(9, 'マージ＋再融合後の MHA が学習後 LoRA と数値一致',
          worst < 1e-4, f'6 層の max|Δ|={worst:.3e}')

    # 10. 検出器全体（スコアは降順ソート済みなので順序安定。bbox は同点付近で
    #     並びが入れ替わるため要素比較しない）。
    s_m, _ = predict_scores(plain2, d_val)
    s_l2, _ = predict_scores(model, d_val)
    ds2 = float((s_m - s_l2).abs().max())
    check(10, 'マージ後 plain モデルの検出スコアが LoRA モデルと一致',
          ds2 < 1e-3, f'max|Δscore|={ds2:.3e} / 検出数={len(s_m)}')

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
