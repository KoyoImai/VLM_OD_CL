"""学習済み LoRA チェックポイント（base＋LoRA）を、LoRA を base へマージした plain ckpt に変換する。
各 LoRA 層について  W <- W + (alpha/r) * (B @ A)  を計算し、lora_A/lora_B/lora_scaling を削除。
`lora.decompose_mha` で q/k/v/o に分解した MHA があれば、マージ後に元の
in_proj_weight / in_proj_bias / out_proj.* へ再融合する。
出力は通常の GroundingDINO 構造の state_dict（plain config でそのままロード可能）。

使い方: python merge_lora.py <in_ckpt(base+LoRA)> <out_ckpt(merged plain)>
"""
import sys
import torch


def refuse_decomposed_mha(sd):
    """DecomposedMHA の linear_{q,k,v,o} を nn.MultiheadAttention の形へ戻す。

    LoRA のマージ後に呼ぶこと（マージ前に呼ぶと LoRA が取り残される）。
    """
    tail = '.linear_q.weight'
    prefixes = sorted(k[:-len(tail)] for k in list(sd) if k.endswith(tail))
    for pfx in prefixes:
        ws = [sd.pop(f'{pfx}.linear_{n}.weight') for n in ('q', 'k', 'v')]
        bs = [sd.pop(f'{pfx}.linear_{n}.bias') for n in ('q', 'k', 'v')]
        sd[f'{pfx}.in_proj_weight'] = torch.cat(ws, dim=0)
        sd[f'{pfx}.in_proj_bias'] = torch.cat(bs, dim=0)
        sd[f'{pfx}.out_proj.weight'] = sd.pop(f'{pfx}.linear_o.weight')
        sd[f'{pfx}.out_proj.bias'] = sd.pop(f'{pfx}.linear_o.bias')
    return len(prefixes)


def merge(in_ckpt, out_ckpt):
    ck = torch.load(in_ckpt, map_location='cpu')
    sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck

    prefixes = [k[:-len('.lora_A')] for k in list(sd) if k.endswith('.lora_A')]
    if not prefixes:
        print('[warn] lora_A が見つかりません。すでに plain か、対象外の ckpt です。')
    n = 0
    for pfx in prefixes:
        A = sd[pfx + '.lora_A'].float()                 # (r, in)
        B = sd[pfx + '.lora_B'].float()                 # (out, r)
        scaling = float(sd[pfx + '.lora_scaling'])      # alpha/r
        W = sd[pfx + '.weight']
        Wm = W.float() + scaling * (B @ A)              # (out, in)
        sd[pfx + '.weight'] = Wm.to(W.dtype)
        del sd[pfx + '.lora_A'], sd[pfx + '.lora_B'], sd[pfx + '.lora_scaling']
        n += 1

    n_mha = refuse_decomposed_mha(sd)

    remaining = [k for k in sd if 'lora_' in k]
    if isinstance(ck, dict) and 'state_dict' in ck:
        ck['state_dict'] = sd
        out = ck
    else:
        out = sd
    torch.save(out, out_ckpt)
    print(f'[merge] {n} 層をマージ / MHA {n_mha} 個を再融合 -> {out_ckpt} '
          f'(残存 lora キー: {len(remaining)})')


if __name__ == '__main__':
    merge(sys.argv[1], sys.argv[2])
