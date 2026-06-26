"""学習済み LoRA チェックポイント（base＋LoRA）を、LoRA を base へマージした plain ckpt に変換する。
各 LoRA 層について  W <- W + (alpha/r) * (B @ A)  を計算し、lora_A/lora_B/lora_scaling を削除。
出力は通常の GroundingDINO 構造の state_dict（plain config でそのままロード可能）。

使い方: python merge_lora.py <in_ckpt(base+LoRA)> <out_ckpt(merged plain)>
"""
import sys
import torch


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

    remaining = [k for k in sd if 'lora_' in k]
    if isinstance(ck, dict) and 'state_dict' in ck:
        ck['state_dict'] = sd
        out = ck
    else:
        out = sd
    torch.save(out, out_ckpt)
    print(f'[merge] {n} 層をマージ -> {out_ckpt} (残存 lora キー: {len(remaining)})')


if __name__ == '__main__':
    merge(sys.argv[1], sys.argv[2])
