"""LoRA 線形層（既存 nn.Linear を置換）。
- 元の weight/bias を **同じキー名**で保持（事前学習 ckpt をそのままロード可能）。
- 追加する低ランク行列は lora_A / lora_B（これだけ学習）。scaling=alpha/r は buffer に保存（マージ時に参照）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        # 元の重み/バイアスを同名で保持（凍結）
        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        # LoRA 本体（学習対象）。B=0 初期化なので初期出力は元と一致
        self.r = int(r)
        self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
        self.register_buffer('lora_scaling', torch.tensor(float(alpha) / float(r)))
        self.merged = False
        self.init_lora()

    @torch.no_grad()
    def init_lora(self):
        """LoRA の初期状態: A は kaiming、B は 0（=> ΔW=0 で θ0 と厳密一致）。

        `__init__` から呼ぶほか、`GroundingDINOLoRA.init_weights` からも呼ぶ。
        `DINO.init_weights` が encoder/decoder 配下の dim>1 パラメータを一括で
        xavier 初期化してしまい（mmdet/models/detectors/dino.py:72-75）、
        lora_A/lora_B もその対象に入るため、後から復元する必要がある。
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if not self.merged:
            out = out + self.lora_scaling * ((x @ self.lora_A.t()) @ self.lora_B.t())
        return out

    @torch.no_grad()
    def merge(self):
        """LoRA を base 重みへ統合（マージ方式・評価用）。"""
        if self.merged:
            return
        self.weight.add_(self.lora_scaling * (self.lora_B @ self.lora_A))
        self.merged = True
