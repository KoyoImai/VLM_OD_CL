# DitHub (Cappellino et al., NeurIPS 2025) のコア層.
#
# 本ファイルは VLM 物体検出継続学習プロジェクト (exp_021) の新規追加物であり、
# 上流 mmdetection のコードには含まれない。既存ファイルは一切変更していない。
# 登録は exp_021 の config が custom_imports で本モジュールを import することで行う。
# 台帳・設計: experiments/exp_021/implementation_plan.md
# 手法の正本: papers/DitHub_implementation_notes.md
#
# 公式実装 (https://github.com/chiara-cap/DitHub) との対応:
#   DitHubLinear   <-> LinearPool (クラス別 A + 共有 B + warmup A)
#   DitHubState    <-> TaskMemory (singleton)
#   DecomposedMHA  <-> apply_custom_attention + MultiHeadAttention
#     (融合 in_proj を q/k/v の独立 Linear に分解し、LoRA 差し替えを可能にする)

import math
import re
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

LORA_R = 16
LORA_ALPHA = 8
LAMBDA_A = 0.3
LAMBDA_B = 0.7


def canonical_key(class_name: str) -> str:
    """クラス名を ParameterDict のキーに正規化する (小文字化 + 非英数字を _ に)."""
    key = re.sub(r'[^0-9a-z]+', '_', class_name.strip().lower()).strip('_')
    return f'class_{key}'


class DitHubState:
    """フェーズとサンプル別クラス選択を全 DitHubLinear へ配る singleton."""

    def __init__(self) -> None:
        self.train_class_keys: Optional[List[str]] = None
        self.eval_class_keys: Optional[List[str]] = None


STATE = DitHubState()


class DitHubLinear(nn.Linear):
    """凍結された元の Linear + クラス別 A プール + 共有 B (公式 LinearPool 相当).

    self.weight / self.bias は事前学習の重みそのもの (キー名維持・凍結)。
    学習対象は warmup_lora_a / per_class_lora_A / shared_lora_b のみ。
    フェーズは persistent buffer `dithub_per_class_enabled` に保持し、
    checkpoint から評価するときも正しい forward が再現される。
    """

    def __init__(self, in_features: int, out_features: int,
                 class_keys: List[str], bias: bool = True,
                 r: int = LORA_R, lora_alpha: int = LORA_ALPHA) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.r = r
        self.scaling = lora_alpha / r
        self.warmup_lora_a = nn.Parameter(self.weight.new_zeros(
            (r, in_features)))
        self.shared_lora_b = nn.Parameter(self.weight.new_zeros(
            (out_features, r)))
        self.per_class_lora_A = nn.ParameterDict({
            k: nn.Parameter(self.weight.new_zeros((r, in_features)))
            for k in class_keys
        })
        self.register_buffer('dithub_per_class_enabled',
                             torch.zeros((), dtype=torch.uint8))
        self.init_lora()

    @torch.no_grad()
    def init_lora(self) -> None:
        """公式の初期状態: A は kaiming、B は 0 (=> ΔW=0 で θ0 と厳密一致)."""
        nn.init.kaiming_uniform_(self.warmup_lora_a, a=math.sqrt(5))
        for key in self.per_class_lora_A:
            nn.init.kaiming_uniform_(self.per_class_lora_A[key],
                                     a=math.sqrt(5))
        nn.init.zeros_(self.shared_lora_b)

    @torch.no_grad()
    def enable_per_class(self, lambda_a: float = LAMBDA_A,
                         trained_keys: Optional[set] = None) -> None:
        """warmup -> specialization 切替 (公式 enable_per_class / 式3).

        trained_keys に含まれる (過去に実学習された) クラスは
        A <- λ_A·A_wu + (1-λ_A)·A_old の融合、それ以外は A_wu のコピー。
        単発ドメイン学習では trained_keys は常に空。
        """
        trained_keys = trained_keys or set()
        for key in self.per_class_lora_A:
            if key in trained_keys:
                self.per_class_lora_A[key].data = (
                    lambda_a * self.warmup_lora_a.data +
                    (1.0 - lambda_a) * self.per_class_lora_A[key].data)
            else:
                self.per_class_lora_A[key].data.copy_(self.warmup_lora_a.data)
        self.dithub_per_class_enabled.fill_(1)

    def _delta_single(self, x: Tensor, lora_a: Tensor) -> Tensor:
        """全サンプル共通の A を適用した ΔW·x."""
        return F.linear(F.linear(x, lora_a), self.shared_lora_b) * self.scaling

    def _delta_per_sample(self, x: Tensor, keys: List[str]) -> Tensor:
        """サンプル別の A を適用した ΔW·x (公式の bmm 経路)."""
        lora_a = torch.stack([self.per_class_lora_A[k] for k in keys])
        if x.dim() == 2:
            h = torch.bmm(x.unsqueeze(1), lora_a.transpose(1, 2)).squeeze(1)
            return F.linear(h, self.shared_lora_b) * self.scaling
        assert x.dim() == 3, f'unsupported input dim {x.dim()}'
        transposed = False
        if x.shape[0] != len(keys):
            # seq-first (L, bs, E) のケース (テキスト self-attn 内部)
            assert x.shape[1] == len(keys), (
                f'batch mismatch: x={tuple(x.shape)}, classes={len(keys)}')
            x = x.transpose(0, 1)
            transposed = True
        h = torch.bmm(x, lora_a.transpose(1, 2))
        delta = F.linear(h, self.shared_lora_b) * self.scaling
        if transposed:
            delta = delta.transpose(0, 1)
        return delta

    def forward(self, x: Tensor) -> Tensor:
        base = F.linear(x, self.weight, self.bias)
        enabled = bool(self.dithub_per_class_enabled)
        if self.training:
            if not enabled:
                return base + self._delta_single(x, self.warmup_lora_a)
            keys = STATE.train_class_keys
            assert keys is not None, (
                'DitHubState.train_class_keys not set; detector.loss() must '
                'set per-sample classes before forward')
            return base + self._delta_per_sample(x, keys)
        # ---- 評価 ----
        if not enabled:
            # warmup フェーズ中の val: 全プロンプトクラスに warmup A を適用
            # (裁定④。その時点のモデルの実体と一致する意味論)
            return base + self._delta_single(x, self.warmup_lora_a)
        keys = STATE.eval_class_keys or []
        mods = [self.per_class_lora_A[k] for k in keys
                if k in self.per_class_lora_A]
        if not mods:
            return base  # モジュールを持つクラスがプロンプトに無い => θ0 と一致
        composite = torch.stack(
            [self.shared_lora_b @ a for a in mods]).mean(dim=0)
        return base + F.linear(x, composite) * self.scaling


class DecomposedMHA(nn.Module):
    """nn.MultiheadAttention を q/k/v/o の独立 Linear に分解した等価実装.

    mmcv の MultiheadAttention ラッパーは内部 attn を seq-first (L, bs, E) で
    呼ぶため、その呼び出し規約 (query, key, value, attn_mask,
    key_padding_mask) -> (out, weights) を踏襲する。
    checkpoint の in_proj_weight / in_proj_bias / out_proj.* は
    _load_from_state_dict で分解して読み込む (既存 ckpt を無変換でロード可能)。
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.linear_o = nn.Linear(embed_dim, embed_dim)

    @torch.no_grad()
    def load_from_multihead(self, mha: nn.MultiheadAttention) -> None:
        q_w, k_w, v_w = torch.chunk(mha.in_proj_weight, 3, dim=0)
        q_b, k_b, v_b = torch.chunk(mha.in_proj_bias, 3, dim=0)
        self.linear_q.weight.copy_(q_w)
        self.linear_k.weight.copy_(k_w)
        self.linear_v.weight.copy_(v_w)
        self.linear_q.bias.copy_(q_b)
        self.linear_k.bias.copy_(k_b)
        self.linear_v.bias.copy_(v_b)
        self.linear_o.weight.copy_(mha.out_proj.weight)
        self.linear_o.bias.copy_(mha.out_proj.bias)

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args,
                              **kwargs) -> None:
        # 事前学習 ckpt の融合キーを分解キーへ変換してから通常ロードに委ねる
        in_w = state_dict.pop(prefix + 'in_proj_weight', None)
        if in_w is not None:
            in_b = state_dict.pop(prefix + 'in_proj_bias')
            for name, w, b in zip(('q', 'k', 'v'),
                                  torch.chunk(in_w, 3, dim=0),
                                  torch.chunk(in_b, 3, dim=0)):
                state_dict[f'{prefix}linear_{name}.weight'] = w
                state_dict[f'{prefix}linear_{name}.bias'] = b
            out_w = state_dict.pop(prefix + 'out_proj.weight')
            out_b = state_dict.pop(prefix + 'out_proj.bias')
            state_dict[prefix + 'linear_o.weight'] = out_w
            state_dict[prefix + 'linear_o.bias'] = out_b
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True,
                **kwargs):
        # 入力は seq-first (L, bs, E)
        tgt_len, bsz, _ = query.shape
        src_len = key.shape[0]
        q = self.linear_q(query) * (self.head_dim ** -0.5)
        k = self.linear_k(key)
        v = self.linear_v(value)

        def to_heads(t: Tensor, length: int) -> Tensor:
            return t.reshape(length, bsz * self.num_heads,
                             self.head_dim).transpose(0, 1)

        q, k, v = to_heads(q, tgt_len), to_heads(k, src_len), to_heads(
            v, src_len)
        attn = torch.bmm(q, k.transpose(1, 2))  # (bs*H, L, S)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(attn_mask, float('-inf'))
            else:
                attn = attn + attn_mask
        if key_padding_mask is not None:
            mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(
                -1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1,
                                                    src_len)
            attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)
        out = torch.bmm(attn, v)  # (bs*H, L, hd)
        out = out.transpose(0, 1).reshape(tgt_len, bsz, self.embed_dim)
        out = self.linear_o(out)
        return out, None
