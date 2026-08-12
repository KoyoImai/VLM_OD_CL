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
    """クラス名を ParameterDict のキーに正規化する (小文字化 + 非英数字を _ に).

    学習時のクラス名は pipeline 側の ``clean_name`` (括弧内を除去) を通った後の
    文字列で、評価時は metainfo の生のクラス名である。両者で同じキーになるよう、
    ここでも括弧の除去を行う (2026-08-10 修正)。
    """
    name = re.sub(r'\(.*\)', '', class_name)
    key = re.sub(r'[^0-9a-z]+', '_', name.strip().lower()).strip('_')
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
        # 公式 LinearPool.forward の do_warmup_a は
        #   self.training and num_classes_task > 1 and not self.per_class
        # であり、クラスが 1 つしかないタスクでは warmup を行わず最初から
        # そのクラスの A を学習する (ODinW-13 では 13 タスク中 6 タスクが該当)。
        self.single_class_task = len(class_keys) == 1
        # ライブラリに実際に保存されている (= 学習済みの) クラスキー。
        # checkpoint の per_class_lora_A のキー集合から復元する。空なら
        # 「全スロットが有効」として扱う (単発学習・学習中の挙動は不変)。
        self.library_class_keys: set = set()
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
            elif not self.single_class_task:
                self.per_class_lora_A[key].data.copy_(self.warmup_lora_a.data)
            # 単一クラスタスクで未学習の場合は何もしない。公式 enable_per_class は
            # len(classes)==1 のとき counter>0 でなければ A に触れず per_class を
            # 立てるだけで、iter 0 から学習してきた A をそのまま引き継ぐ。
        self.dithub_per_class_enabled.fill_(1)

    @torch.no_grad()
    def reinit_warmup_a(self) -> None:
        """warmup_lora_a を kaiming で引き直す (タスク開始時).

        公式はタスクごとにモデルを θ0 から作り直し、per_class_lora_A と
        shared_lora_b だけを TaskMemory から復元する
        (`lora_pool.py` LinearPool.__init__、`main.py:302` load_model、
        `train.init_checkpoint` は空)。**warmup_lora_a は復元されず、毎タスク
        kaiming で初期化される**。本実装は前タスクのライブラリ checkpoint を
        load_from で読むため、明示的に引き直さないと持ち越してしまう。
        """
        nn.init.kaiming_uniform_(self.warmup_lora_a, a=math.sqrt(5))

    @torch.no_grad()
    def set_per_class_enabled(self, enabled: bool) -> None:
        """フェーズフラグだけを設定する (A の融合・コピーは行わない).

        `dithub_per_class_enabled` は persistent buffer なので checkpoint に
        載る。逐次学習で前タスクのライブラリを load_from すると 1 が読み込まれ、
        次タスクが warmup を飛ばして specialization から始まってしまう
        (2026-08-10 に exp_023 のログから確認。修正前は t>=2 で warmup_lora_a
        の勾配が完全に消え、iter=max/2 の切替で学習済み A が未学習の
        warmup_lora_a で上書きされていた)。学習開始時に DitHubPhaseHook が
        本メソッドで明示的に初期化する。
        """
        self.dithub_per_class_enabled.fill_(1 if enabled else 0)

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

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args,
                              **kwargs) -> None:
        # checkpoint に含まれる per_class_lora_A のキー = ライブラリに保存済み
        # (=学習済み) のクラス。評価時にこれで絞り込み、一度も学習していない
        # スロット (kaiming 初期値のまま) が合成 ΔW に混ざるのを防ぐ。
        loaded = {
            k[len(prefix + 'per_class_lora_A.'):]
            for k in state_dict if k.startswith(prefix + 'per_class_lora_A.')
        }
        if loaded:
            self.library_class_keys = loaded
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        base = F.linear(x, self.weight, self.bias)
        enabled = bool(self.dithub_per_class_enabled)
        if self.training:
            if not enabled and not self.single_class_task:
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
        pool = self.library_class_keys or set(self.per_class_lora_A)
        mods = [self.per_class_lora_A[k] for k in keys if k in pool]
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
        # 入力は seq-first (L, bs, E)。
        # 射影の間だけ batch-first (bs, L, E) に直してから Linear を通す。
        # Linear は最終次元にのみ作用するので出力は数学的に同一だが、
        # DitHubLinear._delta_per_sample が「先頭次元 = バッチ」を形状から
        # 推定するため、seq_len == batch_size のときに seq-first を
        # batch-first と誤認する危険を消せる (2026-08-10 修正)。
        tgt_len, bsz, _ = query.shape
        src_len = key.shape[0]

        def proj(linear: nn.Linear, t: Tensor) -> Tensor:
            return linear(t.transpose(0, 1)).transpose(0, 1)

        q = proj(self.linear_q, query) * (self.head_dim ** -0.5)
        k = proj(self.linear_k, key)
        v = proj(self.linear_v, value)

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
        out = proj(self.linear_o, out)
        return out, None
