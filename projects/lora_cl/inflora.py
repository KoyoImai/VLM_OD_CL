"""InfLoRA（CVPR 2024, liangyanshuo/InfLoRA）の MM-Grounding DINO 実装。

exp_042 design.md §2.3。公式実装から持ち込むのは手法の核のみ:
    - 各タスクの学習前に挿入層の入力共分散（Σ x^T x / n）を蓄積し、DualGPM の
      メモリで射影してから SVD の上位 r 主成分で A（次元削減行列）を設計・固定する
    - 学習するのは B（ゼロ初期化）のみ
    - タスク終了時に共分散を再収集して DualGPM メモリを更新する
    - ΔW = B·A（スケーリング 1 = alpha=r で使う。公式は係数なしで A に 1/√3）
分類固有の要素（タスク別分類器・local CE・タスク ID 推定）は持ち込まない。

挿入箇所の選択・MHA 分解・base 凍結・タスク後マージ（merge_lora.py。lora_scaling=1 で
ΔW=B·A になる）は projects/lora_cl の既存機構をそのまま使う。

config 例:
    model = dict(
        type='GroundingDINOInfLoRA',
        use_dn=False,
        bbox_head=dict(type='NoDNGroundingDINOHead'),
        lora=dict(r=16, alpha=16,          # alpha=r → scaling=1（公式の ΔW=B·A）
                  exclude_components=[],
                  decompose_mha=[r'^encoder\\.text_layers\\.\\d+\\.self_attn\\.attn$'],
                  include=['backbone', 'language_model', 'text_feat_map', 'encoder']),
        inflora=dict(design_path=None))    # inflora_prepare.py の出力。ドライバが上書き
"""
import numpy as np
import torch
import torch.nn as nn
from mmengine.logging import print_log

from mmdet.registry import MODELS

from .grounding_dino_lora import GroundingDINOLoRA
from .lora_layers import LoRALinear


class InfLoRALinear(LoRALinear):
    """A を設計値に固定し B のみ学習する LoRA 層＋入力共分散の蓄積。

    共分散（cov）は公式の cur_matrix に対応する running mean（(d_in, d_in)）。
    プレーンな属性として持ち、state_dict / checkpoint には入らない。
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__(base, r, alpha)
        self.collect_input = False
        self.cov = None
        self.n_cov = 0

    def forward(self, x):
        if self.collect_input and not self.merged:
            with torch.no_grad():
                x2 = x.detach().float().reshape(-1, self.in_features)
                gram = x2.t() @ x2
                if self.cov is None:
                    self.cov = torch.zeros(
                        self.in_features, self.in_features,
                        dtype=torch.float32, device=x2.device)
                # 公式 vit_inflora.py と同じ running mean
                n_new = self.n_cov + x2.shape[0]
                self.cov = (self.cov * self.n_cov + gram) / n_new
                self.n_cov = n_new
        return super().forward(x)

    def pop_cov(self):
        cov, n = self.cov, self.n_cov
        self.cov, self.n_cov = None, 0
        return cov, n

    @torch.no_grad()
    def set_design(self, a):
        """設計した A を固定し、B をゼロから学習可能にする。"""
        a = torch.as_tensor(np.asarray(a), dtype=self.lora_A.dtype,
                            device=self.lora_A.device)
        assert a.shape == self.lora_A.shape, \
            f'A の形が不一致: {tuple(a.shape)} != {tuple(self.lora_A.shape)}'
        self.lora_A.data.copy_(a)
        nn.init.zeros_(self.lora_B)
        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(True)


@MODELS.register_module()
class GroundingDINOInfLoRA(GroundingDINOLoRA):
    lora_cls = InfLoRALinear

    def __init__(self, *args, lora=None, inflora=None, **kwargs):
        super().__init__(*args, lora=lora, **kwargs)
        inflora = inflora or {}
        self.inflora_design_path = inflora.get('design_path')
        self._inflora_design = None
        if self.inflora_design_path:
            d = torch.load(self.inflora_design_path, map_location='cpu')
            self._inflora_design = d['lora_A']
        self._apply_inflora_mode()

    # ---- InfLoRA 層の列挙（named_modules 順で安定）--------------------------
    def inflora_layers(self):
        return [(n, m) for n, m in self.named_modules()
                if isinstance(m, InfLoRALinear)]

    def _apply_inflora_mode(self):
        """A の設計値の適用（あれば）と、A 凍結・B のみ学習の徹底。"""
        layers = self.inflora_layers()
        if self._inflora_design is not None:
            names = [n for n, _ in layers]
            missing = [n for n in names if n not in self._inflora_design]
            extra = [k for k in self._inflora_design if k not in names]
            assert not missing and not extra, (
                f'設計ファイルと挿入層が食い違う: missing={missing[:3]} '
                f'extra={extra[:3]}')
            for n, m in layers:
                m.set_design(self._inflora_design[n])
            print_log(f'[InfLoRA] 設計 A を {len(layers)} 層へ適用: '
                      f'{self.inflora_design_path}', logger='current')
        else:
            for _, m in layers:
                m.lora_A.requires_grad_(False)
            print_log('[InfLoRA] 設計ファイル無し（共分散収集用の構成。'
                      'B=0 のため forward は base と一致）', logger='current')

    def init_weights(self):
        """DINO.init_weights の xavier 一括初期化から設計 A を復元する。

        親クラス（GroundingDINOLoRA.init_weights）が LoRA を kaiming/zero に戻し
        _freeze_base で lora_A/lora_B を学習可にするため、その後に設計値の再適用と
        A の凍結をやり直す。
        """
        super().init_weights()
        self._apply_inflora_mode()

    # ---- 入力共分散の収集 ---------------------------------------------------
    def set_input_collection(self, flag: bool):
        for _, m in self.inflora_layers():
            m.collect_input = bool(flag)

    def pop_covariances(self):
        """収集済みの共分散 {層名: (d_in, d_in)} を取り出して内部をリセットする。"""
        out = {}
        for n, m in self.inflora_layers():
            cov, cnt = m.pop_cov()
            assert cov is not None and cnt > 0, f'{n}: 共分散が未収集'
            out[n] = cov.cpu()
        return out
