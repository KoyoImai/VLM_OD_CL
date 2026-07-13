# DitHub 用検出器: MM-Grounding DINO + クラス別 LoRA プール.
#
# 本ファイルは VLM 物体検出継続学習プロジェクト (exp_021) の新規追加物であり、
# 上流 mmdetection のコードには含まれない。既存ファイルは一切変更していない。
# 登録は exp_021 の config が custom_imports で本モジュールを import することで行う。
# 台帳・設計: experiments/exp_021/implementation_plan.md
#
# 変更点:
#   1. encoder の text_layers の融合 attention を DecomposedMHA へ差し替え
#      (q/k/v/o を独立 Linear 化。数学的に等価、ckpt は無変換ロード可)
#   2. 公式の規則 (out_features >= 128 の Linear、除外リスト付き) に該当する層を
#      DitHubLinear へ差し替え (実質 encoder 全体 + memory_trans_fc)
#   3. loss()/predict() で DitHubState にクラス選択を設定
#   4. LoRA 以外の全パラメータを凍結

import random

import torch
from torch import nn

from mmdet.registry import MODELS
from mmdet.models.layers.dithub_layers import (STATE, DecomposedMHA,
                                               DitHubLinear, canonical_key)
from .grounding_dino import GroundingDINO

# 公式 get_lora_modules の除外規則 ('transformer.dec / bert / backbone /
# feat_map') を mmdet のモジュール名に写像したもの。memory_trans_fc は公式
# コード同様に対象に含まれる (implementation_plan.md 裁定①)。
EXCLUDE_PREFIXES = ('backbone', 'language_model', 'text_feat_map', 'neck',
                    'decoder', 'bbox_head', 'query_embedding',
                    'dn_query_generator', 'memory_trans_norm',
                    'data_preprocessor')
OUT_MIN = 128


@MODELS.register_module()
class DitHubGroundingDINO(GroundingDINO):
    """MM-Grounding DINO に DitHub (クラス別 LoRA ライブラリ) を組み込んだ検出器.

    Args:
        dithub_classes (list[str]): 当該タスク (ドメイン) のクラス名一覧。
            クラス別 A の確保に使う。
        lora_r (int): LoRA rank。公式既定 16。
        lora_alpha (int): LoRA alpha。公式既定 8 (scaling = alpha/r = 0.5)。
        lambda_a (float): 式3 の融合係数。公式既定 0.3 (単発学習では不活性)。
    """

    def __init__(self, *args, dithub_classes, lora_r: int = 16,
                 lora_alpha: int = 8, lambda_a: float = 0.3,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert dithub_classes, 'dithub_classes must be a non-empty list'
        self.dithub_class_keys = [canonical_key(c) for c in dithub_classes]
        self.lambda_a = lambda_a
        self._convert_text_attention()
        self.dithub_layer_names = self._apply_dithub(lora_r, lora_alpha)
        self._freeze_base()

    # ---- 構築 ----------------------------------------------------------

    def _convert_text_attention(self) -> None:
        """text_layers の nn.MultiheadAttention を q/k/v/o 分解型へ差し替える."""
        for layer in self.encoder.text_layers:
            old = layer.self_attn.attn
            assert isinstance(old, nn.MultiheadAttention)
            new = DecomposedMHA(old.embed_dim, old.num_heads, old.dropout)
            new.load_from_multihead(old)
            layer.self_attn.attn = new

    def _apply_dithub(self, lora_r: int, lora_alpha: int) -> list:
        """公式規則に該当する Linear を DitHubLinear へ差し替える."""
        replaced = []
        targets = []
        for m_name, module in self.named_modules():
            for c_name, child in module.named_children():
                qualified = f'{m_name}.{c_name}' if m_name else c_name
                if not isinstance(child, nn.Linear):
                    continue
                if isinstance(child, DitHubLinear):
                    continue
                if child.out_features < OUT_MIN:
                    continue
                if any(qualified.startswith(p) for p in EXCLUDE_PREFIXES):
                    continue
                targets.append((module, c_name, qualified, child))
        for module, c_name, qualified, child in targets:
            new = DitHubLinear(
                child.in_features,
                child.out_features,
                class_keys=self.dithub_class_keys,
                bias=child.bias is not None,
                r=lora_r,
                lora_alpha=lora_alpha)
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None:
                    new.bias.copy_(child.bias)
            setattr(module, c_name, new)
            replaced.append(qualified)
        assert replaced, 'no layer matched the DitHub adaptation rule'
        return replaced

    def _freeze_base(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = 'lora_' in name

    def init_weights(self) -> None:
        # DINO.init_weights は encoder 配下の dim>1 パラメータを一括 Xavier
        # 初期化するため、LoRA (A: kaiming, B: 0) を後から復元する
        super().init_weights()
        for m in self.modules():
            if isinstance(m, DitHubLinear):
                m.init_lora()
        self._freeze_base()

    # ---- フェーズ切替 ----------------------------------------------------

    def enable_per_class(self, trained_keys=None) -> None:
        """warmup -> specialization 切替 (DitHubPhaseHook から呼ばれる)."""
        for m in self.modules():
            if isinstance(m, DitHubLinear):
                m.enable_per_class(self.lambda_a, trained_keys)

    # ---- クラス選択の設定 -------------------------------------------------

    def loss(self, batch_inputs, batch_data_samples):
        keys = []
        for ds in batch_data_samples:
            names = list(ds.text)
            labels = ds.gt_instances.labels.unique().tolist()
            if not labels:
                labels = [0]  # 公式実装と同じフォールバック
            label = random.choice(labels)
            keys.append(canonical_key(names[label]))
        STATE.train_class_keys = keys
        # 注: DDP で学習する場合、config 側で encoder の activation checkpointing
        # を無効化 (num_cp=0) し find_unused_parameters=True とすること。
        # 再入 backward (fairscale checkpoint_wrapper) と可変な使用パラメータ
        # 集合 (クラス別 A) の組は DDP と両立しない (二重マークで落ちる)。
        return super().loss(batch_inputs, batch_data_samples)

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text = batch_data_samples[0].text
        names = list(text) if isinstance(text, (list, tuple)) else [text]
        STATE.eval_class_keys = [canonical_key(n) for n in names]
        return super().predict(
            batch_inputs, batch_data_samples, rescale=rescale)
