# Copyright (c) OpenMMLab. All rights reserved.
"""蒸留E+（projector 蒸留・全データ対象）の GroundingDINO（exp_044 design.md §2.2）.

`KDGroundingDINO`（exp_028 実装＝note07 の方法）との差分は 2 点だけ:

1. **projector 蒸留**: 学生の特徴を projector に通し、projector の出力を教師の特徴に
   近づける。projector は 4 つの蒸留点（neck 後画像 / text_feat_map 後テキスト /
   融合後画像 memory / 融合後テキスト memory_text）に**独立**な同型の点ごと 2 層 MLP
   `Linear(256→256) → GELU → Linear(256→256)`（画像特徴には各空間位置に適用 =
   1×1 conv として実装）。標準初期化・残差なし（恒等初期化は非線形 MLP では原理的に
   不可能であることを確認済み。design.md §2.2）。
   projector は学生と同じ optimizer で学習され、state_dict に含まれるため
   **タスク間は load_from で引き継がれる**。評価（plain config）では読まれず無害。

2. **蒸留対象の全データ化**: バッファ由来サンプル限定をやめ、ミニバッチの全サンプルに
   蒸留を掛ける。実装は親クラスの「バッチ末尾 n 件」機構をそのまま使い、n をバッチ
   サイズ以上に固定することで末尾スライスが全体になる（この場合、親の
   `_assert_buffer_slice` は検査対象なしで早期 return する）。

損失の集計（L2、3 項平均、fus は画像側・テキスト側の 0.5 和）・教師の構築・特徴捕捉は
親クラスをそのまま使う。教師 deepcopy（init_weights 内）には projector も複製されるが、
教師側では使われず凍結されるだけで無害。

`__init__.py` には登録しない。config の custom_imports でフルパス指定して読み込む。
"""
from typing import Dict, List, Optional

import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from .kd_grounding_dino import KDGroundingDINO

_FEAT_DIM = 256


def _mlp_conv() -> nn.Module:
    """画像特徴（B, C, H, W）用: 各空間位置に同じ 2 層 MLP（= 1×1 conv）."""
    return nn.Sequential(
        nn.Conv2d(_FEAT_DIM, _FEAT_DIM, 1),
        nn.GELU(),
        nn.Conv2d(_FEAT_DIM, _FEAT_DIM, 1),
    )


def _mlp_linear() -> nn.Module:
    """系列特徴（B, L, C）用: 各トークンに同じ 2 層 MLP."""
    return nn.Sequential(
        nn.Linear(_FEAT_DIM, _FEAT_DIM),
        nn.GELU(),
        nn.Linear(_FEAT_DIM, _FEAT_DIM),
    )


@MODELS.register_module()
class KDProjGroundingDINO(KDGroundingDINO):

    def __init__(self, *args, kd: Optional[dict] = None, **kwargs) -> None:
        super().__init__(*args, kd=kd, **kwargs)
        # 蒸留対象の全データ化: バッチ末尾スライスを常に全体にする（docstring 参照）
        self.num_buffer_per_batch = 10**9
        # 4 点独立の projector（学生側のみで使用）
        self.kd_proj_img = _mlp_conv()          # neck 後のマルチスケール画像特徴
        self.kd_proj_txt = _mlp_linear()        # text_feat_map 後のテキスト特徴
        self.kd_proj_fus_img = _mlp_linear()    # 融合後画像側 memory（位置の系列）
        self.kd_proj_fus_txt = _mlp_linear()    # 融合後テキスト側 memory_text

    _PROJ_KEYS = ('kd_proj_img', 'kd_proj_txt',
                  'kd_proj_fus_img', 'kd_proj_fus_txt')

    def init_weights(self) -> None:
        """教師の deepcopy に projector を含めずに親の教師構築を行う.

        教師は projector を使わない。また親クラス（KDGroundingDINO.init_weights）は
        「教師の全キーが teacher_ckpt に存在する」ことを検査するため、projector を
        含めたまま deepcopy すると、θ0 に無い kd_proj_* が欠落と誤検出されて
        assert で止まる（2026-08-16 の実行前検証で検出）。一時的に外して親を呼び、
        復元する。学生側の projector は本メソッドの外（__init__ の標準初期化）のまま。
        """
        if self._teacher is not None:
            return
        projs = {k: self._modules.pop(k) for k in self._PROJ_KEYS}
        try:
            super().init_weights()
        finally:
            self._modules.update(projs)

    def _project_student(self, s: Dict) -> Dict:
        """学生の捕捉特徴に projector を適用した辞書を返す（教師側は触らない）."""
        out = dict(s)
        out['img'] = [self.kd_proj_img(x) for x in s['img']]
        out['txt'] = self.kd_proj_txt(s['txt'])
        out['mem'] = self.kd_proj_fus_img(s['mem'])
        out['mem_txt'] = self.kd_proj_fus_txt(s['mem_txt'])
        return out

    def _kd_losses(self, s: Dict, t: Dict, img_masks: List[Tensor],
                   buf: slice, diag: Optional[dict] = None) -> Tensor:
        return super()._kd_losses(
            self._project_student(s), t, img_masks, buf, diag=diag)
