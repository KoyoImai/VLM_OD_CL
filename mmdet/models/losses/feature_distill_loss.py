"""特徴蒸留の損失（exp_028/029/030/031 用・新規ファイル。既存コードは無変更）.

design: experiments/exp_028/design.md §4.3

すべての対象を「有効な位置ごとに距離 → 有効位置数で平均 → 特徴次元で正規化」という
同一の形で扱う。画像の多スケール特徴も [B, N, C] に平坦化することで、テキスト・融合後と
まったく同じ関数で処理できる。

    L2      : (1/C) * ||s_i - t_i||^2   を有効位置 i で平均
    cosine  : 1 - cos(s_i, t_i)         を有効位置 i で平均

`__init__.py` には登録しない（既存無変更の方針）。config の custom_imports で
`mmdet.models.losses.feature_distill_loss` をフルパス指定して読み込む。
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class FeatureDistillLoss(nn.Module):
    """位置ごとの特徴距離を有効位置で平均する蒸留損失.

    Args:
        form (str): 'l2' または 'cosine'。
        eps (float): コサイン計算のゼロ割り防止。
    """

    def __init__(self, form: str = 'l2', eps: float = 1e-8) -> None:
        super().__init__()
        assert form in ('l2', 'cosine'), f'unknown form: {form}'
        self.form = form
        self.eps = eps

    # ------------------------------------------------------------------ 距離
    def pointwise(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """位置ごとの距離。s, t: [..., C] -> [...]"""
        if self.form == 'l2':
            # (1/C) * ||s - t||^2 。mean が 1/C の正規化を兼ねる。
            return ((s - t)**2).mean(dim=-1)
        # cosine: 向きのみを比較する。次元による正規化は不要。
        return 1.0 - F.cosine_similarity(s, t, dim=-1, eps=self.eps)

    # -------------------------------------------------------------- 系列/平坦
    def seq_loss(self,
                 s: torch.Tensor,
                 t: torch.Tensor,
                 valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, N, C] の系列を有効位置で平均する。

        Args:
            s, t: [B, N, C]
            valid: [B, N] の bool/0-1。True(1) が有効位置。None なら全位置有効。

        Returns:
            [B] のサンプルごとの損失。
        """
        d = self.pointwise(s, t)  # [B, N]
        if valid is None:
            return d.mean(dim=1)
        m = valid.to(d.dtype)
        return (d * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def map_loss(self, s_feats: List[torch.Tensor],
                 t_feats: List[torch.Tensor],
                 valid_masks: Optional[List[torch.Tensor]] = None
                 ) -> torch.Tensor:
        """多スケール特徴マップ。各スケールを平坦化して seq_loss に流し、スケール間で平均。

        Args:
            s_feats, t_feats: 長さ L のリスト。各要素 [B, C, H, W]。
            valid_masks: 長さ L のリスト。各要素 [B, H, W]（True が実画像領域）。

        Returns:
            [B] のサンプルごとの損失。
        """
        assert len(s_feats) == len(t_feats)
        per_level = []
        for lvl, (s, t) in enumerate(zip(s_feats, t_feats)):
            b, c = s.shape[0], s.shape[1]
            s_flat = s.flatten(2).transpose(1, 2)  # [B, H*W, C]
            t_flat = t.flatten(2).transpose(1, 2)
            v = None
            if valid_masks is not None:
                v = valid_masks[lvl].flatten(1)  # [B, H*W]
            per_level.append(self.seq_loss(s_flat, t_flat, v))
        return torch.stack(per_level, dim=0).mean(dim=0)  # スケール間平均
