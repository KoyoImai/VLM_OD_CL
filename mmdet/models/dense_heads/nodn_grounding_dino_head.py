# dn (contrastive denoising) を無効化したときに使う GroundingDINO head.
#
# 本ファイルは VLM 物体検出継続学習プロジェクト (exp_034) の新規追加物であり、
# 上流 mmdetection のコードには含まれない。上流ファイルは一切変更していない。
#
# 背景: 上流 mmdet/models/dense_heads/dino_head.py:462-463 は
#     num_denoising_queries = dn_meta['num_denoising_queries']
#     if dn_meta is not None:
# の順で書かれており、None チェックより前に dn_meta を参照している。
# そのため dn_meta=None (dn 無し) を渡すと TypeError で落ちる。
# ここでは split_outputs の None チェック順序だけを直す。それ以外の挙動は
# GroundingDINOHead と同一である。
#
# なお DINOHead.loss_by_feat は all_layers_denoising_cls_scores が None の場合に
# dn 損失の計算を飛ばすように既に書かれているため、修正はこの 1 メソッドで足りる。

from typing import Optional, Tuple

from torch import Tensor

from mmdet.registry import MODELS
from .grounding_dino_head import GroundingDINOHead


@MODELS.register_module()
class NoDNGroundingDINOHead(GroundingDINOHead):
    """``dn_meta=None`` を受け付ける GroundingDINOHead."""

    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_bbox_preds: Tensor,
                      dn_meta: Optional[dict]) -> Tuple[Tensor, ...]:
        """dn 部分と matching 部分に分割する.

        上流と違い、``dn_meta`` を参照する前に None を判定する。
        ``dn_meta`` が None のときは denoising 側を None、matching 側を
        入力そのものとして返す。
        """
        if dn_meta is not None:
            num_denoising_queries = dn_meta['num_denoising_queries']
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds)
