# dn (contrastive denoising) を無効化するための query generator.
#
# 本ファイルは VLM 物体検出継続学習プロジェクト (exp_034) の新規追加物であり、
# 上流 mmdetection のコードには含まれない。上流ファイルは一切変更していない。
#
# 背景: 上流 mmdet では dn を config から無効化できない。
#   - mmdet/models/detectors/dino.py:44 は dn_cfg=None でも
#     CdnQueryGenerator(**dn_cfg) を呼ぶため TypeError で落ちる。
#   - group_cfg の num_dn_queries=0 は dino_layers.py の get_num_groups が
#     num_groups < 1 を 1 に引き上げるため無効化にならない。
# そこで CdnQueryGenerator を継承し、__call__ だけを「長さ 0 の dn クエリと
# dn_meta=None を返す」ものに差し替える。
#
# label_embedding をはじめとする属性・パラメータは親クラスのまま保持するので、
# state_dict のキーは dn 有効時と完全に同一である (逐次学習で ckpt を読み継ぐ際に
# 形が変わらない)。

from typing import Optional, Tuple

from torch import Tensor

from mmdet.structures import SampleList
from .transformer.dino_layers import CdnQueryGenerator


class NoDNQueryGenerator(CdnQueryGenerator):
    """dn を行わない CdnQueryGenerator.

    長さ 0 の dn クエリと ``dn_meta=None`` を返す。上流の
    ``GroundingDINO.pre_decoder`` (grounding_dino.py:388-393) は
    ``torch.cat`` で連結するだけなので、長さ 0 ならクエリ列は
    matching part のみとなり、dn が存在しない状態と等価になる。
    """

    def __call__(self, batch_data_samples: SampleList
                 ) -> Tuple[Tensor, Tensor, Optional[Tensor], None]:
        weight = self.label_embedding.weight
        bs = len(batch_data_samples)
        dn_label_query = weight.new_zeros((bs, 0, self.embed_dims))
        dn_bbox_query = weight.new_zeros((bs, 0, 4))
        # attn_mask=None は decoder の self_attn_mask 無しを意味する。
        # dn が無いときは matching part 同士のマスクも不要。
        return dn_label_query, dn_bbox_query, None, None
