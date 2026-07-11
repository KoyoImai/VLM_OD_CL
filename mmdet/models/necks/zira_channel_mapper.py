# ZiRa 用 neck: ChannelMapper + 射影 conv に並列する RDB.
#
# 本ファイルは VLM 物体検出継続学習プロジェクト (exp_020) の新規追加物であり、
# 上流 mmdetection のコードには含まれない。既存ファイルは一切変更していない。
# 登録は exp_020 の config が custom_imports で本モジュールを import することで行う。
# 台帳・設計: experiments/exp_020/implementation_plan.md
#
# 公式実装では input_proj (conv + GroupNorm) の conv 出力へ GroupNorm の前に
# RDB 出力を加算する: GN(conv(x) + rdb(x))。ChannelMapper の convs / extra_convs
# の構造とキー名は親クラスのまま保ち (checkpoint 互換)、forward だけを置き換える。

from typing import Tuple

from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.models.layers.zira_layers import ZiRaConvRDB
from .channel_mapper import ChannelMapper


@MODELS.register_module()
class ZiRaChannelMapper(ChannelMapper):
    """ChannelMapper に ZiRa の RDB conv を並列追加した neck."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 本実装は conv -> norm -> act の順序を前提に GN 前へ加算する
        for cm in list(self.convs) + list(self.extra_convs or []):
            assert cm.order == ('conv', 'norm', 'act'), (
                'ZiRaChannelMapper assumes ConvModule order (conv, norm, act), '
                f'got {cm.order}')
        self.rdb_convs = nn.ModuleList(
            [ZiRaConvRDB(cm.conv) for cm in self.convs])
        if self.extra_convs:
            self.rdb_extra_convs = nn.ModuleList(
                [ZiRaConvRDB(cm.conv) for cm in self.extra_convs])
        else:
            self.rdb_extra_convs = None

    def init_weights(self) -> None:
        # 親の init_cfg (Xavier, layer='Conv2d') が RDB conv も初期化してしまう
        # ため、後から RDB を公式の初期状態 (HLRB=1e-8, LLRB=0, s=0.1) に戻す
        super().init_weights()
        for m in self.modules():
            if isinstance(m, ZiRaConvRDB):
                m.init_rdb()

    @staticmethod
    def _forward_one(cm, rdb: ZiRaConvRDB, x: Tensor) -> Tensor:
        out = cm.conv(x) + rdb(x)
        if cm.with_norm:
            out = cm.norm(out)
        if cm.with_activation:
            out = cm.activate(out)
        return out

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        assert len(inputs) == len(self.convs)
        outs = [
            self._forward_one(self.convs[i], self.rdb_convs[i], inputs[i])
            for i in range(len(inputs))
        ]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                inp = inputs[-1] if i == 0 else outs[-1]
                outs.append(
                    self._forward_one(self.extra_convs[i],
                                      self.rdb_extra_convs[i], inp))
        return tuple(outs)
