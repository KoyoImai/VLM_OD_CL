# ZiRa (Deng et al., NeurIPS 2024) RDB layers.
#
# 本ファイルは VLM 物体検出継続学習プロジェクト (exp_020) の新規追加物であり、
# 上流 mmdetection のコードには含まれない。既存ファイルは一切変更していない。
# 登録は exp_020 の config が custom_imports で本モジュールを import することで行う。
# 台帳・設計: experiments/exp_020/implementation_plan.md
# 手法の正本: papers/ZiRa_implementation_notes.md
#
# 公式実装 (https://github.com/JarintotionDin/ZiRaGroundingDINO) の
# RepZeroLinear / RepZeroConv2d に対応する。相違点:
#   - 凍結された事前学習層 (base) を同一モジュール内に持つ
#     (checkpoint のキー名を保つため。base の forward は数式上同一)
#   - 評価時も全和 (base + s*HLRB + LLRB) を返す。公式は評価時 LLRB のみだが、
#     公式はタスク末尾に HLRB を LLRB へ融合してから評価するため数学的に等価
#   - 言語側 HLRB の bias も 1e-8 で初期化する (公式コードは linear の HLRB bias
#     のみ nn.Linear 既定の乱数初期化のまま残っているが、論文の
#     "initialize the parameters of the total RDB as zero" に合わせた)

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# 公式実装の定数 (zero_value / lan_scale / vis_scale)
HLRB_INIT = 1e-8
SCALING_INIT = 0.1


def _smooth_l1_to_zero(x: Tensor) -> Tensor:
    """ZiL の 1 項: SmoothL1(x, 0), reduction='mean' (公式実装の既定)."""
    return F.smooth_l1_loss(x, torch.zeros_like(x), reduction='mean')


class ZiRaLinear(nn.Linear):
    """text_feat_map を置き換える RDB 付き Linear.

    self.weight / self.bias は凍結される事前学習層そのもの (キー名維持)。
    hlrb / llrb / scaling が学習対象で、forward は
    base(x) + scaling * hlrb(x) + llrb(x)。
    学習時は self.zil に ZiL (2 項) を保持し、検出器が回収する。
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.hlrb = nn.Linear(in_features, out_features, bias=bias)
        self.llrb = nn.Linear(in_features, out_features, bias=bias)
        self.scaling = nn.Parameter(torch.ones(1) * SCALING_INIT)
        self.zil = None
        self.init_rdb()

    @torch.no_grad()
    def init_rdb(self) -> None:
        """RDB を公式の初期状態 (HLRB=1e-8, LLRB=0, s=0.1) にする."""
        nn.init.constant_(self.hlrb.weight, HLRB_INIT)
        if self.hlrb.bias is not None:
            nn.init.constant_(self.hlrb.bias, HLRB_INIT)
        nn.init.constant_(self.llrb.weight, 0.0)
        if self.llrb.bias is not None:
            nn.init.constant_(self.llrb.bias, 0.0)
        self.scaling.fill_(SCALING_INIT)

    @torch.no_grad()
    def rep_merge(self) -> None:
        """公式 __rep__ と同一: LLRB へ融合し HLRB と s を再初期化する."""
        self.llrb.weight += self.scaling * self.hlrb.weight
        if self.llrb.bias is not None:
            self.llrb.bias += self.scaling.squeeze() * self.hlrb.bias
        self.init_rdb()

    def forward(self, x: Tensor) -> Tensor:
        base = F.linear(x, self.weight, self.bias)
        branch = self.scaling * self.hlrb(x)
        rdb = branch + self.llrb(x)
        if self.training:
            self.zil = _smooth_l1_to_zero(branch) + _smooth_l1_to_zero(rdb)
        return base + rdb


class ZiRaConvRDB(nn.Module):
    """neck の射影 conv に並列する RDB (基準 conv と同形の 2 枝).

    base conv 自体は ChannelMapper 側に残る (本モジュールは RDB 出力のみ返す)。
    template_conv から in/out ch, kernel, stride, padding, bias を写して構築する。
    """

    def __init__(self, template_conv: nn.Conv2d) -> None:
        super().__init__()
        kwargs = dict(
            kernel_size=template_conv.kernel_size,
            stride=template_conv.stride,
            padding=template_conv.padding,
            dilation=template_conv.dilation,
            groups=template_conv.groups,
            bias=template_conv.bias is not None)
        self.hlrb = nn.Conv2d(template_conv.in_channels,
                              template_conv.out_channels, **kwargs)
        self.llrb = nn.Conv2d(template_conv.in_channels,
                              template_conv.out_channels, **kwargs)
        self.scaling = nn.Parameter(torch.ones(1) * SCALING_INIT)
        self.zil = None
        self.init_rdb()

    @torch.no_grad()
    def init_rdb(self) -> None:
        nn.init.constant_(self.hlrb.weight, HLRB_INIT)
        if self.hlrb.bias is not None:
            nn.init.constant_(self.hlrb.bias, HLRB_INIT)
        nn.init.constant_(self.llrb.weight, 0.0)
        if self.llrb.bias is not None:
            nn.init.constant_(self.llrb.bias, 0.0)
        self.scaling.fill_(SCALING_INIT)

    @torch.no_grad()
    def rep_merge(self) -> None:
        self.llrb.weight += self.scaling * self.hlrb.weight
        if self.llrb.bias is not None:
            self.llrb.bias += self.scaling.squeeze() * self.hlrb.bias
        self.init_rdb()

    def forward(self, x: Tensor) -> Tensor:
        branch = self.scaling * self.hlrb(x)
        rdb = branch + self.llrb(x)
        if self.training:
            self.zil = _smooth_l1_to_zero(branch) + _smooth_l1_to_zero(rdb)
        return rdb
