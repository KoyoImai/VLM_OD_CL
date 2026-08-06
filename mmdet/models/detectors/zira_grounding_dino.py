# ZiRa 用検出器: MM-Grounding DINO + RDB (text_feat_map 差し替え) + ZiL 回収 + 凍結.
#
# 本ファイルは VLM 物体検出継続学習プロジェクト (exp_020) の新規追加物であり、
# 上流 mmdetection のコードには含まれない。既存ファイルは一切変更していない。
# 登録は exp_020 の config が custom_imports で本モジュールを import することで行う。
# 台帳・設計: experiments/exp_020/implementation_plan.md
#
# 変更は 3 点のみ:
#   1. text_feat_map を ZiRaLinear に差し替える (checkpoint キー名は維持され、
#      親クラスの 3 箇所の呼び出しはそのまま動く)
#   2. loss() で全 RDB 層から ZiL を回収し loss_zil として加える
#   3. RDB (hlrb / llrb / scaling) 以外の全パラメータを requires_grad=False にする
#      (公式の freeze_all + unfreeze adapter と同方式)
#   4. use_dn=False のとき dn クエリ生成器を NoDNQueryGenerator に差し替える
#      (exp_034 で追加。既定 True なので exp_020 / exp_023 の挙動は変わらない)
# neck 側の RDB は config で neck.type='ZiRaChannelMapper' を指定して挿入する。

import torch

from mmdet.registry import MODELS
from mmdet.models.layers.nodn_query_generator import NoDNQueryGenerator
from mmdet.models.layers.zira_layers import ZiRaConvRDB, ZiRaLinear
from .grounding_dino import GroundingDINO

# 学習対象 (RDB) と判定するパラメータ名の部分文字列
RDB_NAME_TOKENS = ('hlrb', 'llrb', 'scaling')


def _build_nodn_query_generator(gen) -> NoDNQueryGenerator:
    """既存の CdnQueryGenerator と同じ設定で NoDNQueryGenerator を作る.

    state_dict のキー (dn_query_generator.label_embedding.weight) は同一のまま
    保たれる。中身は dn を使わないため参照されない。
    """
    if gen.dynamic_dn_groups:
        group_cfg = dict(dynamic=True, num_dn_queries=gen.num_dn_queries)
    else:
        group_cfg = dict(dynamic=False, num_groups=gen.num_groups)
    return NoDNQueryGenerator(
        num_classes=gen.num_classes,
        embed_dims=gen.embed_dims,
        num_matching_queries=gen.num_matching_queries,
        label_noise_scale=gen.label_noise_scale,
        box_noise_scale=gen.box_noise_scale,
        group_cfg=group_cfg)


@MODELS.register_module()
class ZiRaGroundingDINO(GroundingDINO):
    """MM-Grounding DINO に ZiRa (RDB + ZiL) を組み込んだ検出器.

    Args:
        zil_loss_weight (float): ZiL の損失重み λ。公式実装の既定 0.1。
        use_dn (bool): dn (contrastive denoising) を使うかどうか。既定 True
            (exp_020 / exp_023 の挙動と同一)。False にすると dn クエリ生成器を
            NoDNQueryGenerator に差し替え、dn 損失 18 項が消える。公式 ZiRa は
            dn_number=0 であり、その再現 (exp_034) では False を使う。
    """

    def __init__(self, *args, zil_loss_weight: float = 0.1,
                 use_dn: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.zil_loss_weight = zil_loss_weight
        self.use_dn = use_dn

        if not use_dn:
            self.dn_query_generator = _build_nodn_query_generator(
                self.dn_query_generator)

        # text_feat_map (Linear 768->256) を RDB 付きに差し替える。
        # 事前学習重みのキー名 (text_feat_map.weight / .bias) は変わらない。
        old = self.text_feat_map
        new = ZiRaLinear(
            old.in_features, old.out_features, bias=old.bias is not None)
        with torch.no_grad():
            new.weight.copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.text_feat_map = new

        self._freeze_base()

    def _freeze_base(self) -> None:
        """RDB 以外の全パラメータを凍結する."""
        for name, param in self.named_parameters():
            param.requires_grad = any(
                token in name for token in RDB_NAME_TOKENS)

    def init_weights(self) -> None:
        # 親の init_weights は text_feat_map.weight を Xavier で初期化する
        # (その後 checkpoint の load で上書きされる)。RDB は公式の初期状態へ
        # 戻す。neck 側 RDB は ZiRaChannelMapper.init_weights が自身で戻す。
        super().init_weights()
        self.text_feat_map.init_rdb()

    def _collect_zil(self) -> torch.Tensor:
        """全 RDB 層が forward で保持した ZiL を合算して回収する."""
        total = None
        for module in self.modules():
            if isinstance(module, (ZiRaLinear, ZiRaConvRDB)):
                assert module.zil is not None, (
                    f'ZiL not computed for {type(module).__name__}; '
                    'loss() must be called right after a training forward')
                total = module.zil if total is None else total + module.zil
                module.zil = None
        assert total is not None
        return total

    def loss(self, batch_inputs, batch_data_samples):
        losses = super().loss(batch_inputs, batch_data_samples)
        losses['loss_zil'] = self.zil_loss_weight * self._collect_zil()
        return losses
