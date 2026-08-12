"""GroundingDINOLoRA: GroundingDINO を継承し、指定した nn.Linear に LoRA を注入する。
- 対象は config の lora 指定（①構成要素 include ②種別 include_types ③正規表現 include_patterns の AND、exclude_patterns で除外）。
- nn.Linear のみ対象。backbone(Swin) と language_model(BERT) は既定で除外するが、
  `lora.exclude_components` を明示指定すれば対象にできる。
    exclude_components=[]                  -> Swin も BERT も対象
    exclude_components=['language_model']  -> Swin のみ対象
  既定値は ('backbone', 'language_model') なので、指定しなければ従来どおり。
- nn.MultiheadAttention は in_proj_weight が生 Parameter・out_proj が functional 経由の
  ため、そのままでは LoRA が入らない。`lora.decompose_mha`（正規表現のリスト）に
  該当する MHA を q/k/v/o の独立 nn.Linear へ分解してから注入する。分解は既存の
  DecomposedMHA（mmdet/models/layers/dithub_layers.py、本プロジェクトが追加した
  ファイル）を import して使うだけで、mmdet のコードは一切変更しない。
- 既存コードは不変（新規登録名のみ、インスタンス改変のみ）。
"""
import re
import torch.nn as nn
from mmengine.logging import print_log
from mmdet.registry import MODELS
from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.models.layers.dithub_layers import DecomposedMHA
from mmdet.models.layers.nodn_query_generator import NoDNQueryGenerator

from .lora_layers import LoRALinear


def _build_nodn_query_generator(gen) -> NoDNQueryGenerator:
    """既存の CdnQueryGenerator と同じ設定で NoDNQueryGenerator を作る。

    state_dict のキー (dn_query_generator.label_embedding.weight) は同一のまま
    保たれるため、θ0 のロードにも ckpt の構造にも影響しない。
    ODinW-13 では公式 ZiRa / DitHub がともに dn_number=0 で学習しており
    (exp_034 / exp_037 と同条件にするため)、config から dn を切れるようにする。
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


# 既定で LoRA の対象から外す構成要素（config の lora.exclude_components で上書き可）。
# 事前学習済み表現をなるべく触らない、という従来の方針をそのまま既定値にしている。
DEFAULT_EXCLUDE_COMPONENTS = ('backbone', 'language_model')


def _get_parent(model, name):
    parent = model
    for p in name.split('.')[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent


def _set_submodule(model, name, new):
    parent = _get_parent(model, name)
    last = name.split('.')[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


@MODELS.register_module()
class GroundingDINOLoRA(GroundingDINO):
    def __init__(self, *args, lora=None, use_dn: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        assert lora is not None, 'lora 設定が必要です'
        self.use_dn = use_dn
        if not use_dn:
            self.dn_query_generator = _build_nodn_query_generator(
                self.dn_query_generator)
        self.decomposed_mha = self._decompose_mha(
            lora.get('decompose_mha') or [])
        self._inject_lora(lora)

    def _decompose_mha(self, patterns):
        """該当する nn.MultiheadAttention を q/k/v/o 分解型へ差し替える。

        Args:
            patterns (list[str]): モジュール名に対する正規表現のリスト。
                例: [r'^encoder\\.text_layers\\.\\d+\\.self_attn\\.attn$']
                空リストなら何もしない（従来どおりの挙動）。

        分解後の重みは元と数学的に等価で、DecomposedMHA._load_from_state_dict が
        事前学習 ckpt の in_proj_weight / in_proj_bias / out_proj.* を分割して
        読むため、θ0 を無変換でロードできる。分解した層は名前が
        `...attn.linear_{q,k,v,o}` になり、通常の nn.Linear として LoRA の
        選択対象（include / include_patterns）に入る。
        """
        if not patterns:
            return []
        replaced = []
        for name, module in list(self.named_modules()):
            if not isinstance(module, nn.MultiheadAttention):
                continue
            if not any(re.search(p, name) for p in patterns):
                continue
            new = DecomposedMHA(module.embed_dim, module.num_heads,
                                module.dropout)
            new.load_from_multihead(module)
            _set_submodule(self, name, new)
            replaced.append(name)
        print_log(f'[LoRA] MHA 分解={len(replaced)} 個: ' + ', '.join(replaced),
                  logger='current')
        return replaced

    @staticmethod
    def _match(name, module, cfg):
        if not isinstance(module, nn.Linear):
            return False
        if name.split('.')[0] in cfg.get('exclude_components',
                                         DEFAULT_EXCLUDE_COMPONENTS):
            return False
        inc = cfg.get('include')
        types = cfg.get('include_types')
        pats = cfg.get('include_patterns')
        excl = cfg.get('exclude_patterns', []) or []
        comp_ok = (inc is None) or (name.split('.')[0] in inc)
        type_ok = (types is None) or any(t in name for t in types)
        pat_ok = (pats is None) or any(re.search(p, name) for p in pats)
        not_excluded = not any(re.search(p, name) for p in excl)
        return comp_ok and type_ok and pat_ok and not_excluded

    def _inject_lora(self, cfg):
        r = cfg['r']
        alpha = cfg.get('alpha', r)
        excl = tuple(cfg.get('exclude_components', DEFAULT_EXCLUDE_COMPONENTS))
        # include で名指ししたのに既定除外に残っている、という取り違えを
        # 学習開始前に落とす（黙って 0 層になるのを防ぐ）。
        conflict = sorted(set(cfg.get('include') or []) & set(excl))
        assert not conflict, (
            f'include={conflict} は exclude_components={list(excl)} で除外されている。'
            f'対象にするなら lora.exclude_components から外すこと '
            f'(例: exclude_components=[])')
        wrapped, skipped_mha = [], []
        for name, module in list(self.named_modules()):
            if not self._match(name, module, cfg):
                continue
            if isinstance(_get_parent(self, name), nn.MultiheadAttention):
                skipped_mha.append(name)   # out_proj は functional 経由=LoRA無効
                continue
            wrapped.append(name)
        modules = dict(self.named_modules())
        for name in wrapped:
            _set_submodule(self, name, LoRALinear(modules[name], r, alpha))
        self._freeze_base()
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_tot = sum(p.numel() for p in self.parameters())
        from collections import Counter
        by_comp = dict(Counter(n.split('.')[0] for n in wrapped))
        print_log(f'[LoRA] r={r} alpha={alpha} 付与層={len(wrapped)} '
                  f'(MHA out_proj スキップ={len(skipped_mha)}) '
                  f'除外={list(excl)} 内訳={by_comp} '
                  f'trainable={n_tr:,}/{n_tot:,} ({100*n_tr/max(n_tot,1):.2f}%)', logger='current')
        print_log('[LoRA] 対象層: ' + ', '.join(wrapped), logger='current')
        if skipped_mha:
            print_log('[LoRA] スキップ(MHA out_proj): ' + ', '.join(skipped_mha), logger='current')

    def _freeze_base(self):
        """LoRA のみ学習可、それ以外は凍結。"""
        for pn, p in self.named_parameters():
            p.requires_grad_(pn.endswith('lora_A') or pn.endswith('lora_B'))

    def init_weights(self) -> None:
        """LoRA の初期状態と凍結を init_weights の後に復元する。

        DINO.init_weights は encoder / decoder 配下の dim>1 パラメータを一括で
        xavier 初期化する (mmdet/models/detectors/dino.py:72-75)。lora_A/lora_B は
        どちらも 2 次元なのでこの対象に入り、B=0 初期化が壊れる。lora_* は
        事前学習 ckpt に無いため load_from でも戻らず、学習開始時点で ΔW != 0
        （実測 ||ΔW||_F/||W||_F 中央値 0.108、最大 0.455）になっていた。
        """
        super().init_weights()
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.init_lora()
        self._freeze_base()
