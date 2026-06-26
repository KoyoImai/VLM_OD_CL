"""GroundingDINOLoRA: GroundingDINO を継承し、指定した nn.Linear に LoRA を注入する。
- 対象は config の lora 指定（①構成要素 include ②種別 include_types ③正規表現 include_patterns の AND、exclude_patterns で除外）。
- nn.Linear のみ対象。backbone/language_model は既定除外。
- nn.MultiheadAttention 直下の out_proj は functional 経由で LoRA が効かないため自動スキップ。
- 既存コードは不変（新規登録名のみ、インスタンス改変のみ）。
"""
import re
import torch.nn as nn
from mmengine.logging import print_log
from mmdet.registry import MODELS
from mmdet.models.detectors.grounding_dino import GroundingDINO

from .lora_layers import LoRALinear


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
    def __init__(self, *args, lora=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert lora is not None, 'lora 設定が必要です'
        self._inject_lora(lora)

    @staticmethod
    def _match(name, module, cfg):
        if not isinstance(module, nn.Linear):
            return False
        if name.startswith(('backbone.', 'language_model.')):
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
        wrapped, skipped_mha = [], []
        for name, module in list(self.named_modules()):
            if not self._match(name, module, cfg):
                continue
            if isinstance(_get_parent(self, name), nn.MultiheadAttention):
                skipped_mha.append(name)   # out_proj は functional 経由=LoRA無効
                continue
            wrapped.append(name)
        for name in wrapped:
            mod = dict(self.named_modules())[name]
            _set_submodule(self, name, LoRALinear(mod, r, alpha))
        # LoRA のみ学習可、それ以外は凍結
        for pn, p in self.named_parameters():
            p.requires_grad_(pn.endswith('lora_A') or pn.endswith('lora_B'))
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_tot = sum(p.numel() for p in self.parameters())
        print_log(f'[LoRA] r={r} alpha={alpha} 付与層={len(wrapped)} '
                  f'(MHA out_proj スキップ={len(skipped_mha)}) '
                  f'trainable={n_tr:,}/{n_tot:,} ({100*n_tr/max(n_tot,1):.2f}%)', logger='current')
        print_log('[LoRA] 対象層: ' + ', '.join(wrapped), logger='current')
        if skipped_mha:
            print_log('[LoRA] スキップ(MHA out_proj): ' + ', '.join(skipped_mha), logger='current')
