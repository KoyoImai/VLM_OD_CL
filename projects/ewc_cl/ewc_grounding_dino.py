"""EWC 付き MM-Grounding DINO（exp_042 design.md §2.2）。

- 学習対象（= EWC 対象）を config の ewc.target_components で選択し、対象外を凍結する。
  'all' で全モジュール（フル FT ＋ 全体ペナルティ）。
- 過去タスクのペナルティは 2 バッファ状態ファイル（projects/ewc_cl/ewc_state.py）から読む。
  t=1 は state_path=None で、ペナルティ無しのフル FT に一致する。
- dn の無効化（use_dn=False）は projects/lora_cl と同じ仕組み（NoDNQueryGenerator への
  差し替え。state_dict のキーは変わらない）。

config 例:
    model = dict(
        type='EWCGroundingDINO',
        use_dn=False,
        bbox_head=dict(type='NoDNGroundingDINOHead'),
        ewc=dict(
            target_components='all',   # または ['backbone', 'encoder', ...]
            lam=1000.0,                # λ。パイロット（design.md §2.4）で決める
            state_path=None))          # θ_{t-1} までの 2 バッファ。ドライバが上書き

注意: 状態バッファは checkpoint に含めない（work_dir の ckpt を肥大させず、
下流のロードで unexpected keys を出さないため）。状態はファイルで持ち回る。
"""
import torch
from mmengine.logging import print_log

from mmdet.registry import MODELS
from mmdet.models.detectors.grounding_dino import GroundingDINO

from projects.lora_cl.grounding_dino_lora import _build_nodn_query_generator
from .ewc_state import apply_target_selection, load_state, penalty


@MODELS.register_module()
class EWCGroundingDINO(GroundingDINO):

    def __init__(self, *args, ewc=None, use_dn: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        assert ewc is not None, 'ewc 設定が必要です'
        self.use_dn = use_dn
        if not use_dn:
            self.dn_query_generator = _build_nodn_query_generator(
                self.dn_query_generator)

        self.ewc_target_components = ewc.get('target_components', 'all')
        self.ewc_lam = float(ewc.get('lam', 0.0))
        self.ewc_state_path = ewc.get('state_path')

        targets = apply_target_selection(self, self.ewc_target_components)
        self._ewc_target_names = targets
        n_all = sum(p.numel() for p in self.parameters())
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print_log(
            f'[EWC] 対象={self.ewc_target_components} '
            f'({len(targets)} params, {n_tr:,}/{n_all:,} 要素) '
            f'λ={self.ewc_lam} state={self.ewc_state_path}',
            logger='current')

        # 状態は CPU で読み、初回の loss() でパラメータのデバイスへ移す
        self._ewc_state = None
        self._ewc_state_device = None
        if self.ewc_state_path:
            self._ewc_state = load_state(self.ewc_state_path)
            missing = [n for n in self._ewc_state['A'] if n not in targets]
            assert not missing, (
                'EWC 状態に学習対象外のパラメータが含まれています（対象の選択が '
                f'前タスクと食い違っている疑い）: {missing[:3]}')
            # 形状検証（別モデル・別 num_classes の状態の誤用を弾く。2026-08-15 監査で追加）
            shapes = {n: tuple(p.shape) for n, p in self.named_parameters()}
            bad = [n for n, v in self._ewc_state['A'].items()
                   if tuple(v.shape) != shapes[n]]
            assert not bad, f'EWC 状態とパラメータの形状が不一致: {bad[:3]}'
            print_log(
                f'[EWC] 状態を読込: {self.ewc_state_path} '
                f'(num_tasks={self._ewc_state["meta"]["num_tasks"]}, '
                f'{len(self._ewc_state["A"])} params)',
                logger='current')

    def _state_on_device(self, device):
        # デバイスが変わったらキャッシュを作り直す（cpu→cuda 移動後の陳腐化ガード）
        if (self._ewc_state_device is None
                or self._ewc_state_device.get('_device') != device):
            self._ewc_state_device = {
                'A': {k: v.to(device) for k, v in self._ewc_state['A'].items()},
                'B': {k: v.to(device) for k, v in self._ewc_state['B'].items()},
                'const': self._ewc_state['const'],
                '_device': device,
            }
        return self._ewc_state_device

    def ewc_penalty(self):
        """現在のパラメータに対するペナルティ (λ/2)·Σ_k F_k(θ−θ*_k)²。無効なら None。"""
        if self._ewc_state is None or self.ewc_lam <= 0:
            return None
        device = next(self.parameters()).device
        state = self._state_on_device(device)
        named = [(n, p) for n, p in self.named_parameters()
                 if n in state['A']]
        return penalty(state, named, self.ewc_lam)

    def loss(self, batch_inputs, batch_data_samples):
        losses = super().loss(batch_inputs, batch_data_samples)
        pen = self.ewc_penalty()
        if pen is not None:
            losses['loss_ewc'] = pen
        return losses
