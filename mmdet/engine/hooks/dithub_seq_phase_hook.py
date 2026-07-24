"""exp_023: DitHub 逐次学習の式3対応フェーズフック（既存コードは無変更）.

背景: DitHub を逐次学習し過去ドメインを評価するには、成長するライブラリ
（全ドメインの per_class_lora_A）を保持する必要がある（→ ドメイン境界で
merge_dithub.py が式4のB融合＋A和集合を行う）。加えて公式忠実（案Y）では、
現ドメインのクラスのうち過去に学習済みのもの（trained_keys）に対して、
specialization 切替時に式3の fetch+merge:
    A_c ← λ_A·A_warmup + (1-λ_A)·A_c_old   (λ_A=0.3)
を適用する（未学習クラスは warmup をコピー=branch）。

既存 DitHubPhaseHook は enable_per_class() を trained_keys 無し（=None→全クラス
branch）で呼ぶため、過去に学習したクラスも warmup で上書きしてしまう。本フックは
DitHubPhaseHook を継承し、config で与えた trained_classes（現ドメイン∩過去ドメインの
クラス名）を canonical_key 化して enable_per_class(trained_keys=...) に渡すことで
式3を発火させる。式3の A_c_old は、逐次ドライバが前ドメインのライブラリ ckpt を
load_from に渡すことでモデルへロード済みである前提（現ドメインのスロットに
過去 A がロードされる重複クラスにのみ式3が効く）。

既存の dithub_phase_hook.py / dithub_layers.py / dithub_grounding_dino.py は
一切変更しない。enable_per_class の式3ロジック自体は既存実装（dithub_layers.py:92-98）
をそのまま用い、本フックは trained_keys を供給するだけ。
"""
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS
from mmdet.models.layers.dithub_layers import canonical_key
from mmdet.engine.hooks.dithub_phase_hook import DitHubPhaseHook


@HOOKS.register_module()
class DitHubSeqPhaseHook(DitHubPhaseHook):
    """warmup->specialization 切替時に trained_keys を渡して式3を発火する.

    Args:
        warmup_epochs / warmup_iters: 親と同じ（どちらか一方）。
        trained_classes (list[str]): 現ドメインのクラスのうち過去ドメインで
            既に学習済みのクラス名一覧（式3の対象）。t=1,2 は空、t=3 は
            ['person','car'] など。既定は空（=全クラス branch＝単発時と同じ）。
    """

    def __init__(self, warmup_epochs=None, warmup_iters=None,
                 trained_classes=None):
        super().__init__(
            warmup_epochs=warmup_epochs, warmup_iters=warmup_iters)
        self.trained_classes = list(trained_classes or [])

    def _enable(self, runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        trained_keys = {canonical_key(c) for c in self.trained_classes}
        model.enable_per_class(trained_keys=trained_keys)
        runner.logger.info(
            '>>> DitHub SEQ: START SPECIALIZATION '
            f'(式3 fetch+merge on {len(trained_keys)} re-trained classes) <<<')
