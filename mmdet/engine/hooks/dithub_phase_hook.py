# DitHub の warmup -> specialization 切替フック.
#
# 本ファイルは VLM 物体検出継続学習プロジェクト (exp_021) の新規追加物であり、
# 上流 mmdetection のコードには含まれない。既存ファイルは一切変更していない。
# 登録は exp_021 の config が custom_imports で本モジュールを import することで行う。
# 台帳・設計: experiments/exp_021/implementation_plan.md
#
# 公式実装ではタスクの学習量の厳密な半分 (iter == max_iter/2) で
# TaskMemory().enable_per_class() が呼ばれる。本フックはそれを mmengine の
# フックとして再現する (epoch ベースと iter ベースの両対応)。

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class DitHubPhaseHook(Hook):
    """warmup フェーズ終了時に detector.enable_per_class() を発火する.

    Args:
        warmup_epochs (int, optional): epoch ベース学習でのフェーズ切替点
            (この epoch 数を終えた時点で切替。20 epoch の等分なら 10)。
        warmup_iters (int, optional): iter ベース学習での切替点
            (公式準拠なら max_iters/2 = 1500)。
        どちらか一方のみを指定する。
    """

    def __init__(self, warmup_epochs=None, warmup_iters=None):
        assert (warmup_epochs is None) != (warmup_iters is None), (
            'set exactly one of warmup_epochs / warmup_iters')
        self.warmup_epochs = warmup_epochs
        self.warmup_iters = warmup_iters

    def _enable(self, runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model.enable_per_class()
        runner.logger.info(
            '>>> DitHub: END WARMUP, START SPECIALIZATION PHASE <<<')

    def before_train_epoch(self, runner):
        if self.warmup_epochs is not None and \
                runner.epoch == self.warmup_epochs:
            self._enable(runner)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if self.warmup_iters is not None and runner.iter == self.warmup_iters:
            self._enable(runner)
