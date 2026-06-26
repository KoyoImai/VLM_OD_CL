"""trainable（requires_grad=True）パラメータのみを optimizer に登録する constructor。
mmengine 既定は凍結 param も param_groups に含めるため、LoRA 学習で誤解を避けるべく明示的に絞る。
"""
import copy
from mmengine.registry import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS,
                               OPTIMIZERS)
from mmengine.optim import DefaultOptimWrapperConstructor


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class TrainableParamsConstructor(DefaultOptimWrapperConstructor):
    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module
        optim_wrapper_cfg = copy.deepcopy(self.optim_wrapper_cfg)
        optim_wrapper_cfg.setdefault('type', 'OptimWrapper')
        optimizer_cfg = copy.deepcopy(self.optimizer_cfg)
        # requires_grad=True のみ（= LoRA のみ）
        optimizer_cfg['params'] = [p for p in model.parameters() if p.requires_grad]
        optimizer = OPTIMIZERS.build(optimizer_cfg)
        optim_wrapper = OPTIM_WRAPPERS.build(
            optim_wrapper_cfg, default_args=dict(optimizer=optimizer))
        return optim_wrapper
