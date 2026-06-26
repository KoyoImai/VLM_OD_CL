# exp_008 sweep / underwater: lr=1e-3, rank=64, alpha=64(scaling=1)
_base_ = './underwater_lora.py'
model = dict(lora=dict(r=64, alpha=64, include=['encoder', 'decoder', 'bbox_head']))
optim_wrapper = dict(optimizer=dict(lr=1e-3))
