# exp_008 sweep / underwater: lr=5e-3, rank=16, alpha=16(scaling=1)
_base_ = './underwater_lora.py'
model = dict(lora=dict(r=16, alpha=16, include=['encoder', 'decoder', 'bbox_head']))
optim_wrapper = dict(optimizer=dict(lr=5e-3))
