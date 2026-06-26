# exp_008 sweep / underwater: lr=1e-4, rank=32, alpha=32(scaling=1)
_base_ = './underwater_lora.py'
model = dict(lora=dict(r=32, alpha=32, include=['encoder', 'decoder', 'bbox_head']))
optim_wrapper = dict(optimizer=dict(lr=1e-4))
