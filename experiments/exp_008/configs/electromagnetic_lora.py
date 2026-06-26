# exp_008 / electromagnetic: 単純 LoRA fine-tune（underwater_lora と同設定、_base_ のみ差し替え）
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_electromagnetic.py'
custom_imports = dict(imports=['projects.lora_cl'], allow_failed_imports=False)

model = dict(
    type='GroundingDINOLoRA',
    lora=dict(r=16, alpha=16, include=['encoder', 'decoder', 'bbox_head']))

train_dataloader = dict(batch_size=8)
optim_wrapper = dict(
    constructor='TrainableParamsConstructor',   # trainable(=LoRA)のみ optimizer に登録
    accumulative_counts=2,
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4))
randomness = dict(seed=0, deterministic=False)
