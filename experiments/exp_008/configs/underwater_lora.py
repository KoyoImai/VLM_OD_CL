# exp_008 / underwater: 単純 LoRA fine-tune
# ドメイン config を継承し、model を GroundingDINOLoRA に差し替え（base 全凍結＋LoRA のみ学習）。
# 対象は ◎ の nn.Linear（encoder/decoder/bbox_head）。MHA out_proj は自動スキップ。
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py'
custom_imports = dict(imports=['projects.lora_cl'], allow_failed_imports=False)

model = dict(
    type='GroundingDINOLoRA',
    lora=dict(
        r=16, alpha=16,
        include=['encoder', 'decoder', 'bbox_head'],
        # 種別/正規表現でさらに絞る場合は include_types / include_patterns / exclude_patterns を追加
    ))

# 学習設定（通常学習 exp_003〜006 と統一: lr=1e-4・実効64・seed0）。
train_dataloader = dict(batch_size=8)
optim_wrapper = dict(
    constructor='TrainableParamsConstructor',   # trainable(=LoRA)のみ optimizer に登録
    accumulative_counts=2,
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4))
randomness = dict(seed=0, deterministic=False)
