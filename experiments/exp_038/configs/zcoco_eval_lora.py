# =============================================================================
# exp_038: ZCOCO 評価（COCO2017-val ゼロショット）/ **マージ前**の LoRA モデル
#
#   実装検証用。通常の逐次評価は plain な θt を測る zcoco_eval.py を使う。
#   本 config は「マージ前（base + LoRA）の ZCOCO」と「マージ後 θt の ZCOCO」を
#   突き合わせ、ZCOCO の低下が merge_lora.py に起因するのか、LoRA の学習結果
#   そのものに起因するのかを切り分けるためのもの。
#
#   lora の設定は lora_odinw13_base.py と同一でなければならない（ckpt の
#   state_dict の構造が一致しないと読めないため）。
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/eval_base_coco.py'

custom_imports = dict(
    imports=['projects.lora_cl'], allow_failed_imports=False)

model = dict(
    type='GroundingDINOLoRA',
    lora=dict(
        r=16,
        alpha=8,
        exclude_components=[],
        decompose_mha=[r'^encoder\.text_layers\.\d+\.self_attn\.attn$'],
        include=['backbone', 'language_model', 'text_feat_map', 'encoder']))
