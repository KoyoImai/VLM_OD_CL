# =============================================================================
# exp_038 条件B: BERT を凍結した素の LoRA の ODinW-13 逐次学習 共通ベース
#
#   条件A（lora_odinw13_base.py、232 層）との差分は lora.include から
#   'language_model' を外す 1 点だけ（design.md §10.2）。
#     条件A: backbone 51 + language_model 72 + encoder 108 + text_feat_map 1 = 232
#     条件B: backbone 51 +                     encoder 108 + text_feat_map 1 = 160
#
#   exclude_components を ['language_model'] に明示して既定除外へ戻す。
#   その他（r=16 / alpha=8、3000 iter、batch 2、AdamW lr 1e-3 / wd 1e-2、
#   iter 1200 で ×0.1、grad clip 0.1(L2)、dn 無効、num_classes 256、
#   COCO 形式 pipeline、seed 0、各タスク後に base へマージ）は条件A と完全に同一。
# =============================================================================
_base_ = './lora_odinw13_base.py'

model = dict(
    lora=dict(
        r=16,
        alpha=8,
        exclude_components=['language_model'],   # BERT を対象から外す
        decompose_mha=[r'^encoder\.text_layers\.\d+\.self_attn\.attn$'],
        include=['backbone', 'text_feat_map', 'encoder']))
