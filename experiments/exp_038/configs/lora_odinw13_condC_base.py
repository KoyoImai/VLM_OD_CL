# =============================================================================
# exp_038 条件C: 学習率 1e-4 の素の LoRA の ODinW-13 逐次学習 共通ベース
#
#   条件A（lora_odinw13_base.py）との差分は lr のみ（design.md §11.2）。
#     条件A: AdamW lr 1e-3 / wd 1e-2、iter 1200 で ×0.1 → 1e-4
#     条件C: AdamW lr 1e-4 / wd 1e-2、iter 1200 で ×0.1 → 1e-5
#
#   挿入箇所は条件A と同一の 232 層（Swin 51 + BERT 72 + encoder 108 + text_feat_map 1）。
#   weight_decay は 1e-2 のまま（2 要因を同時に動かすと lr の効果を切り分けられないため）。
#   Swin / BERT への lr_mult による抑制も行わない（それは別の軸。design.md §11.2）。
#   その他（3000 iter、batch 2、grad clip 0.1(L2)、dn 無効、num_classes 256、
#   COCO 形式 pipeline、seed 0、各タスク後に base へマージ）は条件A と完全に同一。
# =============================================================================
_base_ = './lora_odinw13_base.py'

optim_wrapper = dict(optimizer=dict(lr=0.0001))
