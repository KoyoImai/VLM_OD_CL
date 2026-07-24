# =============================================================================
# exp_026 評価: microscopic valid（ゼロショット再確認用）
#   既存 finetune config を継承し、valid のデータ・評価プロトコル（解像度 800,1333 /
#   batch_size=1 / CocoMetric bbox）をそのまま再利用する。
#   θ0（num_classes=256）に合わせて num_classes を 256 に上書きする。
#   label_embedding は学習時専用（推論では if self.training: ガードで不使用）のため
#   評価 mAP には影響しないが、サイズ不一致の警告を避けるために揃える。
#   microscopic のクラス名は 256 トークン以内に収まるため chunked 推論は不要
#   （実測: aerial 77 / microscopic 131 / documents 207 トークン。videogames のみ 265 で要 chunk）。
#   既存ファイルは無変更。
# =============================================================================
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_microscopic.py'  # noqa

model = dict(bbox_head=dict(num_classes=256))
