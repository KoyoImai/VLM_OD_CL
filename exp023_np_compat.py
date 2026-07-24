"""numpy 互換シム（exp_023 リプレイ用・既存コード無変更）.

既存の RandomSamplingNegPos（mmdet/datasets/transforms/text_transformers.py）が
`np.long` を使うが、numpy>=1.24 で削除された（現環境 1.26）。削除された別名を
`np.int64` として復元する。config の custom_imports から読み込まれ、main プロセスで
適用される（fork される dataloader worker にも継承）。既存の mmdet コードは変更しない。
"""
import numpy as np

if not hasattr(np, 'long'):
    np.long = np.int64
