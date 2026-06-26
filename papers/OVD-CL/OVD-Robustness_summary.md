# Robustness of Open-Vocabulary Detectors under Distribution Shift（要約）

- 出典: *A comprehensive robustness evaluation of open-vocabulary foundation object detection models*. 2024.
- arXiv: 2405.14874
- 領域: OVD地形図／分布シフト頑健性（既調査4領域の外側）

## 中核アイデア
3つの代表的オープン語彙検出器 **OWL-ViT / YOLO-World / Grounding DINO** のゼロショット頑健性を、
COCO-O / COCO-DC / COCO-C などの分布シフト下で**横並びで包括評価**。

## 本研究への含意
- **【ベースライン/背景】**OWL-ViT・OWLv2・YOLO-World は MM-Grounding DINO の**代替バックボーン/比較対象**。
  手法をこれらに移植する一般性の議論や、検出器選択の根拠に使える。
- **【分析設計】**分布シフト系の評価軸（COCO-O/DC/C）は、ZCOCO の「頑健性低下」をドメイン適応前後で測る
  プロトコルの参考になる。
- caveat: 継続学習ではなく**単発ゼロショット頑健性**の評価。CLの忘却軸は別途必要。

## 関連
- 自己学習でのOVD強化: [[OWL-ST_summary]] / 評価基盤: [[RF100-VL_summary]]
</content>
