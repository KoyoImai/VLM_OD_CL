# A Survey on Open-Vocabulary Detection and Segmentation（要約）

- 出典: Zhu, Chen. *A Survey on Open-Vocabulary Detection and Segmentation: Past, Present, and Future*. 2023 (rev. 2024).
- arXiv: 2307.09220
- 領域: OVD地形図（領域8精査・サーベイ）

## 中核アイデア
オープン語彙検出/セグメンテーションの体系的サーベイ。手法を6軸に整理:
視覚-意味空間マッピング / 新規視覚特徴合成 / 領域認識学習 / 擬似ラベル / 知識蒸留 / 転移学習。

## 本研究への含意（背景・地形図）
- **【背景整理】**OVD の全体地図として引用。GLIP/OWL-ViT/YOLO-World/RegionCLIP/Detic 等の位置づけ確認に有用。
- **【手法軸の示唆】**知識蒸留・擬似ラベル・転移学習の軸は、本研究の ZCOCO 保持（[[../VLM-adaptation/ZSCL_summary]] 蒸留, [[OWL-ST_summary]] 擬似ラベル）と直結。
- caveat: 継続学習の軸は中心でない。一次手法は各原論文を参照。

## 関連
- OVD頑健性: [[OVD-Robustness_summary]] / 評価基盤: [[RF100-VL_summary]]
</content>
