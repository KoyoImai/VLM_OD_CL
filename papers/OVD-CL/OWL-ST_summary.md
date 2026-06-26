# OWL-ST: Scaling Open-Vocabulary Detection via Self-Training（要約）

- 出典: Minderer, Gritsenko, Houlsby. *Scaling Open-Vocabulary Object Detection*. NeurIPS 2023.
- arXiv: 2306.09683
- 領域: OVD地形図／Web規模自己学習（既調査4領域の外側）

## 中核アイデア
**OWL-ST**: Web規模の擬似ラベルデータでの**自己学習(self-training)**でオープン語彙検出を拡張。
- **LVIS rare クラスで 31.2 → 44.6 AP（+43% 相対）**。人手box無しのクラスでゼロショット/希少クラス検出が大幅向上。

## 本研究への含意
- **【手法アイデア】自己学習/擬似ラベルが可塑性(適応)とゼロショットを同時に押し上げる**ことの実証。
  → 未ラベル/Web画像での**擬似ラベル・リプレイ**（COCO的概念やゼロショット概念を擬似ラベルで復習）の手掛かり。
  exemplar を保持しない ZCOCO 保持の代替経路になりうる。
- **【ベースライン/可塑性ブースタ】**ドメイン適応時の擬似ラベル併用の比較対象。
- caveat: 継続学習文脈ではなく事前学習スケーリングの研究。計算コスト大。

## 関連
- OVD頑健性: [[OVD-Robustness_summary]] / ゼロショット保持の蒸留版: [[../VLM-adaptation/ZSCL_summary]]
</content>
