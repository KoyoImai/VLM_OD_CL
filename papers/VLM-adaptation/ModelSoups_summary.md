# Model Soups: Averaging Weights of Multiple Fine-Tuned Models（要約）

- 出典: Wortsman et al. *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time*. ICML 2022.
- arXiv: 2203.05482
- 領域: 基盤モデル適応／重み平均（既調査4領域の外側）

## 中核アイデア
**同一初期値**から fine-tune した複数モデルの**重みを平均**すると、精度・OOD頑健性・**新規下流タスクのゼロショット性能**が向上。
**推論/メモリコスト増なし**（単一重みに統合）。一律平均が劣化する場合は **greedy soup**（性能が上がる成分のみ追加）。

## 本研究への含意
- **【手法・ベースライン】ドメイン別に適応した（同一初期＝凍結事前学習からの）モデル/LoRA を平均統合**する
  マージ系ベースライン。パラメータ一定で複数ドメインを蓄積する候補（DitHub のモジュール増加に対する対案）。
- **【分析】**「soup 可能性は loss landscape の幾何に依存」→ ドメイン間の mode connectivity / 線形補間経路の
  損失を測る分析（C2）の動機になる。
- caveat: タスク数増で干渉により劣化（→ greedy/TIES 等の選択的統合が必要）。CLIP/分類・NLP での検証。

## 関連
- 補間: [[WiSE-FT_summary]] / 加減算編集: [[TaskArithmetic_summary]]
</content>
