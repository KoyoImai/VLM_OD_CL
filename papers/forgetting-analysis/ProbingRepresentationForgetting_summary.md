# Probing Representation Forgetting in Continual Learning（要約）

- 出典: Davari, Asadi, Mudur, Aljundi, Belilovsky. *Probing Representation Forgetting in Supervised and Unsupervised Continual Learning*. CVPR 2022.
- arXiv: 2203.13381
- 領域: 機序分析の道具箱（領域8精査）

## 中核アイデア
**representation forgetting** を catastrophic forgetting と区別する診断ツールとして提案。
過去タスク性能の低下ではなく、**新タスク導入の前後で最適線形分類器(linear probe)を学習し直して表現の変化を測る**。
結果、標準モデルは明示的な忘却抑制が無くても（特に長系列で）**表現自体の忘却は小さい**ことが多いと示す
→ 「性能低下＝表現崩壊」とは限らない。

## 本研究への含意（分析設計：A2/A3に直結）
- **【分析設計】**ZCOCO/過去ドメインの性能低下が **(i) 共有特徴(表現)の崩壊** によるのか **(ii) ヘッド/対照層の上書き** に
  よるのかを、**linear probing で切り分ける**。MM-GDINO の backbone/encoder 特徴に probe を載せ、忘却前後で probe 精度を比較。
- もし「表現は保たれヘッドが壊れる」なら、LoRA挿入箇所をヘッド/デコーダ側に絞る根拠になる（議事録 分析A の精緻化）。
- caveat: 検出はROI/クエリ単位のため、検出版の probe 設計（領域特徴の線形分類/回帰）が必要。

## 関連
- 表現の構造: [[TunnelEffect_summary]] / モジュール局在化: [[../CL-foundations/EWC_summary]]
</content>
