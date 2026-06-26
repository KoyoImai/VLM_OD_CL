# OGD: Orthogonal Gradient Descent for Continual Learning（要約）

- 出典: Bennani, Doan, Sugiyama. *Generalisation Guarantees for Continual Learning with Orthogonal Gradient Descent*. 2020.
- arXiv: 2006.11942
- 領域: 忘却機序の理論／勾配直交化の正当化（既調査4領域の外側）

## 中核アイデア
**Orthogonal Gradient Descent (OGD)**: 新タスクの勾配を過去タスクの勾配（が張る空間）に直交する方向へ射影して更新する。
NTK 領域で **OGD が破滅的忘却に頑健であること（generalisation guarantee）を証明**。

## 本研究への含意（手法の理論的支柱）
- 我々の **InfLoRA 型「更新部分空間を過去タスク勾配と直交化」**の直接的な理論的支柱。
  「直交化 → 忘却抑制」が NTK 領域で保証されるため、提案手法の動機を理論で補強できる。
- **【手法】**OGD 自体を（LoRA分岐に対して）簡易ベースラインとして実装する余地。
- caveat: NTK領域の理想化に依存。[[NTK-overlap_summary]] と同じ実機劣化の注意。

## 関連
- 忘却とアラインメント: [[NTK-overlap_summary]] / 活性化部分空間版: papers/forgetting-analysis/GPM_summary
</content>
