# EWC: Overcoming Catastrophic Forgetting in Neural Networks（要約）

- 出典: Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks*. PNAS 2017.
- arXiv: 1612.00796
- 領域: CL基礎・正則化系の正準手法（既調査4領域の外側）

## 中核アイデア
**Elastic Weight Consolidation (EWC)**: 過去タスクで重要な重みの学習を選択的に減速させる。
重要度を **Fisher 情報行列**で測り、`L = L_new + Σ_i (λ/2) F_i (θ_i − θ*_i)²` の二次ペナルティで過去最適点近傍に拘束する。

## 本研究への含意
- **ベースライン**: 正則化系の正準手法。勾配射影系(我々/GPM)・リプレイ系・アーキテクチャ系と並ぶ比較対象。
  online-EWC も含めて検討。
- **分析ツールへの転用**: **Fisher 情報を重み重要度の指標**として使い、「どのモジュール/層が各ドメインで重要か」
  を可視化できる（C2 機序分析: 忘却のモジュール局在化を Fisher 重みで補強）。
- caveat: 二次近似は逐次タスクで誤差が累積。検出器の大規模化で Fisher 推定コストに注意。

## 関連
- 勾配射影系（同じ忘却抑制原理の別系統）: [[NTK-overlap_summary]] / papers/forgetting-analysis/GPM_summary
</content>
