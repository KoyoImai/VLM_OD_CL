# WiSE-FT: Robust Fine-Tuning of Zero-Shot Models（要約）

- 出典: Wortsman et al. *Robust fine-tuning of zero-shot models*. CVPR 2022.
- arXiv: 2109.01903
- 領域: 基盤モデル適応／重み補間によるゼロショット保持（既調査4領域の外側）

## 中核アイデア
標準的 fine-tune は**分布シフトへの頑健性（＝ゼロショット性能）を損なう**。
**WiSE-FT**: ゼロショット重みと fine-tune 後重みを**線形補間（重みアンサンブル）** `θ = (1−α)θ_zs + α θ_ft`。
**追加の推論コストゼロ**で、目標分布性能を保ちつつ頑健性/ゼロショットを回復。

## 本研究への含意
- **【手法・ベースライン】**「凍結事前学習重み ↔ ドメイン適応重み」の WiSE-FT 流補間は、
  **安価で強力な ZCOCO 保持ベースライン**であり、C3 手法の**候補成分**（適応と保持のトレードオフを α で制御）。
- 推論コスト・メモリ増加なし（単一重み）→ DitHub（モジュール増加）に対する**単純さの優位**を主張する材料。
- caveat: CLIP/分類での検証。干渉する成分があると一律補間は劣化しうる。検出器の box/text ヘッド補間は要適応。

## 関連
- 平均化の一般化: [[ModelSoups_summary]] / 編集プリミティブ: [[TaskArithmetic_summary]]
</content>
