# Continual Learning in Low-rank Orthogonal Subspaces（要約）

- 出典: Chaudhry, Khan, Dokania, Torr. *Continual Learning in Low-rank Orthogonal Subspaces*. NeurIPS 2020.
- arXiv: 2010.11635
- 領域: 低ランク直交部分空間CL（領域8精査・**提案手法の重要な先行**）

## 中核アイデア
各タスクを**互いに直交する低ランク部分空間**に割り当てて学習。
Stiefel 多様体上で重みを最適化して等長写像を学び、**タスク勾配を直交化**して干渉を最小化。
エピソードメモリや正則化に対する代替ベースライン。

## 本研究への含意（手法・位置づけの核）
- **【最重要・位置づけ】「低ランク × 直交部分空間」という本研究(InfLoRA型 LoRA + 直交化)の直接的な先行**。
  → 我々の新規性は「**事前学習VLM検出器**で・**COCOをtask-0として直交保護**し・**ゼロショット(ZCOCO)保持**を同時に狙う」点に
    絞られることが明確になる。差別化の記述に必須。
- **【ベースライン】**低ランク直交部分空間法そのものを比較対象に。
- caveat: 分類ベンチでの検証。検出・VLMへの適用とゼロショット保持は扱っていない（＝我々の貢献余地）。

## 関連
- 勾配射影: [[PCGrad_summary]] [[OGD_summary]] / 活性化部分空間: GPM_summary / 既調査の InfLoRA: ../LoRA-CL/InfLoRA_summary
</content>
