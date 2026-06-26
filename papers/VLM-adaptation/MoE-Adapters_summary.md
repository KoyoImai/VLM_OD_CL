# MoE-Adapters: Boosting Continual Learning of CLIP via Mixture-of-Experts Adapters（要約）

- 出典: Yu et al. *Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters*. CVPR 2024.
- arXiv: 2403.11549
- 領域: 基盤モデル適応／推論時ルーティング（既調査4領域の外側）

## 中核アイデア
凍結 CLIP に **MoE アダプタ**を追加し、**DDAS (Distribution Discriminative Auto-Selector)** で入力をルーティング:
- **in-distribution 入力 → 学習済み MoE アダプタ**（適応）。
- **out-of-distribution 入力 → 元の凍結 CLIP**（ゼロショット保持）。
OOD 判定はタスク別オートエンコーダで行う。

## 本研究への含意
- **【手法・別パラダイム】学習時の直交化(InfLoRA型)とは別に、推論時ルーティングで ZCOCO を保持**する道。
  「OOD（=未学習/COCO的入力）は凍結経路へ、in-dist はドメインアダプタへ」。
- **【ベースライン】**MoE/ルーティング系の比較対象。モジュール選択は DitHub と発想が近い。
- caveat: **CLIP分類で検証**。検出は**領域(region)ごとのOODルーティング**が必要で、論文は未実証
  → 検出に literal 適用するには非自明な拡張が要る（=我々の貢献余地 or 実装難所）。

## 関連
- 重み統合系（推論コスト増なし）との対比: [[ModelSoups_summary]] / モジュール式競合: papers/OVD-CL/DitHub_summary
</content>
