# Dynamic-DINO: Fine-Grained MoE Tuning for Real-time OVD（要約）

- 出典: Lu et al. *Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection*. ICCV 2025.
- arXiv: 2507.17436
- 領域: OVD×MoE（領域8精査・**同系バックボーンでのMoE実例**）

## 中核アイデア
**Grounding DINO 1.5 Edge** を Mixture-of-Experts で拡張。
FFN を小さなエキスパート群に分解し、事前学習重みの割当戦略＋ルータ初期化で構成。
推論時は**タスク関連エキスパートのみ活性化**しサブネットを形成。浅層は多様な協調、深層は固定的協調パターン。

## 本研究への含意（手法・ベースライン）
- **【重要・同系実装の前例】Grounding DINO 系で MoE を実装した具体例**。
  我々のモジュール式（DitHub/MoE-Adapters系）案を**MM-GDINO上で実現する技術的参照**になる。
- **【手法アイデア】**ドメイン別エキスパート＋ルーティングで、新ドメインは新エキスパート追加・過去/COCOは既存経路保持
  → 忘却抑制とZCOCO保持に転用可能（推論時ルーティングは [[../VLM-adaptation/MoE-Adapters_summary]] と同系）。
- caveat: 主目的は実時間効率で、逐次CL・忘却・ZCOCO保持は未評価。FFN分解の重み割当が前提。

## 関連
- MoE適応/ルーティング: [[../VLM-adaptation/MoE-Adapters_summary]] / モジュール式競合: DitHub_summary
</content>
