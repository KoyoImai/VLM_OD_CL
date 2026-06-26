# Roboflow100-VL: A Multi-Domain Object Detection Benchmark for VLMs（要約）

- 出典: Roboflow100-VL benchmark paper. 2025.
- arXiv: 2505.20612
- 領域: OVD評価ベンチマーク（本研究のデータ基盤の上流）

## 中核アイデア
**VLM事前学習に乏しい概念**を含む 100 検出データセットのベンチマーク。
- 規模: **564 クラス / 164,149 画像 / 1,355,491 アノテーション / 7 ドメイン**
  （Aerial, Document, Flora & Fauna, Industrial, Medical, Sports, Other）。
- 適応問題の定量化: **GroundingDINO は ODinW-13 で 49.2 mAP だが RF100-VL ではゼロショット ~16 mAP**
  （全体 15.7、**medical は <2%**）— 深刻なドメインギャップ。

## 本研究への含意
- **【評価基盤】**本研究の `rf100_domain`（6ドメイン）の上流。RF100-VL を**ドメイン増加CLのタスク系列**として用い、
  かつ「適応問題が深刻に存在する」ことの定量的根拠（課題1）として引用できる。
- **【ドメイン選定】medical / microscopic / electromagnetic 等の VLM非典型ドメインが最難**
  → 適応と保持のトレードオフを最も強くテストするケースとして選定する指針。
- caveat: 単一だが正準な一次ベンチマーク。本研究は real_world(405クラス) を除外する運用である点に注意。

## 関連
- データ統計の正本: ../../README.md / RF100原論文: papers/Roboflow100_summary
</content>
