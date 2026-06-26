# The Tunnel Effect: Building Data Representations in Deep Networks（要約）

- 出典: Masarczyk, Ostaszewski, Imani, Pascanu, Miłoś, Trzciński. *The Tunnel Effect*. NeurIPS 2023.
- arXiv: 2305.19753
- 領域: 機序分析の道具箱（領域8精査）

## 中核アイデア
十分に深いネットは2領域に自然分割される:
- **extractor（前段）**: 線形分離可能な表現を形成。
- **tunnel（中後段）**: 表現を**圧縮**し、最終性能への寄与は小さいが**OOD/転移性能を損なう**。
学習初期から出現し、容量とタスク難度の比で tunnel の深さが決まる。

## 本研究への含意（分析設計＋手法）
- **【分析設計】**MM-GDINO のどの深さで表現が圧縮（tunnel化）しているかを層別 probe（[[ProbingRepresentationForgetting_summary]]）で同定。
  **ZCOCO/転移の劣化が tunnel 層に起因**するなら、LoRA/凍結方針を層深度で設計する根拠になる。
- **【手法】**tunnel 層を凍結 or 低ランク適応に限定し、extractor を保護する層別戦略。
- caveat: 分類CNN/ViTでの知見。検出Transformer・言語枝への転移は要検証。

## 関連
- 表現忘却の測定: [[ProbingRepresentationForgetting_summary]]
</content>
