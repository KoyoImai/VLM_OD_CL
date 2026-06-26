# C-Flat: Make Continual Learning Stronger via Flat Minima（要約）

- 出典: Bian et al. *Make Continual Learning Stronger via C-Flat*. 2024.
- arXiv: 2404.00986
- 領域: 損失地形・最適化（領域8精査）

## 中核アイデア
**C-Flat (Continual Flatness)**: CL で**平坦な極小(flat minima)**を狙う plug-and-play 最適化。
鋭い極小ではなく「近傍が一様に低損失」な解を探し、既存CL手法に上乗せして汎化/忘却抑制を改善。

## 本研究への含意（分析A4＋手法上乗せ）
- **【手法】**SAM 系の平坦化を提案手法(直交化LoRA)に**直交的に上乗せ**できる安価な改良候補。
- **【分析設計 A4】**「**sharpness(鋭さ)↔忘却量**」の相関を測る分析。適応後の解の鋭さを忘却前後で比較し、
  平坦な解ほど ZCOCO/過去ドメインを保つかを検証。
- caveat: 主に分類CLでの検証。検出損失での平坦化コスト（2回逆伝播）に注意。

## 関連
- 損失地形: [[LinearModeConnectivity_summary]]
</content>
