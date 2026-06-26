# Task Arithmetic: Editing Models with Task Vectors（要約）

- 出典: Ilharco et al. *Editing Models with Task Arithmetic*. ICLR 2023.
- OpenReview: id=6t0Kwf8-jrj
- 領域: 基盤モデル適応／訓練不要のモデル編集（既調査4領域の外側）

## 中核アイデア
**task vector = (fine-tune 後重み − 事前学習重み)**。重み空間の方向ベクトルとして編集できる。
- **加算**: 複数 task vector を足すと**一度に複数タスクで性能向上（マージ）**。
- **減算(否定)**: task vector を引くと**対象タスクの性能のみ低下、他タスクはほぼ不変（選択的アンラーニング）**。

## 本研究への含意
- **【手法・ベースライン】task vector 加算 = ドメインを蓄積しつつ COCO を保つマージ系ベースライン**
  （joint 再学習なしで多ドメイン化）。
- **【分析ツール】否定 = 制御された忘却実験**に使える。特定ドメインを「引いて」何が壊れるかを見ることで、
  ドメインごとの重み方向と干渉構造を C2 分析できる。
- caveat: タスク数増で**重み干渉・符号衝突**により精度低下。
  → **TIES-Merging (2306.01708) / DARE / Task Singular Vectors (CVPR2025)** が改良かつ C2 の部分空間/符号衝突分析の手掛かり。CLIP/分類・NLP 検証。

## 関連
- 平均統合: [[ModelSoups_summary]] [[WiSE-FT_summary]]
</content>
