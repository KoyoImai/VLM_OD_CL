# MM-Grounding DINO 要約（学習・評価設定の観点）

出典: MM-Grounding DINO（arXiv:2401.02361v2）。本プロジェクトのベースモデル。

## 概要
Grounding DINO を MMDetection 上で再実装・拡張したオープン語彙物体検出モデル。
OVD / Phrase Grounding / REC を統一的に扱い、obj365+GoldG+GRIT+V3Det で事前学習。

## 入力スケール
- 論文本文には具体的なピクセル値（短辺/長辺）の明記なし。
- 実装（mmdet config）側の既定：学習は多スケール `RandomChoiceResize`（短辺 480〜800, 長辺上限 1333）、
  評価は `FixScaleResize (800, 1333)`。**論文ではなく config が正本**。

## データ拡張（§2.3）
- **RandomResize（多スケール）**, **RandomCrop**, **RandomFlip** を使用。
- **Random Negative Sampling**：他画像のカテゴリ/説明を負例テキストとして連結し幻覚を抑制。
- Color jitter / Mosaic / MixUp は使用していない（記載なし）。

## 学習スケジュール
- 事前学習: 30 epochs, total batch 128, 32×3090, クエリ数 900, テキストは BERT-base-uncased。
- **ファインチューニング**（COCO/LVIS/下流）:
  - close-set/open-vocabulary FT は概ね **12 epochs（1x）**。
  - 下流タスクはデータ規模により 12 または 50 epochs。RefCOCO 系は 5 epochs で改善。
- 具体的な lr / optimizer / wd は本文に明記なし（config 参照）。

## テキスト処理
- OVD はカテゴリを `.` 区切りで連結（例 `"People. Ball. Racket. Cat."`）。
- `max_text_len` の具体値は本文になし（実装既定 256、本プロジェクトでは多クラス時 512 に拡張）。

## 本プロジェクトへの示唆
- 我々の underwater config（20 epochs, lr 1e-4, backbone/LM 凍結）は論文の FT 流儀（短め・head中心）に整合。
- 拡張は「flip + 多スケール + crop + 負例サンプリング」が標準。重い拡張（mosaic/mixup）は非採用が論文流儀。
- 関連: [[ZiRaGroundingDINO_summary]]（同モデルでの継続学習）, [[Roboflow100_summary]]（評価データ）。
