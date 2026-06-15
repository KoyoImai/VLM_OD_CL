# exp_000 メモ：事前学習 MM-Grounding DINO の zero-shot 評価（underwater）

## 結論
fine-tune 前の素の MM-Grounding DINO は、underwater ドメインで **mAP=0.051（mAP_50=0.075）** にとどまる。
これを本研究における underwater の **出発点（lower bound）** として記録する。

## 観察
- **AP は非常に低いが AR は中程度**（AR@100=0.207）。物体の位置はある程度拾えているが、
  正しいクラス名に対応づけられていない／スコアが低いことが示唆される。
  → Grounding DINO は領域提案は汎用的だが、ドメイン固有のテキスト（クラス名）との対照が弱い。
- **small 物体が特に弱い**（mAP_s=0.009, AR_s=0.105）。640×640 リサイズ＋小型海中生物の影響。
- underwater のクラスには `Caespitose-a/b`, `Columnar`, `Corymbose`, `Massive-Faviidae` など
  サンゴの専門分類名が多く、事前学習の語彙・概念に乏しいと考えられる。これが低 AP の主因の仮説。

## 妥当性チェック
- 評価はエラーなく完走（3,576 枚）。num_classes=28 のチェックポイントロードも問題なし。
- mAP が 0 ではなく 0.05 程度出ているため、クラス名とアノテーションの対応は概ね機能している
  （順序ズレなら 0 近傍になりやすい）。ただしクラス別 AP は未確認。

## 次の問い・改善案
1. **クラス別 AP の分解**：汎用語（fish, shark, jellyfish, starfish 等）と専門語（サンゴ分類）で
   AP がどれだけ違うか。低 AP が「未知概念」起因か「テキスト表現」起因かを切り分ける。
   → `classwise=True` を val_evaluator に設定して再評価すれば取得可能。
2. **プロンプト工夫**：専門クラス名を説明的な表現（例: "Columnar coral"）に置換した場合の zero-shot 改善。
   忘却対策の前に「テキスト側でどこまで稼げるか」を把握する。
3. **fine-tune との対比（exp_001 候補）**：underwater 単体 fine-tune の到達 mAP を測り、
   zero-shot 0.051 からの伸び＝「ドメイン適応の余地」を定量化する。これが継続学習の upper bound。
4. **他ドメインへの横展開**：同じ zero-shot 評価を aerial / microscopic 等にも適用し、
   ドメインごとの lower bound 表を作る。忘却の影響が大きいドメインを事前に把握できる。
