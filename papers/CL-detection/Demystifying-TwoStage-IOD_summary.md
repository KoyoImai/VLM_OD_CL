# 二段増分物体検出の忘却 解剖 要約（Demystifying CF in Two-Stage IOD, ICML 2025）

arXiv:2502.05540 (Wu et al.)。NSGP-RePRE を提案。**本研究の機序分析(C2)に最も直接流用できる論文。**

## 機序の核（検出特有の忘却）
- 二段検出器(Faster R-CNN)では破滅的忘却が **RoI Head の分類器に主に局在**し、
  **バウンディングボックス回帰器は増分段階を通じて頑健**（RPNのみの忘却は ~1.3% mAP）。
- **固定提案(fixed proposals)を用いた解剖実験**で、回帰を分離すると結果ほぼ不変・分類駆動の予測は急速劣化、と検証。
  → 「忘却は局在(localization)でなく分類(classification)に影響」。

## 手法（NSGP）
- **Null Space Gradient Projection**: 旧入力が張る部分空間に直交する方向で特徴抽出器を更新。
  GPM(Saha ICLR21)/Adam-NSCL(CVPR21)/**InfLoRA の直交化原理と同族**（null空間/直交勾配ファミリー）。

## 本研究との関係
- **分析手法の流用**: 「固定提案で分類/局在を切り分ける解剖」を Grounding DINO に適用し、
  クエリ/対照ヘッド・テキストエンコーダのどこで忘却するか（モジュール局在化＝分析A）を測る設計に使える。
- **重要な留保**: これは**二段Faster R-CNN**の知見。Grounding DINO（クエリ＋テキスト対照）に
  そのまま転移する保証はなく、回帰の忘却もゼロでなく'minimal'、著者も機序は'unclear'。
  → **転移可否の検証自体が本研究の貢献余地**。
- 関連: [[../forgetting-analysis/GPM_summary]], [[../LoRA-CL/InfLoRA_summary]], [[../../experiments/exp_006.5/related_work_survey]]
