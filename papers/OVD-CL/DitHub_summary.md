# DitHub 要約（Modular incremental Open-Vocabulary Detection, NeurIPS 2025）

arXiv:2503.09271 (AImageLab UniMORE)。**ZiRa と並ぶ最直接の競合**（増分OVD）。

## 中核
- 増分オープン語彙物体検出を **モジュール式**で扱う。効率的な適応(エキスパート)モジュールのライブラリを
  **Version Control System のブランチのように管理（fetch/merge）**し、単一の一枚岩な適応重みを使わない。
- 希少クラスへの適応と複数専門ドメインでの能力強化を、**テキストプロンプトの汎化を活かしつつ**行う。
- ODinW-13 等で評価。

## 本研究との関係・差別化
- LoRA プール/混合・モジュール統合系の最新代表。**本研究の新規性主張で必ず差別化すべき競合**。
- vs 本研究(InfLoRA型・統合でパラメータ一定): DitHub は**モジュールが増える**。
  本研究は「**パラメータ拡張一定＋勾配直交化＋明示的ゼロショット(COCO)保護**」で差別化。
- モジュール局在化の発想は本研究の機序分析(どこを適応すべきか)とも関連。
- 関連: [[../ZiRaGroundingDINO_summary]], [[../../experiments/exp_006.5/related_work_survey]]
