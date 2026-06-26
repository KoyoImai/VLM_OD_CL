# CODA-Prompt 要約（COntinual Decomposed Attention-based Prompting, CVPR 2023）

arXiv:2211.13218。プロンプトベース継続学習（L2P/DualPrompt 系）の代表的比較対象。

## 中核
- **リハーサル不要**の継続学習。凍結した事前学習 ViT 上で、**入力条件付き重みにより合成されるプロンプト成分集合**を学習。
- クラス増分・**ドメイン増分**の両設定に対応。**平均最終精度**を主指標。
- DualPrompt 比で最大 +4.5% の精度改善。

## 本研究との関係
- プロンプト系 CL（L2P→DualPrompt→CODA-Prompt）の到達点として比較対象。
- **ドメイン増分設定＋平均最終精度**という評価設計は、本研究の CL 指標（Avg）設計の参考。
- LoRA ではなくプロンプトで適応する点が本研究（LoRA）との対比軸。
- 関連: [[../../experiments/exp_006.5/related_work_survey]]
