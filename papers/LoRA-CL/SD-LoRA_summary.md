# SD-LoRA 要約（Scalable Decoupled LoRA, ICLR 2025 Oral）

arXiv:2501.13198。LoRA ベース継続学習の比較対象（パラメータ効率・選択機構排除の観点）。

## 中核
- LoRA 成分の **大きさ(magnitude)と方向(direction)を分離**して学習。
- 過去タスクの**方向を凍結**し、その大きさ＋新方向のみ学習することでクラス増分学習。
- **リハーサル不要**かつ **コンポーネント選択機構なし**で直接推論可能。
  → プロンプト/LoRA プールの拡張やタスクリハーサルに依存する既存手法のスケーラビリティ問題を回避。

## 本研究との関係
- InfLoRA・O-LoRA と並ぶ LoRA-CL 比較対象。特に「パラメータ拡張を抑える／推論時の選択機構を排する」観点での代替設計。
- 本研究（InfLoRA型・統合でパラメータ一定）と「拡張をどう抑えるか」を対比できる。
- 関連: [[InfLoRA_summary]], [[O-LoRA_summary]], [[../../experiments/exp_006.5/related_work_survey]]
