# 内在次元 要約（Intrinsic Dimensionality Explains the Effectiveness of LM Fine-Tuning, Aghajanyan et al., ACL 2021）

LoRA の低ランク性の理論的根拠。本研究の **分析B（ΔW の低ランク性）** の裏付け。

## 中核
- 事前学習モデルの fine-tune は **低い内在次元（少数の有効自由度）**で行える。
  → 大きな重み空間でも、適応に必要な変化は低次元部分空間に収まる。
- これが LoRA（重み変化を低ランク AB で近似）が有効である根拠。

## 本研究との関係
- **分析B**: 各ドメイン単独FTの ΔW を SVD し「適応が少数主成分に集中」することを確認する分析は、
  本論文の主張を検出器（MM-GDINO）で再確認する位置づけ → LoRA 採用と rank 選択を正当化。
- 関連: [[GPM_summary]], [[../LoRA-CL/InfLoRA_summary]], [[../../experiments/exp_006.5/related_work_survey]]
