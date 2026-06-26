# O-LoRA 要約（Orthogonal subspace learning, EMNLP 2023 Findings）

arXiv:2310.14152 (Wang et al.)。本研究の **最重要比較対象**（直交化＋ゼロショット保持の先行実証）。

## 中核
- 継続学習を **LoRA 部分空間の直交学習**で行う。各タスク t で LoRA(A_t, B_t) を学習。
- 過去タスクの **A 行列の列空間を旧タスク勾配部分空間の代理**とみなし、新タスク LoRA をそれと直交化:
  直交損失 `L_orth = Σ_{i<t} ‖A_iᵀ A_t‖²`。全目的 = タスク対数尤度 + λ1·Σ L_orth。
- 旧 LoRA パラメータ {A_i,B_i | i<t} は凍結。勾配や過去データを保存せず **Orthogonal Gradient Descent を近似**＝リハーサル不要。

## 結果（ゼロショット保持の実証）
- Alpaca 調整済み LLaMA-7B を継続学習 → **MMLU: O-LoRA 33.6%**（base ~34.4%）を維持。
  直交化なし LoRA-CL 23.3% / inc-LoRA-CL 28.6%（≈ランダム25%）と崩壊。
  → **「直交化なしの LoRA-CL はゼロショットを破壊」「直交化が保持に寄与」**を示す最も近い先行証拠。

## 本研究との関係・差別化
- 設計思想（過去への干渉を部分空間直交化で抑える）は InfLoRA／本研究と共有。
- InfLoRA との差: O-LoRA は損失で直交を**促す**、InfLoRA は B_t 設計で部分空間に**閉じ込める**（より厳密）。
- **注意**: O-LoRA の保持実証は **LLM(MMLU)** であり、検出 ZCOCO への外挿は未保証。
- 新規性主張: 本研究は「直交化＋**検出のゼロショット(COCO)を task-0 として明示保護**」で O-LoRA を超える。
- 関連: [[InfLoRA_summary]], [[../../experiments/exp_006.5/related_work_survey]]
