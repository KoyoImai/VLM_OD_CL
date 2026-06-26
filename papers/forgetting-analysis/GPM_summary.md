# GPM 要約（Gradient Projection Memory, Saha et al., ICLR 2021）

勾配部分空間ベース継続学習の基盤。InfLoRA の旧タスク勾配空間近似(DualGPM)の原型。

## 中核
- 過去タスクの**入力が張る部分空間（重要部分空間）**を保持し、新タスクの更新を**その直交補空間に射影**する。
  → 過去タスクに重要な方向を変えずに学習＝忘却を抑制。
- 関連系: **Adam-NSCL**(Wang, CVPR21, 特徴のnull空間へ射影), **OGD/NSGP**（null空間/直交勾配ファミリー）。

## 本研究との関係（機序分析への流用）
- **分析C/D の理論的基盤**: ドメイン間の「重要部分空間（入力活性化部分空間 or 勾配空間）の重なり」を測れば、
  干渉＝忘却の原因を定量化できる。GPM の部分空間構成（活性化のSVD）をそのまま流用可能。
- InfLoRA は GPM の部分空間概念を LoRA に持ち込んだもの＝手法と分析が同じ言語で語れる。
- 関連: [[../LoRA-CL/InfLoRA_summary]], [[../CL-detection/Demystifying-TwoStage-IOD_summary]], [[../../experiments/exp_006.5/related_work_survey]]
