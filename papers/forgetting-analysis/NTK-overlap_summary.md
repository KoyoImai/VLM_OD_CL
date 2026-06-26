# NTK Overlap Matrix: A Theoretical Analysis of Catastrophic Forgetting（要約）

- 出典: Doan et al. *A Theoretical Analysis of Catastrophic Forgetting through the NTK Overlap Matrix*. AISTATS 2021.
- arXiv: 2010.04003
- 領域: 忘却機序の理論（既調査の GPM/Intrinsic Dim を補完）

## 中核アイデア
NTK（Neural Tangent Kernel）領域で破滅的忘却を解析。
- **NTK overlap matrix** が忘却の中核量。**タスク間のアラインメントが強いほど忘却(CF)が増大**する。
- 投影勾配法（OGD 等）が忘却をどう緩和するかを理論的に示す。

## 本研究への含意（分析設計の核）
- **【分析設計】タスク間アラインメント（NTK重なり）を測り、忘却リスクを予測・説明する**。
  各新ドメイン × 過去ドメイン、さらに **COCO を task-0** として含めた重なり行列を作れば、
  「重なり量 ∝ 忘却量／ZCOCO低下」を定量化できる（議事録の分析C/Dの理論的裏付け）。
- **【手法正当化】**「過去タスクの勾配空間と直交させる(InfLoRA型)」設計が忘却を減らす理論的根拠を与える。

## caveat
- NTK領域（無限幅・過剰パラメータの理想化）での保証であり、実機では劣化しうる。
- 別研究(2401.12617)は**忘却がタスク類似度に非単調**（中程度の類似で最大）と報告。
  → NTK重なりは「忘却リスクの予測子」であって「不変の法則」ではない。
- 直交勾配の頑健性証明: [[OGD_summary]]

## 関連
- 同族の勾配射影: papers/forgetting-analysis/GPM_summary / [[../CL-foundations/EWC_summary]]
</content>
