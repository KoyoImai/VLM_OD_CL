# CL-DETR 要約（Continual Detection Transformer, CVPR 2023）

arXiv:2304.03110。クエリベース検出器(DETR)の継続学習の代表的比較対象。

## 中核
- 増分物体検出(IOD)は破滅的忘却に陥る。**汎用の知識蒸留(KD)・事例リプレイ(ER)は Deformable DETR / UP-DETR
  に直接適用しても効果が低い** → Transformer 検出器**固有の適応**が必要。
- KD と ER を検出器向けに改良し、COCO2017 の IOD 設定で **当時(2023)の SOTA**。

## 本研究との関係
- Grounding DINO もクエリベース検出器であり、DETR 系 CL の知見・比較対象として重要。
- 「汎用 KD/ER がそのままでは効かない」は、本研究で素朴な対策の限界を示す根拠。
- **注意**: 2023時点 SOTA。ZiRa(2024)/DUET(2025) に後継されており「現行 SOTA」としては引用しない。
- 関連: [[../../experiments/exp_006.5/related_work_survey]], [[../ZiRaGroundingDINO_summary]]
