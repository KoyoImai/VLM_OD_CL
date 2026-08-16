# 物体検出における EWC の適用例（調査メモ、2026-08-15）

exp_042（EWC を MM-Grounding DINO に実装）の前提調査。ユーザーの依頼による Web 調査
（Semantic Scholar / arXiv / WebSearch）。読んだのは抄録・検索要約・TiROD は本文 PDF。

## 見つかった適用例

### 1. IncDet（IEEE TNNLS 2020）— 中心的な論文
"IncDet: In Defense of Elastic Weight Consolidation for Incremental Object Detection"
(Liu et al., https://ieeexplore.ieee.org/document/9127478)

- クラス増分の物体検出に EWC を**素朴に適用すると破滅的忘却が残る**とし、原因を 2 つ特定:
  1. 新クラス画像に旧クラスの物体がアノテーション無しで写り込み、旧クラスが背景として
     抑制される（クラス増分特有の欠落アノテーション問題）
  2. **二次正則化損失が勾配爆発を起こしやすい**（Fisher の大きい要素と λ の積が大きくなる）
- 対処: 擬似バウンディングボックス（旧モデルの予測でアノテーション補完）＋
  **二次損失を Huber 型に置き換えて勾配をクリップ**（IncDet）
- 注: 使用検出器は抄録からは特定できず（本文未読）。

### 2. TiROD ベンチマーク（2024）— ドメイン増分検出での実測
arXiv:2409.16215（本文 PDF を確認）。本研究に最も近いドメイン増分検出ベンチマーク。

- NanoDet / FCOS に IncDet（EWC 損失係数 5000）を適用し、LwF・SID・リプレイ系と比較。
- Cross-Domain シナリオの Final mAP（3 seed 平均）:
  Fine-Tuning 13.4±1.34 / LwF 16.2±2.50 / **IncDet 12.9±0.60** / SID 20.8±2.42 /
  Replay 38.4±0.74 / K-Means Replay 42.7±0.20
- IncDet は安定性指標（RSD 0.32）は正則化系で最良だが、Final mAP は素の Fine-Tuning を
  下回った。**ドメインシフト下では正則化系全般がリプレイ系に大差で劣る**というのが
  ベンチマークの結論（"regularization methods generally underperformed compared to
  replay-based techniques"）。

### 3. YOLO v5 + EWC（AIPR 2023, ACM）
https://dl.acm.org/doi/10.1145/3641584.3641651。YOLOv5 に EWC を組み込んだ増分検出。
本文は取得不可（403）で詳細未確認。

## 見つからなかったもの

- **VLM / open-vocabulary 検出器（Grounding DINO 系）への EWC 適用例は見つからず**。
  ZiRa 論文（全文 grep）・手元の検出 CL 要約（CL-DETR・TiROD 等）にも EWC 比較なし。
  OVD の継続学習（COVD、CLIFF 等）も蒸留・リプレイ・プロンプト系が中心。
- → exp_042 の「MM-Grounding DINO ＋ EWC」に直接の前例は確認できていない。

## exp_042 への含意

1. **適用可能性は前例で裏付けられる**（古典検出器では確立したベースライン）。
2. IncDet の問題 1（欠落アノテーション）は**クラス増分特有**で、ODinW-13 のタスク別
   プロンプト・タスク別データの設定にはほぼ該当しない。
3. IncDet の問題 2（**二次ペナルティの勾配爆発**）は本実験にも直接関係する。共通枠の
   grad clip 0.1（L2）が緩衝になるが、λ パイロットで大 λ の学習が発散しないかを見る。
   TiROD の係数 5000 は λ グリッド {0, 10², 10³, 10⁴} の妥当性の参考になる。
4. TiROD の実測（EWC 系はドメインシフト下でリプレイに大差で劣る）は、本プロジェクトの
   RF100 でのリプレイの強さとも整合する。exp_042 の EWC が弱くても、それは文献と整合する
   結果として読める。
