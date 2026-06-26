# PCGrad: Gradient Surgery for Multi-Task Learning（要約）

- 出典: Yu, Kumar, Gupta, Levine, Hausman, Finn. *Gradient Surgery for Multi-Task Learning*. NeurIPS 2020.
- arXiv: 2001.06782
- 領域: マルチタスク勾配干渉（領域8精査）

## 中核アイデア
タスク間の**勾配干渉**を3条件で特徴づけ、**PCGrad**で対処:
2タスクの勾配が**コサイン類似度<0（衝突）**のとき、一方の勾配を**他方の法平面に射影**して衝突成分を除去する。
モデル非依存で、教師あり/強化学習で効率を改善。

## 本研究への含意（手法/ベースライン＋分析）
- **【分析設計】**ドメイン間（および各ドメイン↔COCO）の**勾配コサイン/衝突率**を測り、
  「衝突が大きいペアほど忘却が大きい」を定量化（議事録 分析C/D の勾配版、NTK重なりと相補）。
- **【手法/ベースライン】**逐次の代わりに、あるいは LoRA 更新時に**勾配衝突成分を射影除去**する緩和。
  直交化(InfLoRA型)と思想が近く、**比較ベースライン**かつ**統合候補**（CAGrad/GradNorm も同系）。
- caveat: 本来は同時マルチタスク向け。逐次CLでは過去勾配の保持/近似が必要。

## 関連
- 直交部分空間(同族): [[LowRankOrthogonalSubspaces_summary]] / NTK重なり: [[NTK-overlap_summary]]
</content>
