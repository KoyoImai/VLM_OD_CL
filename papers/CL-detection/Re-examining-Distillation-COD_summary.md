# 蒸留の再検討 要約（Re-examining Distillation for Continual Object Detection, BMVC 2022）

arXiv:2204.01407 (Verwimp et al.)。KD ベース継続検出のベースラインと機序知見。

## 中核
- 二段検出器で **RPN への蒸留は有効**だが、**分類ヘッドの蒸留は失敗**する
  （過信した誤り教師予測が分類ヘッド学習を妨げる）。
- 対策: 現在の ground-truth で誤教師予測を検出し、MSE の代わりに **適応的 Huber 損失**を分類ヘッド蒸留に使用。
- クラス増分だけでなく**現実的なドメイン増分設定でも有効**。

## 本研究との関係
- 「**検出の忘却は分類側に集中**」という機序知見を補強（[[Demystifying-TwoStage-IOD_summary]] と整合）。
- KD ベースの比較ベースラインとして参照可能。
- ドメイン増分でも蒸留改良が効く点は、本研究の評価設定（ドメイン増加）と親和的。
- 関連: [[../../experiments/exp_006.5/related_work_survey]]
