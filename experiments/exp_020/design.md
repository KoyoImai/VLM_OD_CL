# exp_020 設計書: ZiRa 再現（単発ドメイン学習・全 6 ドメイン）

作成 2026-07-10。実行前のユーザー承認ゲート。
実装の詳細と検証結果は [[implementation_plan]]、手法の正本は
[[../../papers/ZiRa_implementation_notes]]。

## 1. 目的

既存手法 ZiRa（NeurIPS 2024、リプレイ・蒸留なしで VLODM の逐次適応と
ゼロショット保持を両立すると主張する代表手法）を MM-Grounding DINO 上で
忠実に再現し、本プロジェクトの統一プロトコル（単発ドメイン学習 + ZCOCO 評価、
seed=0）で測定する。得られる行は既存基準線（frozen / unfrozen / neckTfm /
bert_tfm / extfusion / downstream）と同じ表に並び、
(a) 既存手法の適応能力が我々のドメイン群でどこに位置するか、
(b) ZiL による忘却抑制がどの程度効くか、を実測で確定させる。

## 2. 条件

1 条件 × 6 ドメイン（underwater / aerial / videogames / microscopic /
documents / electromagnetic。real_world は除外〔ユーザー指定 2026-07-10〕）。

- モデル: ZiRaGroundingDINO（本体全凍結、RDB 25 テンソルのみ学習。
  text_feat_map と neck 射影 conv 4 本に並列挿入、ZiL λ=0.1、η=0.2、s=0.1）
- 学習: exp_xxx 統一（AdamW lr 1e-4 / wd 1e-4、20 epoch、milestones[15]、
  4 GPU dist_train、randomness=dict(seed=0) 明示、clip_grad 0.1）
- config: `experiments/exp_020/configs/zira_{domain}.py`
- 実行: `bash experiments/exp_020/run_train.sh`（ドメイン順は上記の並び。
  各ドメイン: 学習 → best ckpt を `zira_eval_coco.py` で ZCOCO 評価)

## 3. 測定と比較基準線

各ドメインで適応 mAP（自ドメイン val、best epoch）と ZCOCO mAP（best ckpt）。
比較先は module_rollback_summary.md の frozen / unfrozen / neckTfm 行
（全て seed=0・同一プロトコル）。θ0 ZCOCO = 0.504。

解釈の観点（事前宣言）:
- 適応: ZiRa の挿入位置は exp_017 neckTfm と同じ箇所（学習形態は直接更新 vs
  並列枝で異なる）。適応が neckTfm 水準（0.11〜0.22）に留まるか、それを
  超えるか。unfrozen / extfusion 水準に届くとは予想していないが、
  結果を見てから判定は変えない（実測値をそのまま記録する）。
- 忘却: ZCOCO が neckTfm（0.447〜0.465）や θ0 にどこまで近いか。
  ZiL の出力ノルム抑制が我々の lr / スケジュールでも機能するかを
  loss_zil の学習曲線と合わせて確認する。

## 4. 実装検証（実行済み、7/7 OK）

check_zira_setup.py で確認済み: ①学習対象= RDB 25 テンソルのみ・本体 896
凍結、①b ckpt ロード後も RDB 初期状態維持、②θ0 等価性（bbox 差 5.8e-4）、
③勾配疎通、④loss_zil 初期値 ≈0、⑤融合等価性（相対差 ≤3.1e-5）、
⑥融合後状態。η は optimizer グループで LLRB=2e-5 / 他=1e-4 を確認。

## 5. 既知の注意点

- dn_query_generator.label_embedding は ckpt と形が合わず乱数初期化のまま
  凍結される。これは本体を凍結する既存条件（exp_017/018/019 の凍結側）と
  同一の状況であり、比較条件は揃っている。
- λ=0.1 は公式の lr 1e-3 に対する値。我々は lr 1e-4 のため損失バランスが
  同じに働く保証はなく、loss_zil と適応 mAP の立ち上がりを学習曲線で監視する。
  適応が全く立ち上がらない場合は中断して相談する（勝手に λ は変えない）。
- 単発学習のためタスク間融合（Rep+）は発生しない。評価は全和 forward で
  融合後と数学的に等価（implementation_plan.md 3.1/3.5）。

## 6. コスト

6 run × 半日前後 + ZCOCO 各 30 分 ≈ 3〜4 日（4 GPU 占有、直列実行）。

## 7. 成果物

- `experiments/exp_020/{domain}_zira_work_dir`（ckpt・ログ・scalars.json）
- `experiments/exp_020/{domain}_zira_zcoco`（ZCOCO 評価）
- `results/zira_reproduction.md`: 6 ドメインの適応・ZCOCO 実測値と
  基準線比較表（学習完了後に作成）
- module_rollback_summary.md の横断表への ZiRa 行追記

## 関連
- 実装計画・新規ファイル台帳: [[implementation_plan]]
- 基準線: [[../module_rollback_summary]]

## 追記（2026-07-11、ユーザー承認済み）: 公式ハイパラ対照 run

20 epoch 版で ZCOCO 低下（underwater −6.8 pt / aerial −3.1 pt）が論文の水準
（13 タスク逐次で −1.3 pt）を大きく超えたため、原因が実装か学習量の体制かを
切り分ける対照 run を実施する。診断済みの事実: 凍結部 907 テンソルは θ0 と
ビット一致（凍結の破れなし）、loss_zil は動的平衡（ピーク 0.81 → 0.73）、
RDB 実効ノルムは本体の 21〜33% まで成長（s は 0.1 → 0.54〜0.80）。

- config: `configs/zira_official_underwater.py`（2000 iter / batch 2 / 1 GPU /
  lr 1e-3 / iter 800 decay / η=0.2 / λ=0.1 / dn 有効 / seed=0）
- 対象: underwater のみ
- 判定: ZCOCO 低下が ~1 pt 級に収まれば実装は忠実（差は体制）、
  収まらなければ実装を再監査する
