# exp_059 実行手順（蒸留係数 λ 感度分析・クラスタ 3 ジョブ）

design: [[design]]。**2026-09-01 承認済み。**push・clone・sbatch はユーザー実施。

条件: Ours（蒸留E・旧バッチ [4,1,1]×6・exp_035/040 kdE 継承）で
**λ = 5（w50）/ 15（w150）/ 20（w200）**。λ=10 は exp_035/040 の実測を再利用。
6 ドメイン通し・seed 0・ckpt は epoch_20 のみ。
**λ は config に焼き込み済みで、ドライバ・sbatch は loss_weight を渡さない**
（exp_033 の λ 不達事故の再発防止。design §1.1）。

## 0. 状態

| | 状態 |
|---|---|
| config 18 本・ドライバ・sbatch 3 本・検証 | 生成済み |
| 実行前検証（check_exp059_setup.py --step） | **2/2 OK**（2026-09-01。全キー diff = loss_weight＋ckpt 方針のみ・loss_kd の λ 比例を実測: w200/w50 = 4.000000） |
| クラスタ投入 | 未実施 |

## 1. 本環境で検証（実施済み・再現手順）

```bash
python experiments/exp_059/check_exp059_setup.py
CUDA_VISIBLE_DEVICES=0 python experiments/exp_059/check_exp059_setup.py --step
```

## 2. クラスタ（ユーザー実施）

```bash
# 本環境: commit & push
git add experiments/exp_059/
git commit -m "add exp_059 (KD loss-weight sensitivity, lambda 5/15/20)"
git push

# クラスタ: 専用クローン
git clone https://github.com/KoyoImai/VLM_OD_CL.git /home/kouyou/VLM_OD_CL_exp059
cd /home/kouyou/VLM_OD_CL_exp059
git log --oneline -1
ls experiments/exp_059/configs/ | wc -l   # 18
sbatch experiments/exp_059/sbatch_w50.sh   # 各 約 90〜120 h（--time=144:00:00）
sbatch experiments/exp_059/sbatch_w150.sh
sbatch experiments/exp_059/sbatch_w200.sh
squeue -u kouyou
```

### 投入後 10 分の確認（λ 不達の最終防衛線）

```bash
LOG=/home/kouyou/logs/result_exp059_w50_<JOBID>.txt
grep -m2 "loss_kd" $LOG
# 実効 config の確認（t=1 の work_dir が出来てから）
grep -m1 "loss_weight" /home/kouyou/VLM_OD_CL_exp059/experiments/exp_059/w50_underwater_work_dir/*/vis_data/config.py
# 期待値: w50=5.0 / w150=15.0 / w200=20.0
```

## 3. 再開・回収

再開は同じ sbatch 再投入（epoch_20 / 評価ディレクトリ有無でスキップ。json の無い
評価ディレクトリは消してから）。回収は exp_051 README §4 と同形式（REMOTE を
VLM_OD_CL_exp059 に読み替え）。集計は note15 形式で、λ=10（exp_035/040）と並べる。
