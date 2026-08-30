# exp_054 実行手順（seed 頑健性: 6 手法 × seed 1,2・クラスタ 12 ジョブ）

design: [[design]]。**実行は design.md の承認後（2026-08-31 承認済み）。**
push・clone・sbatch はユーザー実施。

条件: note15 の 6 手法（Ours=蒸留E λ=10 旧バッチ / ER=リプレイ / EWC λ=1000 /
InfLoRA / ZiRa・DitHub replay free）を **seed=1, 2** で 6 ドメイン通し再実行。
config は既存実験のものをそのまま使い、上書きは `randomness.seed` と
ckpt 保存方針（epoch_20 のみ・optimizer 無し）だけ。補助スクリプト
（EWC の Fisher 推定、InfLoRA の prepare/update）の --seed も同じ値にする。
バッファ json は既存を共用（リプレイ画像の選択は固定）。

## 0. 状態

| | 状態 |
|---|---|
| ドライバ 5 本・sbatch 12 本・検証 | 生成済み |
| 実行前検証（check_exp054_setup.py --step） | **4/4 OK**（2026-08-31。config 36 本 build・seed/ckpt 上書き到達・補助スクリプト --seed・Ours/EWC の seed=1 1 step 有限） |
| クラスタ投入 | 未実施 |

## 1. 本環境で検証（実施済み・再現手順）

```bash
python experiments/exp_054/check_exp054_setup.py
CUDA_VISIBLE_DEVICES=0 python experiments/exp_054/check_exp054_setup.py --step
```

## 2. クラスタ（ユーザー実施）

```bash
# 0) 前提確認（EWC videogames の実効構成。design.md §4.1）
#    exp_045 の t=3 を最終的に通した構成（encoder num_cp / partition）を確認する:
cat /home/kouyou/VLM_OD_CL_exp045/experiments/exp_045/ewc_videogames_work_dir/*/vis_data/config.py | grep -m2 num_cp
#    → 2026-08-31 確認済み: 実効 config は num_cp=0。48GB では OOM する構成のため、
#      exp_054 の EWC 2 本は sbatch を a100 パーティションに変更済み。
#      （当時の EWC ジョブのノード名が node21/22 だったかを sacct 等で確認できれば確実）

# 本環境: commit & push
git add experiments/exp_054/
git commit -m "add exp_054 (seed robustness, 6 methods x seed 1,2)"
git push

# クラスタ: 専用クローン
git clone https://github.com/KoyoImai/VLM_OD_CL.git /home/kouyou/VLM_OD_CL_exp054
cd /home/kouyou/VLM_OD_CL_exp054
git log --oneline -1
ls experiments/exp_054/sbatch_*.sh | wc -l   # 12

# 投入（同時実行 4 本制限。残りはキュー待ち 8 本の枠内で順次）
for f in experiments/exp_054/sbatch_*_s1.sh; do sbatch $f; done   # まず seed=1 の 6 本
for f in experiments/exp_054/sbatch_*_s2.sh; do sbatch $f; done   # 続けて seed=2 の 6 本
squeue -u kouyou
```

- キュー上限（実行 4 + 待機 8）を超える場合は seed=1 の 6 本 → 空き次第 seed=2 を投入。
- clone の認証エラーは exp_052 README §2.1 の対処（VS Code ターミナル or unset + PAT）。

### 投入後 10 分の確認

```bash
LOG=/home/kouyou/logs/result_exp054_ours_s1_<JOBID>.txt   # 他ジョブも同様
grep -m2 "seed" $LOG          # randomness.seed=1 が渡っていること
grep -m3 "Epoch(train)" $LOG
```

## 3. 再開

同じ sbatch を再投入（各ドライバは成果物の有無でスキップ:
Ours/ER/EWC=epoch_20 or state、InfLoRA=theta、ZiRa/DitHub=merged。
評価はディレクトリ有無。**json の無い評価ディレクトリは消してから**）。

## 4. 結果の転送（クラスタ → 本環境）

```bash
cd /workspace/kouyou/mmdetection
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp054/experiments/exp_054
LOCAL=experiments/exp_054
rsync -avh --progress \
  --include='*/' --include='eval_*/***' --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress \
  --include='*/' --include='*.log' --include='scalars.json' --include='config.py' \
  --exclude='*' "$REMOTE/" "$LOCAL/"
# ckpt（必要なときのみ。約 50 GB）
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --include='*/' --include='epoch_20.pth' --include='merged_after_t*.pth' \
  --include='theta_t*.pth' --exclude='*' "$REMOTE/" "$LOCAL/"
```
