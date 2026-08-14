# exp_040 実行手順（クラスタ・投入から結果転送まで）

design: [[design]]。実験ノート: [[../../experiment_notes/note11]]。
**実行は design.md の承認後**（行動原理3。2026-08-14 承認済み）。

クラスタ側の作業ディレクトリは `/home/kouyou/VLM_OD_CL`。

## 0. 状態

| | 状態 |
|---|---|
| config 9本・ドライバ・sbatch・検証 | 生成済み（本環境） |
| 実行前検証 9 項目 | 9/9 OK（2026-08-14、本環境） |
| クラスタへの push / 投入 | 未実施 |

## 1. 本環境で検証してから push

**実行前検証は本環境でしか行えない。**クラスタのマスターノードには mmdet の実行環境が無く
（python はコンテナ内にしかない）、計算ノードへは Slurm 経由でしか入れない。
**push する前に本環境で通しておく。**

```bash
cd /workspace/kouyou/mmdetection
python experiments/exp_040/check_exp040_setup.py           # 8/8 OK
python experiments/exp_040/check_exp040_setup.py --build   # 9/9 OK（GPU 1 枚）

git add experiments/exp_040/ && git commit -m "add exp_040"
git push
git log --oneline -1
```

## 2. クラスタで pull

作業ディレクトリは `/home/kouyou/VLM_OD_CL`（2026-08-14 確定）。

```bash
ssh kouyou@192.168.170.100
cd /home/kouyou/VLM_OD_CL
git status --short                   # 変更が無いこと（あれば先に確認）
git pull && git log --oneline -1     # §1 の commit と一致すること
ls experiments/exp_040/configs/ | wc -l    # 9 本
```

**pull の前に `squeue -u kouyou` で、同じクローンを使う実行中ジョブが無いかを確認する。**
実行中のジョブが参照するコードを書き換えると、以降に起動する学習・評価が新しいコードで走る。

## 3. 依存する資産の確認

**初期パラメータは前半3実験の成果物**なので、クラスタ側に無ければ転送が要る。

```bash
for p in exp_024/fullft_replayfree_videogames_work_dir \
         exp_027/fullft_replay_videogames_work_dir \
         exp_035/kdE_condA_l2w100_videogames_work_dir; do
  ls -la /home/kouyou/VLM_OD_CL/experiments/$p/epoch_20.pth 2>/dev/null \
    || echo "MISSING $p/epoch_20.pth"
done
```

`.pth` は `.gitignore` なので clone / pull では入らない。**MISSING が出たら本環境から
転送する**（下記は本環境で実行）。

```bash
cd /workspace/kouyou/mmdetection
DEST=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
for p in exp_024/fullft_replayfree_videogames_work_dir \
         exp_027/fullft_replay_videogames_work_dir \
         exp_035/kdE_condA_l2w100_videogames_work_dir; do
  rsync -avh --progress --partial --partial-dir=.rsync-partial \
    "experiments/$p/epoch_20.pth" "$DEST/$p/"
done
```

1 本 約 1.95 GB × 3 = 約 6 GB。データ（rf100_domain・o365・MSCOCO）とバッファ定義は
exp_027 / exp_028 で使ったものがそのまま使える（新規転送なし）。

## 4. 投入

**3条件を並列に投入できる**（design.md §5.2）。ただしクラスタの同時実行は4ジョブまでなので、
exp_039 が2ジョブ使っている場合は 2 本までに留めるか、空くのを待つ。

`sbatch_*.sh` の `REPO` の既定値が `/home/kouyou/VLM_OD_CL` なので、**REPO の指定は不要**。

```bash
cd /home/kouyou/VLM_OD_CL

sbatch experiments/exp_040/sbatch_replayfree.sh   # 約 19 h
sbatch experiments/exp_040/sbatch_replay.sh       # 約 30 h
sbatch experiments/exp_040/sbatch_kdE.sh          # 約 36 h

squeue -u kouyou
```

別のクローンから走らせたい場合のみ `REPO=<パス> sbatch ...` とする。

## 5. 進捗の確認

```bash
squeue -u kouyou
tail -f /home/kouyou/logs/result_exp040_kdE_<JOBID>.txt

# どのドメインまで終わったか
ls -d /home/kouyou/VLM_OD_CL/experiments/exp_040/*_work_dir/

# 投入後 10 分：学習が始まったか
grep -m3 "Epoch(train)" /home/kouyou/logs/result_exp040_<COND>_<JOBID>.txt
```

蒸留（kdE）は損失に `loss_kd` が出る。λ=10 なので前半 exp_035 と同水準の値になるはず。

```bash
grep -o "loss_kd: [0-9.]*" /home/kouyou/logs/result_exp040_kdE_<JOBID>.txt | head -3
```

## 6. 結果の転送（クラスタ → 本環境）

**転送は rsync のみで行う。git は使わない**（2026-07-26 の方針）。
**ジョブが終わってから行う**（実行中に rsync すると書き込み途中のファイルを掴む）。

```bash
squeue -u kouyou        # exp040 のジョブが消えていること
```

本環境で実行する。

```bash
cd /workspace/kouyou/mmdetection
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments/exp_040
LOCAL=experiments/exp_040
```

### 6.1 評価結果（最優先・軽い）

```bash
rsync -avh --progress \
  --include='*/' --include='*eval*/***' --exclude='*' "$REMOTE/" "$LOCAL/"
```

### 6.2 学習ログ（ckpt は除く）

```bash
rsync -avh --progress \
  --include='*/' --include='*.log' --include='scalars.json' --include='config.py' \
  --exclude='*' "$REMOTE/" "$LOCAL/"
```

### 6.3 チェックポイント（重い。必要なものだけ）

各ドメインの最終 `epoch_20.pth` だけで再評価も追加分析もできる。

```bash
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --include='*/' --include='epoch_20.pth' --exclude='*' "$REMOTE/" "$LOCAL/"
```

`--partial-dir` は必須。容量は 3 条件 × 3 ドメイン × 約 1.95 GB ≒ **18 GB**。

## 7. 完了後

結果は `experiments/exp_040/results/summary.md` に事実のみ記録する（行動原理8）。
前半3ドメインの推移（exp_024 / exp_027 / exp_035 の結果記録）と接続して、
6ドメイン通しての表にする（design.md §3 の判定材料4）。

## 8. 注意

- 作業ディレクトリは `/home/kouyou/VLM_OD_CL`。**pull する前に `squeue` で、同じクローンを
  使う実行中ジョブが無いかを確認する**（§2）。
- `*_work_dir` 配下は編集しない（禁止則）。
- `/local_cache/${SLURM_JOB_ID}` はジョブ終了で自動削除される。出力をそこに置かない。
- 途中で止まっても同じコマンドで再開できる（`epoch_20.pth` があるドメインはスキップ）。
