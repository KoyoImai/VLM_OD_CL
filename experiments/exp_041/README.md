# exp_041 実行手順（クラスタ・新規 clone から結果転送まで）

design: [[design]]。実験ノート: [[../../experiment_notes/note11]]。
**実行は design.md の承認後**（行動原理3。2026-08-15 承認済み）。

クラスタ側の作業ディレクトリは**新規 clone の `/home/kouyou/VLM_OD_CL_exp041`**
（design.md §5。2026-08-15 指示）。exp_040 が使う `/home/kouyou/VLM_OD_CL` とは
git pull が干渉しない。

## 0. 状態

| | 状態 |
|---|---|
| config 22本・ドライバ・sbatch 4本・検証 | 生成済み（本環境） |
| 実行前検証（§6.3 の 8+1 項目） | 全 OK（2026-08-15、本環境。§1 参照） |
| クラスタへの clone / 投入 | 未実施 |

## 1. 本環境で検証してから push

**実行前検証は本環境でしか行えない。**クラスタのマスターノードには mmdet の実行環境が無く
（python はコンテナ内にしかない）、計算ノードへは Slurm 経由でしか入れない。
**push する前に本環境で通しておく。**

```bash
cd /workspace/kouyou/mmdetection
python experiments/exp_041/check_exp041_setup.py              # 静的 6 項目
python experiments/exp_041/check_exp041_setup.py --build      # + model build（GPU 1 枚）
python experiments/exp_041/check_exp041_setup.py --step       # + 初期ckpt読込・実データ1 step
python experiments/exp_041/check_exp041_setup.py --evalcheck  # + 260/152 クラス評価の等価確認

git add experiments/exp_041/ && git commit -m "add exp_041"
git push
git log --oneline -1
```

## 2. クラスタで新規 clone

```bash
ssh kouyou@192.168.170.100
git clone https://github.com/KoyoImai/VLM_OD_CL.git /home/kouyou/VLM_OD_CL_exp041
cd /home/kouyou/VLM_OD_CL_exp041
git log --oneline -1                        # §1 の commit と一致すること
ls experiments/exp_041/configs/ | wc -l     # 22 本
```

既に clone 済みの場合は `cd /home/kouyou/VLM_OD_CL_exp041 && git pull`。

## 3. 初期パラメータのコピー（クラスタ内。転送は不要）

初期パラメータは exp_039 の t=3 融合済み ckpt（計約 5.2 GB）。`.pth` は `.gitignore`
なので clone には入らないが、**exp_039 を実行した `/home/kouyou/VLM_OD_CL` に既にある**ため、
クラスタ内でコピーすれば済む（本環境からの rsync は不要）。

```bash
SRC=/home/kouyou/VLM_OD_CL/experiments/exp_039
DST=/home/kouyou/VLM_OD_CL_exp041/experiments/exp_039
for tag in zira_replayfree zira_replay dithub_replayfree dithub_replay; do
  mkdir -p "$DST/${tag}_merged"
  cp -v "$SRC/${tag}_merged/merged_after_t3_videogames.pth" "$DST/${tag}_merged/"
done
ls -la $DST/*_merged/merged_after_t3_videogames.pth   # 4 本あること
```

万一クラスタ側に無い場合のみ、本環境から転送する（本環境で実行）:

```bash
cd /workspace/kouyou/mmdetection
DEST=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp041/experiments/exp_039
for tag in zira_replayfree zira_replay dithub_replayfree dithub_replay; do
  rsync -avh --progress --partial --partial-dir=.rsync-partial \
    "experiments/exp_039/${tag}_merged/merged_after_t3_videogames.pth" \
    "$DEST/${tag}_merged/"
done
```

データ（rf100_domain・o365・MSCOCO）とバッファ定義は exp_039 / exp_040 で使ったものが
そのまま使える（バッファの json は git 管理下なので clone に入っている）。

## 4. 投入（4ジョブ構成）

**4条件は別ジョブなので並列に投入できる**（design.md §5）。ただしクラスタの同時実行は
4ジョブまでなので、exp_040 などが走っている間は空き枠のぶんだけ走る（投入自体は 8 本まで
キューに積める）。`sbatch_*.sh` の `REPO` 既定値が `/home/kouyou/VLM_OD_CL_exp041` なので
**REPO の指定は不要**。

```bash
cd /home/kouyou/VLM_OD_CL_exp041

sbatch experiments/exp_041/sbatch_zira_replayfree.sh     # 約 25 h
sbatch experiments/exp_041/sbatch_zira_replay.sh         # 約 35 h
sbatch experiments/exp_041/sbatch_dithub_replayfree.sh   # 約 25 h
sbatch experiments/exp_041/sbatch_dithub_replay.sh       # 約 35 h

squeue -u kouyou
```

## 5. 進捗の確認

```bash
squeue -u kouyou
tail -f /home/kouyou/logs/result_exp041_dithub_replay_<JOBID>.txt

# どのドメインまで終わったか（融合済み ckpt の有無で判る）
ls /home/kouyou/VLM_OD_CL_exp041/experiments/exp_041/*_merged/ 2>/dev/null

# 投入後 10 分：学習が始まったか
grep -m3 "Epoch(train)" /home/kouyou/logs/result_exp041_<TAG>_<JOBID>.txt
```

DitHub はフェーズ切替（epoch 10）でログに `START SPECIALIZATION` が出る。
aerial は式3 の対象が 1 クラス（`Fish`）なので
`fetch+merge on 1 re-trained classes` になるはず（design.md §2.3）。

```bash
grep "SPECIALIZATION" /home/kouyou/logs/result_exp041_dithub_*_<JOBID>.txt
```

## 6. 結果の転送（クラスタ → 本環境）

**転送は rsync のみで行う。git は使わない**（2026-07-26 の方針）。
**ジョブが終わってから行う**（実行中に rsync すると書き込み途中のファイルを掴む）。

```bash
squeue -u kouyou        # exp041 のジョブが消えていること
```

本環境で実行する。

```bash
cd /workspace/kouyou/mmdetection
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp041/experiments/exp_041
LOCAL=experiments/exp_041
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

### 6.3 融合済み ckpt（再評価・追加分析はこれで足りる）

```bash
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --include='*/' --include='merged_after_t*.pth' --exclude='*' "$REMOTE/" "$LOCAL/"
```

`--partial-dir` は必須。容量は ZiRa 約 0.76 GB ×3 ×2 条件 ＋ DitHub 約 2〜2.5 GB ×3 ×2 条件
≒ **約 18 GB**（DitHub は A 蓄積で t が進むほど肥大する）。

## 7. 完了後

結果は `experiments/exp_041/results/summary.md` に事実のみ記録する（行動原理8）。
前半3ドメイン（exp_039）と接続して 6 ドメイン通しの表にし、exp_040 の 3 条件と
横並びにする（design.md §3）。

## 8. 注意

- 作業ディレクトリは `/home/kouyou/VLM_OD_CL_exp041`。**exp_040 が使う
  `/home/kouyou/VLM_OD_CL` のコードは書き換えない**（初期パラメータの cp は読むだけ）。
- `*_work_dir` 配下は編集しない（禁止則）。
- `/local_cache/${SLURM_JOB_ID}` はジョブ終了で自動削除される。出力をそこに置かない。
- 途中で止まっても同じコマンドで再開できる（融合済み ckpt があるドメインはスキップ。
  ただし学習途中のドメインは最初からやり直しになる）。
