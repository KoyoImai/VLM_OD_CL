# 別クローンで exp_030〜033 を実行する手順

作成: 2026-07-31

実行中の exp_029 に触れずに新しい実験を投入するため、**クラスタに2つ目のクローンを作る**。
背景と危険性は [PULL_WHILE_RUNNING.md](./PULL_WHILE_RUNNING.md)。

**元の `/home/kouyou/VLM_OD_CL` は一切変更しない。** そこで走っている exp_029 は影響を受けない。

---

## 0. 前提の確認

本環境側は push 済み（2026-07-31 時点で未 push のコミットは 0）。新クローンには次が含まれる。

| 実験 | config | スクリプト | design |
|---|---:|---:|---|
| exp_030（条件A・コサイン） | 12本 | 4本 | あり |
| exp_031（条件B・コサイン） | 12本 | 4本 | あり |
| exp_032（条件A・比率追従） | 12本 | 4本 | あり |
| exp_033（条件A・λ 掃引） | 36本 | 12本 | あり |

**新クローンは exp_027 の学習出力に依存しない。** `run_sequential_kd.sh` が exp_027 の ckpt を
読むのは `CALIBRATE=1` のときだけで、既定では使わない。θ0・データセット・`.sif`・ログ出力先は
すべてクローンの外（`/home/kouyou/{ckpt,datasets,sif,logs}`）にあるので共用される。

---

## 1. クローンを作る

```bash
cd /home/kouyou
git clone https://github.com/KoyoImai/VLM_OD_CL.git VLM_OD_CL_new
cd VLM_OD_CL_new
git log --oneline -3
```

最新コミットが `5aefa4b add exp_033` 以降であることを確認する。

### 中身の確認

```bash
cd /home/kouyou/VLM_OD_CL_new
ls mmdet/models/detectors/kd_grounding_dino.py mmdet/models/losses/feature_distill_loss.py
ls experiments/exp_023/buffer/*.odvg.json | wc -l          # 14 のはず（7ドメイン × 2）
for e in exp_030 exp_031 exp_032 exp_033; do
  echo -n "$e configs: "; ls experiments/$e/configs | wc -l
done
```

---

## 2. `REPO` を新クローンに向ける

`sbatch.sh` の `REPO=` が、コンテナに `/workspace/kouyou/mmdetection` として bind される
パスを決めている。**ここ1箇所を変えるだけ**で新クローンのコードが使われる。

```bash
cd /home/kouyou/VLM_OD_CL_new
sed -i 's|^REPO=/home/kouyou/VLM_OD_CL$|REPO=/home/kouyou/VLM_OD_CL_new|' \
  experiments/exp_03{0,1,2,3}/sbatch_*.sh

# 確認（すべて VLM_OD_CL_new になっていること）
grep -h '^REPO=' experiments/exp_03{0,1,2,3}/sbatch_*.sh | sort | uniq -c
```

**この変更は新クローン側だけで行う。** 元のリポジトリの `sbatch.sh` は触らない。

---

## 3. 投入

すべて `/home/kouyou/VLM_OD_CL_new` から実行する。

```bash
cd /home/kouyou/VLM_OD_CL_new
```

### exp_033（λ 掃引・12条件・6ジョブ）

**同時実行の上限は4**なので2回に分ける。

```bash
# 第1波
sbatch experiments/exp_033/sbatch_a.sh    # λ=2.0  蒸留A・B
sbatch experiments/exp_033/sbatch_b.sh    # λ=2.0  蒸留D・E
sbatch experiments/exp_033/sbatch_c.sh    # λ=3.0  蒸留A・B
sbatch experiments/exp_033/sbatch_d.sh    # λ=3.0  蒸留D・E

# 第1波が終わってから
sbatch experiments/exp_033/sbatch_e.sh    # λ=5.0  蒸留A・B
sbatch experiments/exp_033/sbatch_f.sh    # λ=5.0  蒸留D・E
```

**exp_029 がまだ枠を使っている場合、投入可能な数はその分減る。** `squeue -u kouyou` で
空き枠を確認してから投入する（同時実行4・投入待ち込みで8まで）。

### exp_030 / exp_031（コサイン・各4条件・2ジョブ）

```bash
sbatch experiments/exp_030/sbatch_a.sh
sbatch experiments/exp_030/sbatch_b.sh
sbatch experiments/exp_031/sbatch_a.sh
sbatch experiments/exp_031/sbatch_b.sh
```

### exp_032（比率追従）

**design が未確定**（`max_weight` の扱い、ρ の値、note06 では ρ を3点振る設計）。
現状の config は ρ=0.1 の4条件分しかない。投入は設計確定後。

---

## 4. 成果物の場所が変わる

**出力は新クローン側に作られる。**

```
/home/kouyou/VLM_OD_CL_new/experiments/exp_033/kd{A,B,D,E}_condA_l2w{20,30,50}_{dom}_work_dir/
/home/kouyou/VLM_OD_CL_new/experiments/exp_033/kd{...}_eval_after_{dom}/{on_{evd},zcoco}/
```

回収するときの rsync も**元のリポジトリではなく新クローンを指す**。

```bash
# 本環境で実行
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_new/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments
for e in exp_030 exp_031 exp_033; do
  rsync -avh --progress --partial --partial-dir=.rsync-partial \
    --exclude='/configs/' --exclude='/*.sh' --exclude='/*.py' --exclude='/*.md' \
    "$REMOTE/$e/" "$LOCAL/$e/"
done
```

ジョブログは共用（`/home/kouyou/logs/`）なので変わらない。

---

## 5. 実験が全部終わったあと

新クローンをどうするか決める必要がある。選択肢は2つ。

**案1: 新クローンを消す（推奨）**
成果物を本環境へ rsync で回収し、照合（ckpt 個数・評価件数）が済んでから削除する。
コードは git で管理されているので失われるものはない。

```bash
# 照合してから
rm -rf /home/kouyou/VLM_OD_CL_new
```

**案2: 元のリポジトリに戻す**
元の `/home/kouyou/VLM_OD_CL` で `git pull` して新クローンを消す。ただし
[PULL_WHILE_RUNNING.md](./PULL_WHILE_RUNNING.md) の未追跡ファイル衝突が起きるので、
[RECOVER_CLUSTER_PULL.md](./RECOVER_CLUSTER_PULL.md) の手順が必要になる。

---

## 6. 注意点

- **ディスク**: クローン直後はコード分だけ（数百MB）。出力が溜まると exp_033 で約 1.4TB。
- **`--exclude=node03`** は各 `sbatch_*.sh` に組み込み済み（`/local_cache` が用意されないため）。
- **PORT の衝突なし**: exp_030 が 29700/29710、exp_031 が 29720/29730、exp_032 が 29740/29750、
  exp_033 が 29760〜29810。実行中の exp_029（29680/29690）とも重ならない。
- **元のリポジトリで走っている exp_029 の出力**は元の場所に作られ続ける。回収時は
  [RETRIEVE_EXP028_029.md](./experiments/RETRIEVE_EXP028_029.md) のとおり
  `/home/kouyou/VLM_OD_CL/experiments/` から取る。**2つの場所から回収することになる**点に注意。
