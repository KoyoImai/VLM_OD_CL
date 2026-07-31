# 実験の実行中にクラスタで `git pull` するときの注意

作成: 2026-07-31

**結論: 実行中のジョブがあるうちは pull しない。** どうしても新しいコードを置きたい場合は
別クローンを使う（§3）。

---

## 1. なぜ危険か

### 1.1 実行中の実験のコードが途中で入れ替わる（最も重い）

逐次ドライバ `run_sequential_kd.sh` は**ドメインごとに新しいプロセスを起動する**。

```
underwater 学習（プロセス1） → electromagnetic 学習（プロセス2） → videogames 学習（プロセス3）
                              ↑ ここで pull すると、以降は新しいコードで走る
```

すでに起動しているプロセスは Python のモジュールを import 済みなので影響を受けない。しかし
**次のドメインの学習は pull 後のコードで走る**。結果として「同一条件で3ドメインを逐次学習した」
という前提が崩れる。

評価（`dist_test.sh`）も同様に別プロセスなので、途中で評価コードが変わる可能性がある。

### 1.2 pull が失敗し、その復旧が実行中の出力を壊す

クラスタで生成した出力ファイル（`vis_data/*.json`、`last_checkpoint` など）は未追跡のまま存在する。
一方それらは本環境で rsync 回収してコミットされているため、pull は次で中断する。

```
error: The following untracked working tree files would be overwritten by merge:
```

**pull の中断自体は無害**（何も変更されない）。危険なのは復旧手順のほうで、
[RECOVER_CLUSTER_PULL.md](./RECOVER_CLUSTER_PULL.md) は**未追跡ファイルの削除**を含む。

mmengine は学習中 `vis_data/scalars.json` や `<timestamp>.log` を**開いたまま追記し続ける**。
削除するとプロセスは消えた inode に書き続け、pull で作られる同名ファイルは別の inode になる。
**実行中のジョブの出力が、見えるパスから消える。エラーは出ない。**

`last_checkpoint` を消した場合、ドライバの `cat "$WD/last_checkpoint"` が失敗してジョブが止まる。

---

## 2. 実行前の確認

```bash
ssh kouyou@192.168.170.100 'squeue -u kouyou'
```

**1件でも出たら pull しない。** ジョブ名の対応は次のとおり。

| ジョブ名 | 実験 |
|---|---|
| `exp028{a,b}_kd{AB,DE}_condA_l2` | exp_028 |
| `exp029{a,b}_kd{AB,DE}_condB_l2` | exp_029 |
| `exp030{a,b}_kd{AB,DE}_condA_cos` | exp_030 |
| `exp031{a,b}_kd{AB,DE}_condB_cos` | exp_031 |
| `exp032{a,b}_kd{AB,DE}_condA_l2ratio` | exp_032 |
| `exp033{a..f}_kd{AB,DE}_condA_l2w{20,30,50}` | exp_033 |

### 何が変わるかを事前に見る

pull せずに差分だけ確認する。

```bash
cd /home/kouyou/VLM_OD_CL
git fetch origin
git diff --stat HEAD origin/main -- mmdet experiments/exp_028/run_sequential_kd.sh
```

**`mmdet/` かドライバに差分があれば、実行中の実験に影響する。**
`experiments/exp_0NN/configs/` や新しい実験のディレクトリだけなら影響しない
（ただし pull は全部まとめて入るので、部分的に取り込むことはできない）。

---

## 3. 実行中に新しいコードを置きたい場合：別クローンを使う

実行中のリポジトリに一切触れずに、新しい実験を投入できる。

```bash
cd /home/kouyou
git clone https://github.com/KoyoImai/VLM_OD_CL.git VLM_OD_CL_new
cd VLM_OD_CL_new
git log --oneline -1        # 最新が入っていることを確認
```

投入するときは `sbatch.sh` の `REPO=` を新しいクローンに向ける。

```bash
sed -i 's|^REPO=/home/kouyou/VLM_OD_CL$|REPO=/home/kouyou/VLM_OD_CL_new|' \
  /home/kouyou/VLM_OD_CL_new/experiments/exp_033/sbatch_*.sh
grep -h '^REPO=' /home/kouyou/VLM_OD_CL_new/experiments/exp_033/sbatch_*.sh | sort -u
```

`REPO` はコンテナに `/workspace/kouyou/mmdetection` として bind されるパスなので、
これを変えるだけで新しいクローンのコードが使われる。**元のリポジトリで走っている実験は影響を受けない。**

### 注意

- **成果物の出力先も新しいクローン側**になる（`experiments/exp_033/..._work_dir/`）。回収時は
  そちらから rsync する。
- **バッファ JSON**（`experiments/exp_023/buffer/*.odvg.json`）は git 管理下なのでクローンに含まれる。
- **ディスク**を2倍使うわけではない。クローン直後はコード分だけ（数百MB）。出力が溜まると増える。
- 実験が全部終わったら、クローンを消すか、元のリポジトリに統合するかを決める。

---

## 4. 実行中のジョブが無い場合の手順

1. `squeue -u kouyou` が空であることを確認
2. `git pull`
3. 中断したら [RECOVER_CLUSTER_PULL.md](./RECOVER_CLUSTER_PULL.md) に従う
   （衝突ファイルの列挙 → 内容比較 → バックアップ → 削除 → pull）

---

## 5. 根本的な対処（未決）

この問題は、**実験の出力をリポジトリにコミットしている**限り繰り返す。

`.gitignore` に次を足せば pull の衝突は起きなくなる（§1.2 の危険が消える）。

```
experiments/*/*_work_dir/
experiments/*/*eval*/
```

ただし **§1.1 の「実行中にコードが入れ替わる」問題は残る**。こちらは pull のタイミングを
管理するか、別クローンを使うしかない。

既にコミット済みの exp_020〜029 の出力をどう扱うか（`git rm --cached` で外すか）を含め、
判断が必要。
