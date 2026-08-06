# exp_036 実行手順（λ = 2.0 / 3.0 / 5.0。exp_033 の再実行）

design: [[design]]。共通仕様は [[../exp_028/design]]、掃引の設計は [[../exp_033/design]]。

## exp_033 との違い（ここだけ）

exp_033 は **全12条件が実効 λ=1.0 で学習されていた**。ドライバ
`experiments/exp_028/run_sequential_kd.sh` が `lam="${LAMBDA:-1.0}"` を
`--cfg-options model.kd.loss_weight` として渡し、config の値を上書きするためである
（2026-08-06 検出。詳細は design §0）。

exp_036 では `train_val_{a..f}.sh` で **`export LAMBDA=<λ>`** を明示し、さらに起動時に
config の `loss_weight` と `LAMBDA` の一致を確認するガードを入れている。
**ジョブログの冒頭に `[exp_036_x] 実効 λ = <値>（config と一致）` が出ることを必ず確認する。**

**exp_033 はそのまま残す**（同一条件の反復データとして再現性の推定に使う）。

## 構成（12条件・6ジョブ）

| ジョブ | λ | 蒸留対象 | ジョブ名 | PORT |
|---|---:|---|---|---:|
| `exp_036_a` | 2.0 | A・B | `exp036a_kdAB_condA_l2w20` | 29840 |
| `exp_036_b` | 2.0 | D・E | `exp036b_kdDE_condA_l2w20` | 29850 |
| `exp_036_c` | 3.0 | A・B | `exp036c_kdAB_condA_l2w30` | 29860 |
| `exp_036_d` | 3.0 | D・E | `exp036d_kdDE_condA_l2w30` | 29870 |
| `exp_036_e` | 5.0 | A・B | `exp036e_kdAB_condA_l2w50` | 29880 |
| `exp_036_f` | 5.0 | D・E | `exp036f_kdDE_condA_l2w50` | 29890 |

各ジョブは2条件を直列に実行する（1条件 ≒ 32 時間、1ジョブ ≒ 64 時間）。

## 1. リポジトリをクラスタへ反映

本環境で:

```bash
git add experiments/exp_036
git commit -m "add exp_036 (lambda sweep, fixed)"
git push
```

クラスタのマスターノードで:

```bash
cd /home/kouyou/VLM_OD_CL
git pull
```

## 2. 投入

同時実行の上限は4なので2回に分ける。

```bash
sbatch experiments/exp_036/sbatch_a.sh
sbatch experiments/exp_036/sbatch_b.sh
sbatch experiments/exp_036/sbatch_c.sh
sbatch experiments/exp_036/sbatch_d.sh
```

上の4本が終わってから:

```bash
sbatch experiments/exp_036/sbatch_e.sh
sbatch experiments/exp_036/sbatch_f.sh
```

蒸留対象を1つずつ投入する場合:

```bash
RUN_STAGE=A sbatch experiments/exp_036/sbatch_a.sh
RUN_STAGE=B sbatch experiments/exp_036/sbatch_a.sh
RUN_STAGE=D sbatch experiments/exp_036/sbatch_b.sh
RUN_STAGE=E sbatch experiments/exp_036/sbatch_b.sh
```

## 3. 投入直後に確認すること

```bash
squeue -u kouyou
grep "実効 λ" /home/kouyou/logs/result_exp036a_kdAB_condA_l2w20_<JOBID>.txt
```

`[exp_036_a] 実効 λ = 2.0（config と一致）` が出ていれば λ が正しく効いている。
出ていない場合は学習を止めて確認する。

## 4. 結果の取得（本環境へ）

```bash
rsync -avz --progress \
  kouyou@<クラスタのホスト>:/home/kouyou/VLM_OD_CL/experiments/exp_036/ \
  /workspace/kouyou/mmdetection/experiments/exp_036/
```

## 投入前の注意

- exp_035（λ=10.0、2ジョブ）と同時に走らせるとジョブ枠を圧迫する。投入順はユーザーが決める。
- ディスクは 12条件 × 3ドメイン × 20 epoch × 2.0GB ＝ **約 1.4 TB**。
- `--exclude=node03`（`/local_cache` が用意されないため。2026-07-25）。

## 実行前検証の結果（2026-08-06、本環境で実施）

1. **config の差分**: 36本すべて、exp_028 の対応 config との差は `model.kd.loss_weight` の1キーのみ
   （1.0 → 2.0 / 3.0 / 5.0）。**OK**
2. **実効 λ の一致**: 6本の `train_val_*.sh` のガードを本環境で実行し、すべて通過。**OK**

   ```
   [exp_036_a] 実効 λ = 2.0（config と一致）
   [exp_036_b] 実効 λ = 2.0（config と一致）
   [exp_036_c] 実効 λ = 3.0（config と一致）
   [exp_036_d] 実効 λ = 3.0（config と一致）
   [exp_036_e] 実効 λ = 5.0（config と一致）
   [exp_036_f] 実効 λ = 5.0（config と一致）
   ```
3. **損失が線形にスケールすること**: exp_035 で4対象とも 10.0000 倍を確認済み。
   係数の値に依存しないため再実施しない。
