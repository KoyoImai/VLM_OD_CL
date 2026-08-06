# exp_035 実行手順（λ = 10.0）

design: [[design]]（2026-08-06 承認済み）。共通仕様は [[../exp_028/design]]、掃引の枠組みは [[../exp_033/design]]。

## 構成

| ジョブ | 蒸留対象 | ジョブ名 | PORT |
|---|---|---|---:|
| `exp_035_a` | A（画像）・B（テキスト） | `exp035a_kdAB_condA_l2w100` | 29820 |
| `exp_035_b` | D（融合後）・E（3項平均） | `exp035b_kdDE_condA_l2w100` | 29830 |

各ジョブは2条件を直列に実行する（1条件 ≒ 32 時間、1ジョブ ≒ 64 時間）。
2ジョブは並列に投入できる。

## 1. リポジトリをクラスタへ反映

本環境（このリポジトリ）で以下を実行し、クラスタ側で `git pull` する。

```bash
git add experiments/exp_035
git commit -m "add exp_035 (lambda=10.0)"
git push
```

クラスタのマスターノードで:

```bash
cd /home/kouyou/VLM_OD_CL
git pull
```

## 2. 投入

マスターノードで実行する。

```bash
# 2条件ずつ直列（推奨。各ジョブ約64時間）
sbatch experiments/exp_035/sbatch_a.sh
sbatch experiments/exp_035/sbatch_b.sh
```

蒸留対象を1つずつ分けて投入したい場合:

```bash
RUN_STAGE=A sbatch experiments/exp_035/sbatch_a.sh
RUN_STAGE=B sbatch experiments/exp_035/sbatch_a.sh
RUN_STAGE=D sbatch experiments/exp_035/sbatch_b.sh
RUN_STAGE=E sbatch experiments/exp_035/sbatch_b.sh
```

## 3. 状態確認

```bash
squeue -u kouyou
sacct -u kouyou --starttime today --format=JobID,JobName%32,State,Elapsed,NodeList
tail -f /home/kouyou/logs/result_exp035a_kdAB_condA_l2w100_<JOBID>.txt
```

## 4. 結果の取得（本環境へ）

本環境で実行する。

```bash
rsync -avz --progress \
  kouyou@<クラスタのホスト>:/home/kouyou/VLM_OD_CL/experiments/exp_035/ \
  /workspace/kouyou/mmdetection/experiments/exp_035/
```

## 投入前の注意

- exp_033 の λ=5.0 が3条件未完了（蒸留B は結果なし、蒸留A・E は underwater まで）。
  同時に走らせるとジョブ枠（同時実行上限4）を圧迫する。
- ディスクは 4条件 × 3ドメイン × 20 epoch × 2.0GB ＝ 約 480 GB。
- `--exclude=node03`（`/local_cache` が用意されないため。2026-07-25）。

## ★ λ の指定について（exp_033 の不具合を踏まえた対処）

ドライバ `experiments/exp_028/run_sequential_kd.sh` は 84 行目で `lam="${LAMBDA:-1.0}"` とし、
91 行目で `--cfg-options model.kd.loss_weight="$lam"` を渡す。**config に書いた値は
コマンドラインから上書きされる**ため、`LAMBDA` を設定しないと λ=1.0 で学習される。

exp_033 はこれに気づかず投入したため、**全12条件（work_dir 32件すべて）が実効 λ=1.0 で
学習されていた**（2026-08-06 に学習ログの `loss_kd` が λ によらず同一であることから検出。
work_dir の `vis_data/config.py` で `loss_weight: 1.0` を確認）。

exp_035 では `train_val_{a,b}.sh` で **`export LAMBDA=10.0`** を明示し、さらに起動時に
config の `loss_weight` と `LAMBDA` の一致を確認するガードを入れてある（食い違えば学習を始めない）。
ジョブログの冒頭に `[exp_035] 実効 λ = 10.0（config と一致）` が出ることを確認すること。

## 実行前検証の結果（2026-08-06、本環境で実施）

1. **config の差分**: 12本すべて、exp_028 の対応 config との差は `model.kd.loss_weight`
   （1.0 → 10.0）の1キーのみ。**OK**
2. **損失が線形にスケールすること**: 同一バッチ・同一 seed で λ=1.0 と λ=10.0 の `loss_kd` を比較。
   4対象すべてで**ちょうど 10.0000 倍**。**OK**

   | 対象 | λ=1.0 の `loss_kd` | λ=10.0 の `loss_kd` | 比 |
   |---|---:|---:|---:|
   | A（画像） | 3.174764e-02 | 3.174764e-01 | 10.0000 |
   | B（テキスト） | 3.272670e+00 | 3.272670e+01 | 10.0000 |
   | D（融合後） | 7.785039e-04 | 7.785039e-03 | 10.0000 |
   | E（3項平均） | 1.169966e+00 | 1.169966e+01 | 10.0000 |

   `kd_*` は重み付け前の診断値なので λ によらず不変であることも確認した。
3. **実効 λ の一致**: `train_val_{a,b}.sh` のガードを本環境で実行し、
   `[exp_035] 実効 λ = 10.0（config と一致）` を確認。**OK**
4. exp_028 の検証項目（教師の独立性・凍結・バッファ限定・勾配経路・マスク・数式一致）は
   係数に依存しないため再実施しない。
