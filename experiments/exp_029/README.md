# exp_029 実行手順書（クラスタ）

**条件B（特徴抽出+融合）+ 特徴蒸留（L2ノルム）+ リプレイ** を4条件で実行する。

設計は [design.md](./design.md)（共通仕様は [exp_028/design.md](../exp_028/design.md)）。
実験ノートは [note05.md](../../experiment_notes/note05.md)。対照は [exp_027](../exp_027/README.md)。
条件A 版は [exp_028](../exp_028/README.md)。

**手順は exp_028 とほぼ同一である。** 本書は差分を中心に記す。共通部分は
[exp_028/README.md](../exp_028/README.md) を参照。

すべて**クラスタのマスターノード**で操作する。

---

## 0. この実験の内容

| 条件 | 蒸留対象 | ジョブ |
|---|---|---|
| 029-A | 画像（neck 出力） | **exp_029_a** |
| 029-B | テキスト（`text_feat_map` 出力） | **exp_029_a** |
| 029-D | 融合後（feature enhancer の `memory` と `memory_text`） | **exp_029_b** |
| 029-E | 上記3つすべて（3項平均） | **exp_029_b** |

### exp_028 との違い

| 項目 | exp_028 | exp_029 |
|---|---|---|
| 学習範囲 | 条件A（全モジュール） | **条件B（特徴抽出+融合）** |
| 継承元 config | `exp_027/configs/fullft_replay_*` | **`exp_027/configs/condB_replay_*`** |
| 対照 | exp_027-1 | **exp_027-2** |
| 蒸留の定式化・λ・データ・スケジュール | — | **同一** |

条件B で凍結されるのは decoder / bbox_head / memory_trans_fc / memory_trans_norm /
query_embedding / dn_query_generator（実効 lr = 0、229 パラメータ）。
蒸留対象の3つはいずれも学習対象のモジュールが生成するので、勾配経路は成立する。

### 結果を読む前に知っておくこと

**条件B では encoder 出力より下流が教師と学生でビット単位で同一**である（凍結のため）。
そのため **029-D は、$\mathcal{L}_{\mathrm{fus}} \to 0$ の極限でバッファ内データの検出結果が
教師と一致する**という性質を持つ。蒸留A・B にこの性質は無い。

結果として、exp_029 内の A/B/D 比較は「どこに蒸留をかけるか」の均質な比較になっていない
（D だけが学習範囲の出口にあり、下流全体を固定する別種の制約になる）。詳細は
[design.md §3](./design.md) を参照。

---

## 1. 前提

exp_028 の §1 と同じ。追加で、**対照の exp_027-2（条件B・リプレイのみ）が3ドメイン完了**
していること。**完了済み**（§5 に実測値）。

コード同期は exp_028 と共通（`mmdet/` の新規2本と `experiments/exp_028`・`exp_029`）。
**`run_sequential_kd.sh` は exp_028 のものを共用する**ので、exp_029 のディレクトリには置いていない。

---

## 2. 実行

```bash
cd /home/kouyou/VLM_OD_CL
sbatch experiments/exp_029/sbatch_a.sh    # 蒸留A → 蒸留B（64時間＋α）
sbatch experiments/exp_029/sbatch_b.sh    # 蒸留D → 蒸留E（64時間＋α）
```

単一条件のみ流す場合は `RUN_STAGE=A`（`_a` は `A|B|all`、`_b` は `D|E|all`）。
蒸留重みを変える場合は `LAMBDA=<値>`、校正する場合は `CALIBRATE=1`（既定は λ = 1.0 固定）。

**exp_028 の2ジョブと合わせて4本を同時に投入できる**（クラスタの同時実行上限ちょうど）。

---

## 3. 所要時間

1条件 **実測 31時間55分**（exp_027-1 のログから算出）、1ジョブ（2条件直列）64時間＋α。
`_a` と `_b` を並列に流せば exp_029 全体もほぼ同じ。`--time` は 120 時間で余裕がある。

---

## 4. 監視

```bash
squeue -u kouyou
tail -f /home/kouyou/logs/result_exp029a_kdAB_condB_l2_<JOBID>.txt
tail -f /home/kouyou/logs/result_exp029b_kdDE_condB_l2_<JOBID>.txt
```

学習ログに出る蒸留関連の値（`loss_kd` と診断用の `kd_img` / `kd_txt` / `kd_fus` /
`kd_fus_img` / `kd_fus_txt`）は exp_028 の §4 と同じ。診断値は合計損失に含まれない。

---

## 5. 結果の見方

```bash
cd /home/kouyou/VLM_OD_CL
grep -oE "coco/bbox_mAP: [0-9.]+" \
  experiments/exp_029/kd<kd>_condB_l2_eval_after_<dom>/on_<evd>/*/*.log | tail -1
grep -oE "coco/bbox_mAP: [0-9.]+" \
  experiments/exp_029/kd<kd>_condB_l2_eval_after_<dom>/zcoco/*/*.log | tail -1
```

### 対照（exp_027-2：条件B・リプレイのみ・蒸留なし）

| 学習後 | on_underwater | on_electromagnetic | on_videogames | ZCOCO |
|---|---:|---:|---:|---:|
| underwater | **0.340** | — | — | 0.415 |
| electromagnetic | 0.241 | **0.473** | — | 0.383 |
| videogames | 0.230 | 0.417 | **0.708** | 0.370 |

太字が適応 mAP。**この値との差で蒸留の効果を読む。**

### 参考：条件A の対照（exp_027-1）

条件A と条件B の比較をする場合はこちら。

| 学習後 | on_underwater | on_electromagnetic | on_videogames | ZCOCO |
|---|---:|---:|---:|---:|
| underwater | 0.337 | — | — | 0.411 |
| electromagnetic | 0.229 | 0.475 | — | 0.382 |
| videogames | 0.230 | 0.428 | 0.726 | 0.368 |

蒸留なしの時点で、条件A と条件B の差は9項目すべて 0.018 以内（最大は t=3 適応の
0.726 対 0.708）。**ばらつき（同一環境の再実行で 0.001〜0.007）と同程度から少し大きい程度**である。

### 上限・下限の参照点、判定基準

exp_028 の §5 と同じ（個別チューニング条件B: underwater 0.352 / electromagnetic 0.496 /
videogames 0.744、オラクル、θ0 の ZCOCO 0.504、リプレイなし逐次 exp_025）。

判定基準は **適応と忘却のトレードオフを解決しているものを残す**。判定はユーザーが行う。

---

## 6. 成果物の場所

| 種別 | パス |
|---|---|
| 学習出力 | `experiments/exp_029/kd{A,B,D,E}_condB_l2_{dom}_work_dir/` |
| 評価出力 | `experiments/exp_029/kd{A,B,D,E}_condB_l2_eval_after_{dom}/{on_{evd},zcoco}/` |
| λ 校正の記録 | `experiments/exp_029/calibration/`（`CALIBRATE=1` のときのみ） |
| ジョブログ | `/home/kouyou/logs/{result,error}_exp029{a,b}_*_<JOBID>.txt` |

ckpt は 1 個 2.0GB、4条件 × 3ドメイン × 20 エポック ＝ **約 480GB**。
exp_028 と合わせて約 960GB。回収は [RETRIEVE_FROM_CLUSTER.md](../RETRIEVE_FROM_CLUSTER.md)。

---

## 7. 構成ファイル

| ファイル | 役割 |
|---|---|
| `sbatch_{a,b}.sh` | 器（o365 の bind と node03 除外を含む）。ジョブ名 `exp029{a,b}_kd{AB,DE}_condB_l2`、PORT 29680 / 29690 |
| `train_val_{a,b}.sh` | 中身 |
| `configs/kd{A,B,D,E}_condB_l2_{dom}.py` | 12本。`exp_027/configs/condB_replay_{dom}.py` を継承し、detector と蒸留設定のみ追加（`paramwise_cfg` は継承元のまま） |

ドライバ `run_sequential_kd.sh`、detector、loss、`calibrate_lambda.py` は
**exp_028 のものを共用**する。

---

## 8. 補足：本環境で実行する場合

```bash
bash experiments/exp_028/run_sequential_kd.sh exp_029 condB l2 A
```

第1引数が `exp_029`、第2引数が `condB` である点に注意。第4引数を `B` / `D` / `E` に変えて
他の条件を実行する。
