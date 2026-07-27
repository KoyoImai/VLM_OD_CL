# exp_028 実行手順書（クラスタ）

**条件A（全モジュール）+ 特徴蒸留（L2ノルム）+ リプレイ** を4条件で実行する。

設計は [design.md](./design.md)（実験5 の共通仕様を含む）。実験ノートは
[note05.md](../../experiment_notes/note05.md)。対照は [exp_027](../exp_027/README.md)。
条件B 版は [exp_029](../exp_029/README.md)。

すべて**クラスタのマスターノード**で操作する（計算ノードへの直接 SSH 不可）。

---

## 0. この実験の内容

| 条件 | 蒸留対象 | ジョブ |
|---|---|---|
| 028-A | 画像（neck 出力） | **exp_028_a** |
| 028-B | テキスト（`text_feat_map` 出力） | **exp_028_a** |
| 028-D | 融合後（feature enhancer の `memory` と `memory_text`） | **exp_028_b** |
| 028-E | 上記3つすべて（3項平均） | **exp_028_b** |

- 教師は θ_{t-1}（t=1 は θ0）。凍結して `no_grad` で前向きする。
- 蒸留は**バッファ内データのみ**に適用（1バッチ6枚のうち末尾2枚）。
- **蒸留重み λ = 1.0 固定**（2026-07-27 決定）。
- リプレイ・データ・スケジュールは exp_027 と完全に同一。差分は蒸留の有無だけ。
- 逐次順は underwater → electromagnetic → videogames。

**`_a` と `_b` は独立したジョブなので並列に投入できる。**

---

## 1. 前提

### 1.1 対照の完了
exp_027-1（条件A・リプレイのみ）が3ドメイン完了していること。**完了済み**（§5 に実測値）。

### 1.2 コード同期
```bash
# 本環境
git add mmdet/models/detectors/kd_grounding_dino.py \
        mmdet/models/losses/feature_distill_loss.py \
        experiments/exp_028 experiments/exp_029
git commit -m "add exp_028/029"
git push

# クラスタのマスターノード
cd /home/kouyou/VLM_OD_CL && git pull
```

新規コンポーネントは `mmdet/` に2本。`__init__.py` には登録せず、config の
`custom_imports` でフルパス指定して読み込む。

### 1.3 その他（exp_027 で確認済み）
- θ0: `/home/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`
- `rf100_domain`、`o365v1_stage`、`bert-base-uncased`、`/dataset01/MSCOCO`、`.sif`、`mkdir -p /home/kouyou/logs`
- **node03 の除外は `sbatch_{a,b}.sh` に組み込み済み**（`/local_cache` が用意されないため）

---

## 2. 実行

```bash
cd /home/kouyou/VLM_OD_CL
sbatch experiments/exp_028/sbatch_a.sh    # 蒸留A → 蒸留B（64時間＋α）
sbatch experiments/exp_028/sbatch_b.sh    # 蒸留D → 蒸留E（64時間＋α）
```

単一の条件だけを流す場合。

```bash
RUN_STAGE=A sbatch experiments/exp_028/sbatch_a.sh
RUN_STAGE=B sbatch experiments/exp_028/sbatch_a.sh
RUN_STAGE=D sbatch experiments/exp_028/sbatch_b.sh
RUN_STAGE=E sbatch experiments/exp_028/sbatch_b.sh
```

`_a` は `A|B|all`、`_b` は `D|E|all` のみ受理する（担当外を渡すとエラーで停止）。

### 蒸留重みを変える場合

既定は λ = 1.0。変えるときは `LAMBDA` を渡す。

```bash
LAMBDA=0.1 sbatch experiments/exp_028/sbatch_a.sh
```

参照ドリフト方式で対象ごとに校正する場合（design §4.5 の参考手順。既定では使わない）。

```bash
CALIBRATE=1 sbatch experiments/exp_028/sbatch_a.sh
```

---

## 3. 所要時間とクラスタ枠

exp_027-1 のログから算出した実測値。

| | |
|---|---|
| 1条件（3ドメイン逐次＋評価） | **31時間55分**（uw 8h52m → em 16h37m → vg 5h39m） |
| 1ジョブ（2条件を直列） | **64時間＋α**（α は教師の前向き分） |
| exp_028 全体（`_a` と `_b` を並列） | 同上 |

`--time` は 120 時間で十分な余裕がある。exp_029 と合わせて4ジョブを同時に流すと、
クラスタの同時実行上限（4ジョブ）をちょうど使い切り、L2 側の8条件が約3日で揃う。

---

## 4. 監視

```bash
squeue -u kouyou
tail -f /home/kouyou/logs/result_exp028a_kdAB_condA_l2_<JOBID>.txt
tail -f /home/kouyou/logs/result_exp028b_kdDE_condA_l2_<JOBID>.txt
```

### 学習ログに出る蒸留関連の値

| キー | 内容 | 合計損失に含まれるか |
|---|---|---|
| `loss_kd` | λ を掛けた後の蒸留損失 | **含まれる** |
| `kd_img` | neck 出力の蒸留損失（重み付け前） | 含まれない（診断用） |
| `kd_txt` | `text_feat_map` 出力の蒸留損失 | 含まれない |
| `kd_fus` | 融合後（画像側とテキスト側の平均） | 含まれない |
| `kd_fus_img` / `kd_fus_txt` | 融合後の内訳 | 含まれない |

診断値はキー名に `loss` を含まないため、mmengine の合計規則から外れて**学習には影響しない**。

**λ = 1.0 が強すぎる／弱すぎるかの判断**は、`loss_kd` と検出損失の比を追うとよい。
本環境の1ステップ目の実測では、蒸留E で `loss_kd` 2.35 対 検出損失 4.65（約半分）だった。

GPU 使用状況は Grafana: http://192.168.170.100:3000/dashboards （要ラボ内LAN）。

---

## 5. 結果の見方

```bash
cd /home/kouyou/VLM_OD_CL
# <kd> は A|B|D|E、<dom>/<evd> は underwater|electromagnetic|videogames
grep -oE "coco/bbox_mAP: [0-9.]+" \
  experiments/exp_028/kd<kd>_condA_l2_eval_after_<dom>/on_<evd>/*/*.log | tail -1
grep -oE "coco/bbox_mAP: [0-9.]+" \
  experiments/exp_028/kd<kd>_condA_l2_eval_after_<dom>/zcoco/*/*.log | tail -1
```

**評価結果の json は1ファイルに複数レコードが改行なしで連結されている場合がある。**
`json.load` は失敗するので、`json.JSONDecoder().raw_decode` で全レコードを読み、
最後のものを採る（exp_024〜027 で確認済み）。

### 対照（exp_027-1：条件A・リプレイのみ・蒸留なし）

蒸留の効果はこの値との差で読む。

| 学習後 | on_underwater | on_electromagnetic | on_videogames | ZCOCO |
|---|---:|---:|---:|---:|
| underwater | **0.337** | — | — | 0.411 |
| electromagnetic | 0.229 | **0.475** | — | 0.382 |
| videogames | 0.230 | 0.428 | **0.726** | 0.368 |

太字が適応 mAP。

### 上限・下限の参照点（exp_026）

| 参照点 | underwater | electromagnetic | videogames | ZCOCO |
|---|---:|---:|---:|---:|
| 個別チューニング（適応の上限・条件A） | 0.354 | 0.492 | 0.763 | — |
| オラクル（3ドメイン同時学習） | 0.344 | 0.479 | 0.749 | 0.286 |
| θ0（ZCOCO の上限） | — | — | — | 0.504 |
| リプレイなし逐次（exp_024・下限） | 0.073 | 0.199 | 0.760 | 0.102 |

### 差を読むときの基準

実験4 で実測したばらつきは、**同一環境の再実行で 0.001〜0.007、環境間で最大 0.012**。
これより小さい差は手法の差として読めない。

### 判定基準

**適応と忘却のトレードオフを解決しているものを残す**（design §6）。判定はユーザーが行う。

---

## 6. 成果物の場所

| 種別 | パス |
|---|---|
| 学習出力 | `experiments/exp_028/kd{A,B,D,E}_condA_l2_{dom}_work_dir/` |
| 評価出力 | `experiments/exp_028/kd{A,B,D,E}_condA_l2_eval_after_{dom}/{on_{evd},zcoco}/` |
| λ 校正の記録 | `experiments/exp_028/calibration/`（`CALIBRATE=1` のときのみ） |
| ジョブログ | `/home/kouyou/logs/{result,error}_exp028{a,b}_*_<JOBID>.txt` |

**ckpt は 1 個 2.0GB のまま**（教師は state_dict に含めていないため肥大しない）。
4条件 × 3ドメイン × 20 エポック ＝ **約 480GB**。

本環境への回収は [RETRIEVE_FROM_CLUSTER.md](../RETRIEVE_FROM_CLUSTER.md) を参照。
**rsync 実行中に `git add -A` を使わないこと**（転送中の一時ファイルを巻き込む。
2026-07-27 に push 失敗の実例あり）。

---

## 7. 構成ファイル

| ファイル | 役割 |
|---|---|
| `sbatch_{a,b}.sh` | 器。`#SBATCH`＋rf100 の local_cache ステージング＋Singularity 起動（o365 の bind と node03 除外を含む） |
| `train_val_{a,b}.sh` | 中身。`RUN_STAGE` に応じてドライバを呼ぶ |
| `run_sequential_kd.sh` | 逐次ドライバ。**exp_028〜031 で共用**（引数: 実験・条件・形式・蒸留対象） |
| `calibrate_lambda.py` | λ の校正（既定では未使用。診断にも使える） |
| `configs/kd{A,B,D,E}_condA_l2_{dom}.py` | 12本。exp_027 の config を継承し、detector と蒸留設定のみ追加 |
| `mmdet/models/detectors/kd_grounding_dino.py` | 教師の保持と蒸留損失の追加 |
| `mmdet/models/losses/feature_distill_loss.py` | 位置ごとの距離（L2／コサイン）を有効位置で平均 |

評価 config は `experiments/exp_023/configs/eval_{dom}.py` と
`configs/mm_grounding_dino/eval_base_coco.py` を流用する（ckpt のキー構成が
素の GroundingDINO と同一のため、そのまま使える）。

---

## 8. 補足：本環境で実行する場合

```bash
bash experiments/exp_028/run_sequential_kd.sh exp_028 condA l2 A
```

第4引数を `B` / `D` / `E` に変えて他の条件を実行する。θ0 は `THETA0` 未設定なら
torch hub キャッシュから読む。
