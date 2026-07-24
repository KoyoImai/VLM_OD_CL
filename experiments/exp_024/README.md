# exp_024 / exp_025 / exp_026 実行手順書（クラスタ）

note04（実験1＝比較ベースラインの整備）のうち、exp_023 で埋まらない残りをクラスタで実行するための手順書。
3実験の手順をまとめて本ファイルに記載する。

設計・判定の考え方は各 design.md を参照：
[exp_024](./design.md) ／ [exp_025](../exp_025/design.md) ／ [exp_026](../exp_026/design.md)。
クラスタ運用の全体像は [README_cluster.md](../../README_cluster.md)、移行検証は [exp_023.5](../exp_023.5/design.md)。

すべて**クラスタのマスターノード**で操作する（計算ノードへの直接 SSH 不可）。

---

## 0. 3実験の内容

| 実験 | 内容 | 学習範囲 | 逐次 | ジョブ数 |
|---|---|---|---|---|
| **exp_024** | リプレイ無し逐次FT（条件A） | 全モジュール | あり | 1（3ドメイン直列） |
| **exp_025** | リプレイ無し逐次FT（条件B） | 特徴抽出+融合 | あり | 1（3ドメイン直列） |
| **exp_026** | 個別チューニング(A/B)・個別ZiRa・オラクル・ゼロショット | 各種 | なし | 5（段階分割） |

- 条件A＝全モジュール、条件B＝特徴抽出+融合（note04 の定義）。
- 逐次順は underwater → electromagnetic → videogames。
- exp_024/025 の差分は `paramwise_cfg`（学習範囲）のみ。exp_026 の個別チューニングは
  exp_024/025 の config をそのまま流用し、**初期値が常に θ0** である点だけが逐次と異なる。

---

## 1. 前提（着手前に必ず確認）

### 1.1 exp_023.5 の完了
クラスタで学習・評価が端まで通ることの確認（Tier2）が完了していること。**完了済み**：
ZCOCO が本環境と同値、1エポック学習がエラーなく完走、評価も正常。

### 1.2 コード同期（本環境＝正本 → クラスタ）
```bash
# 本環境
git add experiments/exp_024 experiments/exp_025 experiments/exp_026
git commit -m "add exp_024/025/026"
git push

# クラスタのマスターノード
cd /home/kouyou/VLM_OD_CL && git pull
```

### 1.3 データの確認（**exp_026 のゼロショットは 6 ドメイン全部を使う**）
```bash
ls -1 /home/kouyou/datasets/rf100_domain/
# 期待: aerial documents electromagnetic microscopic underwater videogames （＋real world）
```
不足していれば [README_cluster.md](../../README_cluster.md) §4.1 の `rsync` で追加転送する。

### 1.4 その他（exp_023.5 で確認済み）
- θ0: `/home/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`
- `bert-base-uncased`、`/dataset01/MSCOCO`、`.sif`、`mkdir -p /home/kouyou/logs`

---

## 2. 推奨する投入順

同時実行は **4 ジョブ**まで。長時間の exp_024・025 が 2 枠を占めるため、順序が効く。

**まず exp_026 のゼロショットを単独で流す。** 学習なしで数十分、かつ新規作成した
aerial / microscopic / documents の評価 config を初めて動かすため、
**60時間級のジョブを投げる前に設定ミスを最小コストで検出できる**。

```bash
# ① 最短・検証を兼ねる（数十分）
RUN_STAGE=zeroshot sbatch experiments/exp_026/sbatch.sh

# ② 長時間ジョブ2本を並列投入（2枠を占有）
sbatch experiments/exp_024/sbatch.sh
sbatch experiments/exp_025/sbatch.sh

# ③ 残り2枠で exp_026 を段階的に（短いものから）
RUN_STAGE=indiv_zira  sbatch experiments/exp_026/sbatch.sh
RUN_STAGE=indiv_condA sbatch experiments/exp_026/sbatch.sh
RUN_STAGE=indiv_condB sbatch experiments/exp_026/sbatch.sh
RUN_STAGE=oracle      sbatch experiments/exp_026/sbatch.sh
```

---

## 3. 各実験の実行

### 3.1 exp_024（リプレイ無し逐次FT・条件A）
```bash
sbatch experiments/exp_024/sbatch.sh
```
1ジョブ内で underwater → electromagnetic → videogames を直列に学習し、各ドメイン後に
「現ドメイン＋全過去ドメイン＋ZCOCO」を評価する。t=1 は θ0、t≥2 は前ドメインの last を
`load_from`（重みのみ。LR・optimizer はリセット）。

### 3.2 exp_025（リプレイ無し逐次FT・条件B）
```bash
sbatch experiments/exp_025/sbatch.sh
```
exp_024 と同時に走らせてよい（別ジョブ）。下流（decoder / bbox_head / query 系）は
`lr_mult=0.0` で凍結される。

### 3.3 exp_026（参照点の整備）
段階ごとに投入する。`RUN_STAGE` を省略すると `all`（全段階を1ジョブで直列）になるが、
所要時間が長くなるため**段階分割を推奨**。

| RUN_STAGE | 内容 |
|---|---|
| `zeroshot` | θ0 で 6ドメイン＋COCO を評価（学習なし） |
| `indiv_zira` | 個別 ZiRa 3ドメイン（2000 iter・短い） |
| `indiv_condA` | 個別チューニング（条件A）3ドメイン |
| `indiv_condB` | 個別チューニング（条件B）3ドメイン |
| `oracle` | オラクル（条件A・3ドメイン同時学習） |

---

## 4. 所要時間の目安

exp_023 の実測（約 2.5 秒/iter）からの概算。

| 実験 | 概算 |
|---|---|
| exp_024 / exp_025 | 各 **45〜50 時間** |
| exp_026 `oracle` | 約 **30 時間** |
| exp_026 `indiv_condA` / `indiv_condB` | 各 **40 時間前後** |
| exp_026 `indiv_zira` | 数時間 |
| exp_026 `zeroshot` | 数十分 |

`--time` は 3 実験とも 120 時間（上限 168 時間に対し余裕を確保）。

---

## 5. 監視

```bash
squeue -u kouyou                       # ジョブ状態
scancel <JOBID>                        # 中止

# ログ（%x=ジョブ名 %j=ジョブID）
tail -f /home/kouyou/logs/result_exp024_ftA_replayfree_<JOBID>.txt
tail -f /home/kouyou/logs/result_exp025_ftB_replayfree_<JOBID>.txt
tail -f /home/kouyou/logs/result_exp026_refpoints_<JOBID>.txt
```
GPU 使用状況は Grafana: http://192.168.170.100:3000/dashboards （要ラボ内LAN）。
`nvidia-smi` / `htop` は使用不可。

---

## 6. 結果の見方

```bash
cd /home/kouyou/VLM_OD_CL

# exp_024（条件A）: 各ドメイン学習後の 適応 / 過去ドメイン / ZCOCO
grep -oE "coco/bbox_mAP: [0-9.]+" experiments/exp_024/eval_after_<dom>/on_<evd>/*/*.log | tail -1
grep -oE "coco/bbox_mAP: [0-9.]+" experiments/exp_024/eval_after_<dom>/zcoco/*/*.log | tail -1

# exp_025（条件B）: 同じ構成
grep -oE "coco/bbox_mAP: [0-9.]+" experiments/exp_025/eval_after_<dom>/on_<evd>/*/*.log | tail -1

# exp_026 個別チューニング / 個別ZiRa
grep -oE "coco/bbox_mAP: [0-9.]+" experiments/exp_026/indiv_<method>_eval/<dom>/on_<dom>/*/*.log | tail -1
grep -oE "coco/bbox_mAP: [0-9.]+" experiments/exp_026/indiv_<method>_eval/<dom>/zcoco/*/*.log | tail -1

# exp_026 オラクル
grep -oE "coco/bbox_mAP: [0-9.]+" experiments/exp_026/oracle_condA_eval/on_<dom>/*/*.log | tail -1

# exp_026 ゼロショット
grep -oE "coco/bbox_mAP: [0-9.]+" experiments/exp_026/zeroshot_eval/on_<dom>/*/*.log | tail -1
grep -oE "coco/bbox_mAP: [0-9.]+" experiments/exp_026/zeroshot_eval/zcoco/*/*.log | tail -1
```

### ゼロショットの判定（exp_001 の実測値と照合）
学習を伴わず評価は決定的なので、**ほぼ一致するはず**。乖離があれば θ0・データ・評価設定の
いずれかに差があることを示すため、原因を切り分ける。

| 対象 | 参照値 |
|---|---:|
| COCO2017 | 0.504 |
| underwater | 0.051 |
| aerial | 0.034 |
| electromagnetic | 0.024 |
| videogames | 0.016 |
| documents | 0.006 |
| microscopic | 0.002 |

---

## 7. 成果物の場所

| 実験 | 学習出力 | 評価出力 |
|---|---|---|
| exp_024 | `experiments/exp_024/fullft_replayfree_{dom}_work_dir/` | `experiments/exp_024/eval_after_{dom}/{on_{evd},zcoco}/` |
| exp_025 | `experiments/exp_025/condB_replayfree_{dom}_work_dir/` | `experiments/exp_025/eval_after_{dom}/{on_{evd},zcoco}/` |
| exp_026 個別 | `experiments/exp_026/indiv_{method}_{dom}_work_dir/` | `experiments/exp_026/indiv_{method}_eval/{dom}/{on_{dom},zcoco}/` |
| exp_026 オラクル | `experiments/exp_026/oracle_condA_work_dir/` | `experiments/exp_026/oracle_condA_eval/{on_{dom},zcoco}/` |
| exp_026 ゼロショット | （学習なし） | `experiments/exp_026/zeroshot_eval/{on_{dom},zcoco}/` |

ジョブログ: `/home/kouyou/logs/{result,error}_<ジョブ名>_<JOBID>.txt`

### 本環境への回収
成果物はクラスタのホーム側に残る。本環境へは `rsync` で回収する（[README_cluster.md](../../README_cluster.md) §7）。
```bash
# 本環境で実行
rsync -avh --progress \
  kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments/exp_024/ \
  /workspace/kouyou/mmdetection/experiments/exp_024/
```

**チェックポイント容量に注意**: 条件A/B は 1 epoch ごとに全保持（`max_keep_ckpts=-1`）で、
1個約 2.0 GB。1ドメイン 20 個＝40 GB、exp_024/025 で計 240 GB になる。
ホーム容量（約 3 TB）には収まるが、回収時は必要な ckpt（last）だけを選ぶとよい。

---

## 8. 構成ファイル

| ファイル | 役割 |
|---|---|
| `experiments/exp_02{4,5,6}/sbatch.sh` | 器。`#SBATCH`＋rf100 の local_cache ステージング＋Singularity 起動 |
| `experiments/exp_02{4,5,6}/train_val.sh` | 中身。コンテナ内でドライバを呼ぶ（Slurm/Singularity を知らない） |
| `experiments/exp_024/run_sequential_fullft_replayfree.sh` | exp_024 逐次ドライバ |
| `experiments/exp_025/run_sequential_condB_replayfree.sh` | exp_025 逐次ドライバ |
| `experiments/exp_026/run_indiv.sh` | 個別チューニング（condA / condB / zira） |
| `experiments/exp_026/run_oracle.sh` | オラクル |
| `experiments/exp_026/run_zeroshot.sh` | ゼロショット（6ドメイン＋COCO） |

`train_val.sh` は `sbatch.sh` からコンテナ内で呼ばれる実体。通常は直接触らない。

---

## 9. 補足：本環境で実行する場合

クラスタを使わず本環境で走らせることもできる（GPU が空いている場合）。
`sbatch.sh` を経由せず、ドライバを直接実行する。
```bash
bash experiments/exp_024/run_sequential_fullft_replayfree.sh
bash experiments/exp_025/run_sequential_condB_replayfree.sh
bash experiments/exp_026/run_indiv.sh all
bash experiments/exp_026/run_oracle.sh
bash experiments/exp_026/run_zeroshot.sh
```
θ0 は `THETA0` 未設定なら config の URL（torch hub キャッシュ）から読む。
明示指定したい場合は `THETA0=<path> bash ...` とする。
