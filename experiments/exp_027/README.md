# exp_027 実行手順書（クラスタ）

リプレイあり逐次FT を2条件（条件A＝全モジュール／条件B＝特徴抽出+融合）クラスタで実行するための手順書。

設計・判定の考え方は [design.md](./design.md) を参照。
クラスタ運用の全体像は [README_cluster.md](../../README_cluster.md)、移行検証は [exp_023.5](../exp_023.5/design.md)、
リプレイ無しの対になる実験は [exp_024](../exp_024/README.md)（同ファイルに exp_025/026 の手順も含む）。

すべて**クラスタのマスターノード**で操作する（計算ノードへの直接 SSH 不可）。

---

## 0. この実験の内容

| # | 内容 | 学習範囲 | リプレイ | ジョブ |
|---|---|---|---|---|
| **027-1** | 逐次FT + リプレイ（**条件A**） | 全モジュール | あり | 1（3ドメイン直列） |
| **027-2** | 逐次FT + リプレイ（**条件B**） | 特徴抽出+融合（下流凍結） | あり | 1（3ドメイン直列） |

これが入ると、クラスタ上に 2×2 が揃う。

| | リプレイなし | リプレイあり |
|---|---|---|
| **条件A（全モジュール）** | exp_024 | **exp_027-1** |
| **条件B（特徴抽出+融合）** | exp_025 | **exp_027-2** |

- 逐次順は underwater → electromagnetic → videogames。
- 呼称は exp_024 以降（note04 の定義）に統一。**条件A＝全モジュール／条件B＝特徴抽出+融合**。
- config は exp_023 の設定を `_base_` で丸ごと継承したエイリアス（上書きゼロ）。本環境の exp_023 手法5・6 との差は**実行環境だけ**なので、突き合わせれば環境間の再現性も見られる。

---

## 1. 前提（着手前に必ず確認）

### 1.1 exp_023.5 の完了
クラスタで学習・評価が端まで通ることの確認（Tier2）が完了していること。**完了済み**。

### 1.2 コード同期（本環境＝正本 → クラスタ）
```bash
# 本環境
git add experiments/exp_027
git commit -m "add exp_027"
git push

# クラスタのマスターノード
cd /home/kouyou/VLM_OD_CL && git pull
```
`experiments/exp_023/buffer/*.odvg.json`（計 3.3 MB）も git 管理下なので、`git pull` で配布される。

### 1.3 データの確認（**本実験は Objects365 を使う**）
exp_024/025 と違い、リプレイの参照バッファに Objects365v1 の 1,000 枚を使う。
```bash
ls -d /home/kouyou/datasets/o365v1_stage/Objects365_v1/2019-08-02/train
ls -1 /home/kouyou/datasets/rf100_domain/     # underwater / electromagnetic / videogames が必要
```
**この確認は §2 の `RUN_STAGE=debug` が自動でやる**ので、目視は省略してよい。

### 1.4 その他（exp_023.5 で確認済み）
- θ0: `/home/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`
- `bert-base-uncased`、`/dataset01/MSCOCO`、`.sif`、`mkdir -p /home/kouyou/logs`

---

## 2. 実行手順

### ステップ1: デバッグ実行（**必ず最初に1回**）

本実験がクラスタで新しく必要とするのは「Objects365 のバッファ画像が読めること」だけ。
43時間級のジョブを投げてから転送漏れが発覚すると損失が大きいので、先に数分〜十数分で潰す。

```bash
RUN_STAGE=debug sbatch experiments/exp_027/sbatch.sh
```

6項目を確認し、失敗した項目を明示して非ゼロ終了する。

| # | 確認内容 |
|---|---|
| 1 | bind：Objects365 のステージングディレクトリが見えるか |
| 2 | buffer：バッファ JSON が読めるか（参照1000 / 各ドメイン500 の件数照合） |
| 3 | exists：バッファが指す画像が実在するか（**転送漏れはここで捕まる**） |
| 4 | decode：実際にデコードできるか（転送途中で切れたファイルの検出） |
| 5 | mixing：1バッチの内訳が t=1 で現在4・参照2、t≥2 で現在4・参照1・過去1 か |
| 6 | step：θ0 をロードして `loss()` が通り、条件B の実効 lr が 0 か |

ログ末尾に判定が出る。**「全項目 OK」を確認してから次へ進む**。

```bash
tail -30 /home/kouyou/logs/result_exp027_replay_<JOBID>.txt
```

### ステップ2: 本実行（2ジョブを並列投入）

```bash
RUN_STAGE=fullft sbatch experiments/exp_027/sbatch.sh   # 027-1 条件A + リプレイ
RUN_STAGE=condB  sbatch experiments/exp_027/sbatch.sh   # 027-2 条件B + リプレイ
```

各ジョブが1本で underwater → electromagnetic → videogames を直列に学習し、各ドメイン後に
「現ドメイン＋全過去ドメイン＋ZCOCO」を評価する。t=1 は θ0、t≥2 は前ドメインの last を
`load_from`（重みのみ。LR・optimizer はリセット）。

`RUN_STAGE` を省略すると `all`（debug → fullft → condB を1ジョブで直列）になるが、
86時間超になるため**段階分割を推奨**。

---

## 3. 所要時間とクラスタ枠

exp_023 の実測（electromagnetic が 1,588 iter/epoch × 20 epoch で 22時間34分、約 2.5 秒/iter）からの概算。

| 段階 | 概算 |
|---|---|
| `debug` | 数分〜十数分 |
| `fullft`（027-1） | **43 時間前後**（学習 約40h ＋ 評価 約3h） |
| `condB`（027-2） | **43 時間前後** |

`--time` は 120 時間（上限 168 時間に対し余裕を確保）。

**枠の注意**: クラスタは同時 **4 ジョブ**まで。exp_024・exp_025 と exp_027 の2本で 4 枠が埋まるため、
exp_026 は空きが出てから流すことになる。`debug` は短いので枠が一時的に埋まっていても差し支えない。

---

## 4. 監視

```bash
squeue -u kouyou                       # ジョブ状態
scancel <JOBID>                        # 中止

tail -f /home/kouyou/logs/result_exp027_replay_<JOBID>.txt
tail -f /home/kouyou/logs/error_exp027_replay_<JOBID>.txt
```
027-1 と 027-2 はジョブ名が同じ（`exp027_replay`）なので、**JOBID で区別する**。
どちらかはログ冒頭の `RUN_STAGE = ...` 行で分かる。

GPU 使用状況は Grafana: http://192.168.170.100:3000/dashboards （要ラボ内LAN）。
`nvidia-smi` / `htop` は使用不可。

---

## 5. 結果の見方

```bash
cd /home/kouyou/VLM_OD_CL

# 027-1（条件A + リプレイ）
grep -oE "coco/bbox_mAP: [0-9.]+" experiments/exp_027/fullft_replay_eval_after_<dom>/on_<evd>/*/*.log | tail -1
grep -oE "coco/bbox_mAP: [0-9.]+" experiments/exp_027/fullft_replay_eval_after_<dom>/zcoco/*/*.log | tail -1

# 027-2（条件B + リプレイ）
grep -oE "coco/bbox_mAP: [0-9.]+" experiments/exp_027/condB_replay_eval_after_<dom>/on_<evd>/*/*.log | tail -1
grep -oE "coco/bbox_mAP: [0-9.]+" experiments/exp_027/condB_replay_eval_after_<dom>/zcoco/*/*.log | tail -1
```

### 参照値1: θ0 ゼロショット（exp_001 実測）

| underwater | electromagnetic | videogames | COCO |
|---:|---:|---:|---:|
| 0.051 | 0.024 | 0.016 | 0.504 |

### 参照値2: 本環境（exp_023 手法5＝条件A + リプレイ）の実測

027-1 と**同一設定・別環境**の結果。突き合わせれば環境間の再現性が見られる。

| 学習後の状態 | on_underwater | on_electromagnetic | ZCOCO |
|---|---:|---:|---:|
| underwater まで | 0.3440 | — | 0.4120 |
| electromagnetic まで | 0.2410 | 0.4860 | 0.3740 |
| videogames まで | 本環境で実行中 | 実行中 | 実行中 |

`deterministic=False` のため完全一致は保証されない。乖離があっても、環境差か実行間のばらつきかは
本実験だけでは切り分けられない（design.md §8）。

---

## 6. 成果物の場所

| 実験 | 学習出力 | 評価出力 |
|---|---|---|
| 027-1 | `experiments/exp_027/fullft_replay_{dom}_work_dir/` | `experiments/exp_027/fullft_replay_eval_after_{dom}/{on_{evd},zcoco}/` |
| 027-2 | `experiments/exp_027/condB_replay_{dom}_work_dir/` | `experiments/exp_027/condB_replay_eval_after_{dom}/{on_{evd},zcoco}/` |

ジョブログ: `/home/kouyou/logs/{result,error}_exp027_replay_<JOBID>.txt`

### 本環境への回収
```bash
# 本環境で実行
rsync -avh --progress \
  kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments/exp_027/ \
  /workspace/kouyou/mmdetection/experiments/exp_027/
```

**チェックポイント容量に注意**: 1 epoch ごとに全保持（`max_keep_ckpts=-1`）で 1個 約 2.0 GB。
2手法 × 3ドメイン × 20 個 ＝ **約 240 GB**。exp_024/025 の 240 GB と合わせて約 480 GB で、
ホーム（約 3 TB）には収まる。回収時は必要な ckpt（last）だけを選ぶとよい。

---

## 7. 構成ファイル

| ファイル | 役割 |
|---|---|
| `sbatch.sh` | 器。`#SBATCH`＋rf100 の local_cache ステージング＋Singularity 起動。**o365 の bind を含む**（exp_024/025 との唯一の差） |
| `train_val.sh` | 中身。コンテナ内で `RUN_STAGE` に応じてドライバを呼ぶ（Slurm/Singularity を知らない） |
| `debug_o365.py` | デバッグ実行の実体（§2 ステップ1 の6項目） |
| `run_sequential_fullft_replay.sh` | 027-1 逐次ドライバ（PORT 29620） |
| `run_sequential_condB_replay.sh` | 027-2 逐次ドライバ（PORT 29640） |
| `configs/fullft_replay_{dom}.py` | 条件A。`exp_023/configs/fullft_replay_{dom}.py` の継承エイリアス（上書きゼロ） |
| `configs/condB_replay_{dom}.py` | 条件B。`exp_023/configs/condA_replay_{dom}.py` の継承エイリアス（継承元は旧呼称。中身は特徴抽出+融合） |

評価 config は `experiments/exp_023/configs/eval_{dom}.py` と
`configs/mm_grounding_dino/eval_base_coco.py` を流用（exp_024/025/026 と同じ）。

---

## 8. 補足：本環境で実行する場合

クラスタを使わず本環境で走らせることもできる。`sbatch.sh` を経由せずドライバを直接実行する。
```bash
python experiments/exp_027/debug_o365.py                    # デバッグ（--quick で GPU 不要の1〜4のみ）
bash experiments/exp_027/run_sequential_fullft_replay.sh    # 027-1
bash experiments/exp_027/run_sequential_condB_replay.sh     # 027-2
```
θ0 は `THETA0` 未設定なら config の URL（torch hub キャッシュ）から読む。
明示指定したい場合は `THETA0=<path> bash ...` とする。
`debug_o365.py` の θ0 はクラスタの `/workspace/kouyou/ckpt/` と本環境の torch hub キャッシュを自動で探す。

**注意**: 本環境では exp_023 が同じ内容（手法5・6）を実行中。重複して走らせないこと。
