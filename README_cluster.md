# README_cluster — MPRG GPU Cluster での実行手順

本環境（`/workspace/kouyou/mmdetection`）を**正本**、MPRG GPU Cluster を**実行専用環境**として運用するための手順をまとめる。
クラスタでは config・スクリプトを一切変更せず、Singularity の `--bind` によるパス間接化で本環境と同じ `/workspace/kouyou/...` を再現する。

> 記法: 本書では確定方針を通常記述、未確認・未決定事項を「⚠️ 要確認」で示す。

---

## 0. 位置づけと役割分担

|    役割   |                 環境                  |                         内容                   |
|----------|---------------------------------------|-----------------------------------------------|
| 実装・実行 | 本環境 `/workspace/kouyou/mmdetection` | コード・config・実験設計。ここで編集し git へ push |
|    実行   | クラスタ（Slurm + Singularity）         | 学習・評価のみ。コードは git pull で受け取る       |

- クラスタへは**マスターノードにログイン**し `sbatch` でジョブ投入する。計算ノードへの直接 SSH・`nvidia-smi` は不可（監視は Grafana: `http://192.168.170.100:3000/dashboards`、要ラボ内LAN）。
- 本環境からクラスタ（192.168.170.100）への接続は可能．`docker clone`でプログラムをクラスタ側で用意し，実行結果は`scp`で本環境へ転送する．

---

## 1. クラスタ側のディレクトリ前提
ユーザー名`kouyou`で，ディレクトリ構成は以下の通り．
```
/home/kouyou/
├── mmdetection/                     # 本 repo（git clone / pull で同期）
├── datasets/
│   ├── rf100_domain/                # rf100 各ドメイン（本環境からコピー）
│   └── o365v1_stage/Objects365_v1/2019-08-02/train/    # o365 全体を事前転送（zip 除外可）
└── sif/env.sif                      # Singularity イメージ（依存のみ）
```

- ホーム（NFS, ファイルサーバ RAID6）は**全ノード共通マウント**．ただし NFS 直読みはデータローダのボトルネックになるため，学習時は後述の local_cache へステージングして読む．
- **ホーム容量は 約3TB 制限**．超過するとファイルが作れない／サイズ0で作られる。`quota` で確認が必要．

---

## 2. 環境（依存）: Singularity `.sif` の準備 — 低頻度

本環境の `Dockerfile`（下記）を基に `.sif` を作成する．
**依存のみを固め、mmdet のコードは焼き込まない**（コードは git 同期・editable import で拾う）．

- ベース: `nvidia/cuda:12.8.0-devel-ubuntu24.04`
- Python 3.10（Miniconda）
- torch 2.1.2 + cu121 / torchvision 0.16.2 / torchaudio 2.1.2
- mmengine 0.10.3 / mmcv 2.1.0 / mmdet 3.3.0（editable）

クラスタ計算ノードのドライバは 570.124 / CUDA 12.8 で、cu121 wheel と後方互換．

作成手順:

```bash
# 本環境
## 手順1：DockerコンテナからDockerイメージを作成
docker commit 7cc88b572576 docker-image-of-mmdetection4singularity:cuda12.8-torch2.1

## 手順2：Dockerイメージを圧縮
docker save docker-image-of-mmdetection4singularity:cuda12.8-torch2.1 -o docker-image-of-mmdetection4singularity.tar

# クラスタ環境
## 手順3：Singularityイメージの作成
singularity build --fakeroot docker-image-of-mmdetection4singularity.sif docker-archive://docker-image-of-mmdetection4singularity.tar
```

⚠️ 要確認: `.sif` にコードを焼かない構成のため，`import mmdet` は cwd=repo ルート＋`--bind` した repo から解決させる．
ビルド後にインタラクティブジョブで `import mmdet, mmcv` と CUDA 可視を確認する．

---

## 3. コード同期: git（本環境 = 正本）

```bash
# 本環境で編集 → commit → push（remote: github.com/KoyoImai/VLM_OD_CL.git）
git add -A && git commit -m "..." && git push

# クラスタのマスターノードで
cd /home/kouyou/mmdetection && git pull
```

**前提整備**: 現状 `experiments/` に work_dir・checkpoint が約321G あり，そのままでは git に載らない．
`.gitignore` で学習出力（`experiments/**/*_work_dir/`, `experiments/**/eval_*/` 等）を除外してから同期に載せる．

---

## 4. データ準備（必要分のみホームへ）

測定により、実験1（3ドメイン: underwater → electromagnetic → videogames）で必要なデータは小さい。

| データ | 内容 | 容量 |
|---|---|---|
| rf100 3ドメイン | underwater 898M + electromagnetic 1.2G + videogames 680M（＋各 valid） | 約 2.8G |
| o365（Objects365 v1） | リプレイ参照元。全データを事前転送（`train/` を丸ごと置く） | 46G（`train/`のみ。全体 111G） |
| COCO2017 val（ZCOCO） | クラスタ共有SSD `/dataset01/MSCOCO` を利用（ホームに置かない） | 0（共有） |

- **方針変更（2026-07-24）**: o365 は 1000枚パックを作らず、**全データを事前にホームへ転送**する（抽出スクリプト不要）。学習・評価で実際に読むのは参照バッファの 1000枚だけだが、config の `_o365_root` は o365 全体ツリー（`data_prefix='train/'`）を指すため、`train/` を丸ごと置いておけば config 無変更で解決する。
- 転送は `train/`（46G）＋ ルートの `o365v1_label_map.json` があれば足りる。`*.zip`（≈65G、`train/`と重複）と `test/`・`val/`（≈9G、リプレイ未使用）は除外してよい。全部送るなら 111G。
- ホーム容量は約3TB のため 46〜111G は収まる。`rf100_domain/` と共にホーム `/home/kouyou/datasets/` へ転送。

### 4.1 転送コマンド（本環境 → クラスタ）

本環境からクラスタ（192.168.170.100）へ直接 `rsync` で転送する（`rsync` は中断再開・差分転送・構成保持に強く、`scp` より安全）。

**RF100（全ドメイン, 11G）**: 実験1は3ドメインだが、6ドメイン最終段階を見据えて全7ドメインを転送する（`real world` 5.5G を含む。使わないなら `--exclude 'real world'` で除外可）。

```bash
# 転送先の親ディレクトリを用意（未作成なら）
ssh kouyou@192.168.170.100 'mkdir -p /home/kouyou/datasets'

# rf100_domain/ をまるごと転送（-a: 構成保持, -h: 可読, --progress: 進捗）
rsync -avh --progress \
  /data1/kouyou/datasets/rf100_domain/ \
  kouyou@192.168.170.100:/home/kouyou/datasets/rf100_domain/
```

**o365(Objects365 v1, 全データ事前転送)**: `train/` 実体があれば足りるので `*.zip`（≈65G, `train/`と重複）は除外して転送するのが無駄がない。リプレイ手法はこの `train/` が無いと動かない。

```bash
# zip を除外して転送（推奨。train/ の実体画像は残る）
rsync -avh --progress --exclude '*.zip' \
  /data1/kouyou/datasets/o365v1_stage/ \
  kouyou@192.168.170.100:/home/kouyou/datasets/o365v1_stage/
# ↑ zip 含む 111G を全部送るなら --exclude を外す
```

- 末尾スラッシュに注意：ソース `foo/` の**中身**を宛先 `foo/` 直下へ入れる。
- スペースを含む `real world/` はディレクトリごと一括転送するので、この転送では引用符不要（個別に触るときのみ `"real world"` と引用）。

### 4.2 転送後の検証

```bash
# RF100: サイズとドメイン数
ssh kouyou@192.168.170.100 'du -sh /home/kouyou/datasets/rf100_domain; ls -1 /home/kouyou/datasets/rf100_domain | wc -l'
# 期待: 11G / 7

# o365: train 画像枚数（全体を転送した場合）
ssh kouyou@192.168.170.100 'find /home/kouyou/datasets/o365v1_stage/Objects365_v1/2019-08-02/train -type f | wc -l'
# 期待: 608606
```

---

## 5. パス間接化の仕組み（config 無変更の要）

config が持つ絶対パスは計42箇所（`/workspace/kouyou/datasets/...` 33件、`/workspace/kouyou/mmdetection/experiments/...` 9件）。これらは**コンテナ内部のパス**として扱い、`--bind` で実体を差し替える。

| 層 | 値 | 決定者 |
|---|---|---|
| config が読むパス | `/workspace/kouyou/datasets/...` | 固定・無変更 |
| バインド | ホスト `/local_cache/$JOB_ID/datasets` → コンテナ `/workspace/kouyou/datasets` | sbatch の `--bind` |
| 物理実体 | 計算ノード直結SSD（local_cache） | ステージングのコピー先 |

**唯一の条件**: ステージング先の**相対ディレクトリ構成を config の期待と一致**させる（config 修正ではなくコピー時の配置の話）。

---

## 6. ジョブ実行: ステージング付き sbatch テンプレート

`/local_cache/$SLURM_JOB_ID` は各ジョブ専用のローカルSSD領域で、**ジョブ終了時に自動削除**される（成果物は `--work-dir`＝ホーム側に残すので影響なし）。

構成（案A）: **`sbatch.sh`（器）** が Slurm 制御・ステージング・コンテナ起動を担い、**`train_val.sh`（中身）** をコンテナ内で実行する。`train_val.sh` は Slurm も Singularity も知らない純粋な学習評価スクリプトで、本環境の `run_sequential_*.sh` をそのまま移植できる。

ファイルの置き場所:
- `train_val.sh` → **repo 内**（例 `experiments/exp_024/train_val.sh`）。コンテナ内の学習評価ロジック＝コードなので **git 同期**（正本＝本環境）。コンテナ内絶対パス `/workspace/kouyou/...` で解決。
- `sbatch.sh` → クラスタ運用スクリプト（`/home/kouyou/...`・`/dataset01`・`.sif` パスなどクラスタ固有値を含む）。repo に置くなら `cluster/` 等へ分離。

**sbatch.sh（器：Slurm 制御＋ステージング＋コンテナ起動）**

```bash
#!/bin/bash
#SBATCH --job-name=exp024_uw
#SBATCH --partition=a6000_ada
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64          # node01-04 は 64 スレッド。128 は不可（≤64）
#SBATCH --time=168:00:00
#SBATCH --output=/home/kouyou/logs/result_%x_%j.txt   # %x=ジョブ名 %j=ジョブID。dir は事前作成
#SBATCH --error=/home/kouyou/logs/error_%x_%j.txt

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

# 実行時に使う変数は #SBATCH ブロックの後（本文）で定義する
HOME_DATA=/home/kouyou/datasets
REPO=/home/kouyou/mmdetection
SIF=/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif
STAGE=/local_cache/${SLURM_JOB_ID}/datasets
METHOD="${1:-fullft_replayfree}"     # train_val.sh へ渡す手法識別（exp_024/025）

# ① rf100 のみ local_cache へステージング（毎バッチ大量に読むため高速SSDへ）。
#    o365 は参照バッファの 1000枚（≈80MB）しか読まないので、46G を毎ジョブコピーせず home から直接 bind する。
mkdir -p "$STAGE"
cp -r "$HOME_DATA/rf100_domain" "$STAGE/"

# ② コンテナを起動し、中で train_val.sh を実行（--bind でパス間接化。config 無変更）
#    o365 は home のツリーを同じコンテナ内パスへ入れ子 bind（STAGE bind の内側にマウント）
singularity exec --nv \
  --bind "$REPO":/workspace/kouyou/mmdetection \
  --bind "$STAGE":/workspace/kouyou/datasets \
  --bind "$HOME_DATA/o365v1_stage":/workspace/kouyou/datasets/o365v1_stage \
  --bind /dataset01/MSCOCO:/workspace/kouyou/datasets/coco2017 \
  "$SIF" bash /workspace/kouyou/mmdetection/experiments/exp_024/train_val.sh "$METHOD"
```

**train_val.sh（中身：コンテナ内の学習評価ロジック。Slurm/Singularity を知らない）**

```bash
#!/bin/bash
set -e
cd /workspace/kouyou/mmdetection
METHOD="${1:-fullft_replayfree}"

# 逐次CL: t=1→t=2→t=3、前ドメイン last を load_from（本環境 run_sequential_*.sh と同一の流れ）。
# 各ドメイン学習後に「現ドメイン＋過去ドメイン＋ZCOCO」を評価。
# config・work_dir は /workspace/... 絶対パスで解決されるため、この中身は本環境とクラスタで共通。
bash experiments/exp_024/run_sequential_${METHOD}.sh
# ↑ 逐次ドライバ（dist_train.sh / dist_test.sh の呼び出し列）は exp_024/025 用に別途用意する
```

- 投入前に `--output`/`--error` の出力先ディレクトリ（例 `/home/kouyou/logs`）を作成しておく（無いと Slurm が書き込めずジョブが失敗する）。`mkdir -p /home/kouyou/logs`。
- ステージング（①）はコンテナの外＝計算ノード上で行う（ホーム NFS とローカルSSD の両方が見える）ため、sbatch.sh 側に置く。`train_val.sh` はコンテナ内で走るので local_cache のステージングには関与しない。
- 逐次CL（t=1→t=2→t=3、前ドメイン last を `load_from`）は `train_val.sh` が呼ぶ `run_sequential_*.sh` に実装する（3ドメインを1ジョブ内で直列実行。168h 制限内）。
- ⚠️ 要確認: `--bind` 先 `/workspace/kouyou/...` はイメージ内に存在しないため、Singularity がマウントポイントを自動生成できる設定（overlay/underlay 有効）か。`/dataset01` の自動バインド実績から有効の見込みだが未確認。無効なら `.sif` 側に空ディレクトリを用意する。
- ZCOCO: eval config（`eval_base_coco.py`）が読むのは `data_root=/workspace/kouyou/datasets/coco2017/` 配下の `annotations/instances_val2017.json` と `val2017/` のみ。クラスタ `/dataset01/MSCOCO` も同じ相対構成（`annotations/instances_val2017.json`＋`val2017/`）を持つため、`--bind /dataset01/MSCOCO:/workspace/kouyou/datasets/coco2017` で config 無変更で解決する。⚠️ 要確認: クラスタ側 `instances_val2017.json` が公式 val（80カテゴリ / 5000画像 / 36781アノテーション）と一致するか、実機で件数を確認。

### 6.1 ステージング方針（どのデータをどこから読むか）

原則: **毎イテレーション大量に反復して読むデータは local_cache（ローカルSSD）へコピー、少量しか読まないデータは home から直接 bind する**。NFS 直読みのボトルネックは「大量・反復読み」にのみ効くため、読む量が小さいデータをわざわざコピーしても手間に見合わない。

| データ | 配置 | 理由 |
|---|---|---|
| rf100（現ドメイン等） | local_cache へコピー | 現ドメインは毎バッチ 4/6 枚・全画像を毎エポック走査。NFS 直読みがボトルネックになる |
| o365（リプレイ参照） | home から入れ子 bind | 実際に読むのは参照バッファの **1000枚（≈80MB）のみ**。初回読込後は計算ノードRAM（512GB〜1TB）のページキャッシュに載り、NFS I/O は実質1回。46G をコピーしても使うのは 80MB で無駄 |
| COCO2017 val（ZCOCO） | `/dataset01` 共有SSD を bind | 既にSSD・全ノード同一パス。ホームにも local_cache にも置かない |

**切替条件**: リプレイのバッファを大きくする設計（数千〜数万枚、または毎回 o365 をサンプリングし直す）に変えた場合は、o365 も local_cache へ（または使う分だけ subset を）ステージングする判断に切り替える。現状の 1000枚固定では不要。

---

## 7. 評価と成果物回収

- 評価も同じ `.sif`＋`--bind` で `tools/dist_test.sh` を実行。ZCOCO は `/dataset01/MSCOCO`（共有SSD・高速）を利用。
- 成果物（`experiments/exp_024/.../*_work_dir`, `eval_*`）はホーム側に残る。本環境へは rsync/scp で回収（§0 の LAN 到達性に依存、⚠️ 要確認）。

---

## 8. 制約とチェックポイント方針

| 項目 | 制限 |
|---|---|
| 同時実行ジョブ | 4 |
| 同時投入ジョブ | 8 |
| バッチ最大実行時間 | 168 時間（7日） |
| インタラクティブ最大 | 2 時間 |
| ホーム容量 | 約 3 TB / 人（§1。公式ドキュメント既定は 2TB） |

- **チェックポイント保持**: 本環境と同様に全エポック保持（1個 1.95 GB）だと手法数×3ドメインで容易にホーム容量を圧迫する。⚠️ 要決定: クラスタでは last＋間引き保持にするか、optimizer state を剥いで約1/3（≈0.65 GB/個）にするか。評価・`load_from` は `state_dict` のみ使用のためパイプラインには影響しない（`--resume` のみ不可）。

---

## 9. セットアップ・チェックリスト（初回）

- [ ] `.sif` を本環境 Dockerfile から作成し、マスターノードへ配置（§2）
- [ ] インタラクティブジョブで `import mmdet/mmcv`・CUDA 可視・`--bind` マウントポイント自動生成を確認（§2, §6）
- [ ] o365 の入れ子 bind が見えるか確認（`ls /workspace/kouyou/datasets/o365v1_stage`）（§6.1）
- [ ] `.gitignore` に学習出力を追加し、コードを push→クラスタで pull（§3）
- [ ] RF100 全ドメイン（11G）と o365 全体（`train/` 46G, zip 除外）をホームへ転送（§4.1）
- [ ] `sbatch.sh`／`train_val.sh`（案A）を exp_024/025 用に用意（§6）
- [ ] ZCOCO の `/dataset01/MSCOCO` バインド先パスを eval config と突き合わせ（§6, §7）
- [ ] 成果物回収経路（rsync/scp or 手動）を確定（§0, §7）
- [ ] チェックポイント保持方針を決定（§8）

---

## 付録: 本環境の確認済み事実（2026-07-24 時点）

- パッケージ: torch 2.1.2+cu121 / mmcv 2.1.0 / mmengine 0.10.3 / mmdet 3.3.0（editable）
- git remote: `github.com/KoyoImai/VLM_OD_CL.git`
- データ量: rf100_domain 11G（3ドメイン計 ≈2.8G）/ o365v1_stage 111G（うち参照は1000枚=80.6 MB）
- config 絶対パス: `/workspace/kouyou/datasets` 33件 / `/workspace/kouyou/mmdetection/experiments` 9件
- クラスタ: パーティション `a6000_ada`（node01-06, A6000 Ada×4, VRAM 48GB/GPU）ほか、`a6000`（node11-13）
