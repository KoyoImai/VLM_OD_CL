#!/bin/bash
# =============================================================================
# exp_027 sbatch.sh — クラスタ実行の「器」（案A）。
#   #SBATCH（Slurm）＋ rf100 の local_cache ステージング＋ Singularity 起動を担い、
#   コンテナ内で train_val.sh を実行する。学習/評価ロジックは train_val.sh 側。
#
#   使い方（マスターノードで）:
#     RUN_STAGE=debug  sbatch experiments/exp_027/sbatch.sh   # ★最初に必ずこれ（数分〜十数分）
#     RUN_STAGE=fullft sbatch experiments/exp_027/sbatch.sh   # 027-1 条件A + リプレイ（約43h）
#     RUN_STAGE=condB  sbatch experiments/exp_027/sbatch.sh   # 027-2 条件B + リプレイ（約43h）
#     sbatch experiments/exp_027/sbatch.sh                     # all（debug→fullft→condB・長時間）
#
#   事前準備（README_cluster.md 参照）:
#     - mkdir -p /home/kouyou/logs
#     - θ0 を /home/kouyou/ckpt/ に配置
#     - データ転送済み（§4: rf100_domain, o365v1_stage, bert-base-uncased）
#     - コード同期済み（§3: git pull。buffer/*.odvg.json も git 管理下）
#   ※ 実行は design.md の着手条件を満たしてから（行動原理3）。
#
#   ★ exp_024/025 との違い: 本実験はリプレイを使うため **Objects365 の bind が必要**。
#     読み込むのは参照バッファの 1,000 枚のみで回数が少ないため、local_cache には置かず
#     ホームから直接 bind する（2026-07-24 ユーザー決定・exp_023.5 と同方針）。
#     転送漏れは RUN_STAGE=debug で検出する。
# =============================================================================
#SBATCH --job-name=exp027_replay
#SBATCH --partition=a6000_ada
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64            # node01-04 は 64 スレッド。≤64
#SBATCH --time=120:00:00              # 段階分割前提。上限168hに対し余裕を確保
#SBATCH --output=/home/kouyou/logs/result_%x_%j.txt
#SBATCH --error=/home/kouyou/logs/error_%x_%j.txt

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

HOME_DATA=/home/kouyou/datasets
REPO=/home/kouyou/VLM_OD_CL          # クラスタ上の repo クローン先
SIF=/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif
CACHE=/local_cache/${SLURM_JOB_ID}/datasets
RUN_STAGE="${RUN_STAGE:-all}"

# ⓪ 実行環境の記録（失敗しても必ずログに残るよう、他の処理より先に出す）
#    2026-07-25 に「mkdir: cannot create directory '/local_cache': Permission denied」で
#    落ちた際、mkdir が最初の処理だったためジョブIDもノード名もログに残らなかった。
echo "========== exp_027 job info =========="
echo "hostname        = $(hostname)"
echo "SLURM_JOB_ID    = ${SLURM_JOB_ID:-<未設定>}"
echo "SLURM_NODELIST  = ${SLURM_JOB_NODELIST:-<未設定>}"
echo "RUN_STAGE       = $RUN_STAGE"
echo "date            = $(date '+%F %T')"
ls -ld /local_cache 2>&1 | sed 's/^/local_cache    : /'
echo "======================================"

# ⓪-2 local_cache の可用性チェック
#    /local_cache/${SLURM_JOB_ID} は「割り当てられた計算ノード上に Slurm が自動作成する」
#    一時ディレクトリで、ジョブ終了時に消える（doc_cluster_manual/pages/Usage_MountLocalCache.md）。
#    存在しない場合、mkdir は / への書き込みを試みて Permission denied になる。
#    local_cache を使わない迂回は行わない（2026-07-25 ユーザー指示。指定がない限り禁止）。
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: SLURM_JOB_ID が未設定です。sbatch 経由で実行してください:"
  echo "         RUN_STAGE=debug sbatch experiments/exp_027/sbatch.sh"
  exit 1
fi
if [ ! -d /local_cache ]; then
  echo "ERROR: このノード（$(hostname)）に /local_cache がありません。"
  echo "       上の job info ブロックのノード名を確認し、ノード固有の問題かを切り分けること。"
  exit 1
fi

# ① rf100 を local_cache へステージング（毎バッチ大量に読むため高速SSDへ）。
#    o365 は参照バッファ 1,000 枚しか読まないのでホームから直接 bind する。
mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"

echo "== local_cache staging =="
echo "CACHE = $CACHE"
df -h /local_cache 2>/dev/null | tail -1
du -sh "$CACHE"/* 2>/dev/null
echo "=========================="

# ② コンテナ起動 → train_val.sh 実行（--bind でパス間接化。config 無変更）
#    各データソースを /workspace/kouyou/datasets 直下の独立した葉として bind する
#    （親 datasets を丸ごと bind して入れ子にしない＝exp_023.5 で実証済みの構造）。
singularity exec --nv \
  --bind "$REPO":/workspace/kouyou/mmdetection \
  --bind "$CACHE/rf100_domain":/workspace/kouyou/datasets/rf100_domain \
  --bind "$HOME_DATA/o365v1_stage":/workspace/kouyou/datasets/o365v1_stage \
  --bind "$HOME_DATA/bert-base-uncased":/workspace/kouyou/datasets/bert-base-uncased \
  --bind /dataset01/MSCOCO:/workspace/kouyou/datasets/coco2017 \
  --bind /home/kouyou/ckpt:/workspace/kouyou/ckpt \
  "$SIF" bash /workspace/kouyou/mmdetection/experiments/exp_027/train_val.sh "$RUN_STAGE"
