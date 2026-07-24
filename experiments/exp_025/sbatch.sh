#!/bin/bash
# =============================================================================
# exp_025 sbatch.sh — クラスタ実行の「器」（案A）。
#   #SBATCH（Slurm）＋ rf100 の local_cache ステージング＋ Singularity 起動を担い、
#   コンテナ内で train_val.sh を実行する。学習/評価ロジックは train_val.sh 側。
#
#   使い方（マスターノードで）:
#     sbatch experiments/exp_025/sbatch.sh
#
#   事前準備（README_cluster.md 参照）:
#     - mkdir -p /home/kouyou/logs
#     - θ0 を /home/kouyou/ckpt/ に配置（t=1 の load_from。ドライバは config の θ0 を使う）
#     - データ転送済み（§4: rf100_domain, bert-base-uncased）
#     - コード同期済み（§3: git pull）
#   ※ 実行は design.md 承認かつ exp_023.5 完了後（着手条件）。
#
#   時間見積り: 1ドメイン約20時間（20epoch）×3ドメイン ≒ 60時間 ＋ 評価。
#   バッチ上限 168 時間には収まる想定だが、余裕を持って 120 時間を確保する。
# =============================================================================
#SBATCH --job-name=exp025_ftB_replayfree
#SBATCH --partition=a6000_ada
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64            # node01-04 は 64 スレッド。≤64
#SBATCH --time=120:00:00              # 3ドメイン逐次＋評価。上限168hに対し余裕を確保
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

# ① rf100 のみ local_cache へステージング（毎バッチ大量に読むため高速SSDへ）。
#    リプレイ無しのため o365 は不要だが、bert/coco はホーム・共有SSDから直接 bind する。
mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"

echo "== local_cache staging =="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}   CACHE = $CACHE"
df -h /local_cache 2>/dev/null | tail -1
du -sh "$CACHE"/* 2>/dev/null
echo "=========================="

# ② コンテナ起動 → train_val.sh 実行（--bind でパス間接化。config 無変更）
#    各データソースを /workspace/kouyou/datasets 直下の独立した葉として bind する
#    （親 datasets を丸ごと bind して入れ子にしない＝exp_023.5 で実証済みの構造）。
#    rf100 は local_cache から、bert/coco はホーム・共有SSD から直接。
singularity exec --nv \
  --bind "$REPO":/workspace/kouyou/mmdetection \
  --bind "$CACHE/rf100_domain":/workspace/kouyou/datasets/rf100_domain \
  --bind "$HOME_DATA/bert-base-uncased":/workspace/kouyou/datasets/bert-base-uncased \
  --bind /dataset01/MSCOCO:/workspace/kouyou/datasets/coco2017 \
  --bind /home/kouyou/ckpt:/workspace/kouyou/ckpt \
  "$SIF" bash /workspace/kouyou/mmdetection/experiments/exp_025/train_val.sh
