#!/bin/bash
# =============================================================================
# exp_026 sbatch.sh — クラスタ実行の「器」（案A）。
#   #SBATCH（Slurm）＋ rf100 の local_cache ステージング＋ Singularity 起動を担い、
#   コンテナ内で train_val.sh を実行する。
#
#   使い方（マスターノードで）:
#     RUN_STAGE=zeroshot    sbatch experiments/exp_026/sbatch.sh   # 最短（学習なし・数十分）
#     RUN_STAGE=indiv_condA sbatch experiments/exp_026/sbatch.sh
#     RUN_STAGE=indiv_condB sbatch experiments/exp_026/sbatch.sh
#     RUN_STAGE=indiv_zira  sbatch experiments/exp_026/sbatch.sh   # 論文準拠2000iter・短い
#     RUN_STAGE=oracle      sbatch experiments/exp_026/sbatch.sh
#     sbatch experiments/exp_026/sbatch.sh                          # all（長時間・非推奨）
#
#   段階を分けて投入する前提で設計している。exp_024/025 が長時間ジョブで2枠を占めるため、
#   同時実行4ジョブの残り枠に短い段階から流す（design.md 冒頭「環境の役割分担」）。
#
#   ※ ゼロショットは RF100 **6ドメイン全て** を評価するため、rf100_domain は
#     3ドメインだけでなく全ドメインが転送済みである必要がある（README_cluster §4）。
#   ※ 実行は design.md 承認かつ exp_023.5 完了後（着手条件）。
# =============================================================================
#SBATCH --job-name=exp026_refpoints
#SBATCH --partition=a6000_ada
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64            # node01-04 は 64 スレッド。≤64
#SBATCH --time=120:00:00              # 段階分割前提。all で流す場合は上限168hに注意
#SBATCH --output=/home/kouyou/logs/result_%x_%j.txt
#SBATCH --error=/home/kouyou/logs/error_%x_%j.txt

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

HOME_DATA=/home/kouyou/datasets
REPO=/home/kouyou/VLM_OD_CL
SIF=/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif
CACHE=/local_cache/${SLURM_JOB_ID}/datasets
RUN_STAGE="${RUN_STAGE:-all}"

# ① rf100 を local_cache へステージング（ゼロショットは6ドメイン評価のため全ドメイン必要）
mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"

echo "== local_cache staging =="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}   CACHE = $CACHE   RUN_STAGE = $RUN_STAGE"
df -h /local_cache 2>/dev/null | tail -1
du -sh "$CACHE"/* 2>/dev/null
echo "=========================="

# ② コンテナ起動 → train_val.sh 実行（--bind でパス間接化。config 無変更）
#    各データソースを /workspace/kouyou/datasets 直下の独立した葉として bind
#    （exp_023.5 で実証済みの構造）。リプレイ無しのため o365 は不要。
singularity exec --nv \
  --bind "$REPO":/workspace/kouyou/mmdetection \
  --bind "$CACHE/rf100_domain":/workspace/kouyou/datasets/rf100_domain \
  --bind "$HOME_DATA/bert-base-uncased":/workspace/kouyou/datasets/bert-base-uncased \
  --bind /dataset01/MSCOCO:/workspace/kouyou/datasets/coco2017 \
  --bind /home/kouyou/ckpt:/workspace/kouyou/ckpt \
  "$SIF" bash /workspace/kouyou/mmdetection/experiments/exp_026/train_val.sh "$RUN_STAGE"
