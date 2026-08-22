#!/bin/bash
# =============================================================================
# exp_050 DGS — クラスタ実行の「器」（design.md §5）。
#   RF100 6 ドメイン逐次・バッファ不使用（学習 約 28〜40 h ＋ 特徴抽出 約 1.5 h ＋ 評価）。
#
#   使い方（マスターノードで）:
#     sbatch experiments/exp_050/sbatch_dgs.sh
#   別のクローンから走らせる場合のみ REPO=<パス> を付ける。
#
#   ★ ZCOCO に MSCOCO、θ0 に /home/kouyou/ckpt の bind が必要。Objects365 は不要。
#   ★ node03 は /local_cache が用意されず mkdir で落ちるため除外する。
#   ★ DTG で既存グループへの併合が出た場合、ドライバは exit 2 で停止する（design §2.3）。
#   ※ 実行は design.md の承認後（行動原理3。2026-08-22 承認済み）。
# =============================================================================
#SBATCH --job-name=exp050_dgs
#SBATCH --partition=a6000_ada
#SBATCH --exclude=node03
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=96:00:00
#SBATCH --output=/home/kouyou/logs/result_%x_%j.txt
#SBATCH --error=/home/kouyou/logs/error_%x_%j.txt

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

HOME_DATA=/home/kouyou/datasets
# 作業ディレクトリは exp_050 専用クローン
REPO="${REPO:-/home/kouyou/VLM_OD_CL_exp050}"
SIF=/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif
CACHE=/local_cache/${SLURM_JOB_ID}/datasets

mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"

echo "== local_cache staging =="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}   CACHE = $CACHE   exp_050 dgs"
df -h /local_cache 2>/dev/null | tail -1
du -sh "$CACHE"/* 2>/dev/null
echo "=========================="

singularity exec --nv \
  --bind "$REPO":/workspace/kouyou/mmdetection \
  --bind "$CACHE/rf100_domain":/workspace/kouyou/datasets/rf100_domain \
  --bind "$HOME_DATA/bert-base-uncased":/workspace/kouyou/datasets/bert-base-uncased \
  --bind /dataset01/MSCOCO:/workspace/kouyou/datasets/coco2017 \
  --bind /home/kouyou/ckpt:/workspace/kouyou/ckpt \
  "$SIF" bash -c "GPUS=4 bash /workspace/kouyou/mmdetection/experiments/exp_050/run_dgs_rf100.sh"
