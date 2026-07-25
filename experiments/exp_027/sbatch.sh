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
#
#   ★ node03 の除外（2026-07-25）: ジョブ 20510・20521 が node03 に割り当てられ、
#     「mkdir: cannot create directory '/local_cache': Permission denied」で即失敗した。
#     /local_cache/<ジョブID> は Slurm の prolog が作るが、node03 ではそれが用意されない。
#     node04/05/06 では同一スクリプトが正常動作している（exp_023.5・exp_024/025/026 の実績）。
#     node03 が復旧したら下の --exclude 行を削除すること。
#     詳細: experiments/exp_027/troubleshooting_local_cache.md
# =============================================================================
#SBATCH --job-name=exp027_replay
#SBATCH --partition=a6000_ada
#SBATCH --exclude=node03              # node03 は /local_cache が無く mkdir で落ちる（2026-07-25）
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

# ① rf100 のみ local_cache へステージング（毎バッチ大量に読むため高速SSDへ）。
#    o365 は参照バッファ 1,000 枚しか読まないのでホームから直接 bind する。
mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"

echo "== local_cache staging =="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}   CACHE = $CACHE   RUN_STAGE = $RUN_STAGE"
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
