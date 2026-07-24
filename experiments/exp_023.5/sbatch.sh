#!/bin/bash
# =============================================================================
# exp_023.5 sbatch.sh — クラスタ実行の「器」（案A）。
#   #SBATCH（Slurm）＋ rf100 の local_cache ステージング＋ Singularity 起動を担い、
#   コンテナ内で train_val.sh を実行する。学習/評価ロジックは train_val.sh 側。
#
#   使い方（マスターノードで）:
#     sbatch experiments/exp_023.5/sbatch.sh                    # all（ゼロショット→1ep学習→評価）
#     RUN_STAGE=zeroshot sbatch experiments/exp_023.5/sbatch.sh # ゼロショットのみ（最速）
#     RUN_STAGE=train    sbatch experiments/exp_023.5/sbatch.sh # 1ep学習のみ
#
#   事前準備:
#     - mkdir -p /home/kouyou/logs
#     - θ0 を /home/kouyou/ckpt/ に配置（下記 URL を wget、README §2 参照。計算ノードはオフライン想定）
#         https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/
#           grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/
#           grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
#     - データ転送済み（README §4: rf100_domain, o365v1_stage）
#     - コード同期済み（README §3: git pull）
# =============================================================================
#SBATCH --job-name=exp0235_debug
#SBATCH --partition=a6000_ada
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64            # node01-04 は 64 スレッド。≤64
#SBATCH --time=04:00:00              # デバッグ（ゼロショット＋1ep＋評価）。フル20epは行わない
#SBATCH --output=/home/kouyou/logs/result_%x_%j.txt
#SBATCH --error=/home/kouyou/logs/error_%x_%j.txt

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

HOME_DATA=/home/kouyou/datasets
REPO=/home/kouyou/VLM_OD_CL          # クラスタ上の repo クローン先（git remote 名 = VLM_OD_CL）
SIF=/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif
CACHE=/local_cache/${SLURM_JOB_ID}/datasets
RUN_STAGE="${RUN_STAGE:-all}"        # all|zeroshot|train|eval を train_val.sh へ渡す

# ① rf100 のみ local_cache へステージング（o365 は home から bind、§6.1 の方針）
mkdir -p "$CACHE"
cp -r "$HOME_DATA/rf100_domain" "$CACHE/"

# ② コンテナ起動 → train_val.sh 実行（--bind でパス間接化。config 無変更）
singularity exec --nv \
  --bind "$REPO":/workspace/kouyou/mmdetection \
  --bind "$CACHE":/workspace/kouyou/datasets \
  --bind "$HOME_DATA/o365v1_stage":/workspace/kouyou/datasets/o365v1_stage \
  --bind "$HOME_DATA/bert-base-uncased":/workspace/kouyou/datasets/bert-base-uncased \
  --bind /dataset01/MSCOCO:/workspace/kouyou/datasets/coco2017 \
  --bind /home/kouyou/ckpt:/workspace/kouyou/ckpt \
  "$SIF" bash /workspace/kouyou/mmdetection/experiments/exp_023.5/train_val.sh "$RUN_STAGE"
