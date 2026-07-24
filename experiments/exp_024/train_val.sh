#!/bin/bash
# =============================================================================
# exp_024 train_val.sh — コンテナ内で走る学習評価ロジック（案A の「中身」）。
#   Slurm も Singularity も知らない。sbatch.sh から singularity exec 経由で呼ばれる。
#   逐次CL（underwater -> electromagnetic -> videogames）を1ジョブ内で直列実行し、
#   各ドメイン後に「現ドメイン＋過去ドメイン＋ZCOCO」を評価する。
#   config・work_dir は /workspace/... 絶対パスで解決されるため、本環境とクラスタで共通。
# =============================================================================
set -euo pipefail
cd /workspace/kouyou/mmdetection

# HuggingFace/transformers を書込可キャッシュに向ける（.sif 焼込みの
# HF_HOME=/workspace/kouyou/datasets/HuggingFace は書込不可。BERT はローカルパスから読む）
export HF_HOME=/tmp/hf
export TRANSFORMERS_CACHE=/tmp/hf/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p /tmp/hf/hub

# 分散ポート（dist_* の既定 29500 と衝突回避）
export PORT="${PORT:-29540}"

# θ0（事前学習重み）を明示パスで渡す。ホスト /home/kouyou/ckpt を /workspace/kouyou/ckpt へ
# bind 済み（sbatch.sh）。URL 可用性・キャッシュ状態に依存させないため（exp_023.5 と同方針）。
export THETA0=/workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth

echo "==== [exp_024] リプレイ無し逐次FT（条件A＝全モジュール）開始 ===="
bash experiments/exp_024/run_sequential_fullft_replayfree.sh
echo "==== [exp_024] 完了 ===="
