#!/bin/bash
# =============================================================================
# exp_027 train_val.sh — コンテナ内で走る学習評価ロジック（案A の「中身」）。
#   Slurm も Singularity も知らない。sbatch.sh から singularity exec 経由で呼ばれる。
#
#   段階（$1 / 既定 all）:
#     debug   Objects365 のリプレイ経路の確認のみ（学習なし・数分〜十数分）
#     fullft  027-1 逐次FT + リプレイ（条件A＝全モジュール）3ドメイン
#     condB   027-2 逐次FT + リプレイ（条件B＝特徴抽出+融合）3ドメイン
#     all     debug を通してから fullft → condB（長時間・段階分割を推奨）
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

# θ0（事前学習重み）を明示パスで渡す。ホスト /home/kouyou/ckpt を /workspace/kouyou/ckpt へ
# bind 済み（sbatch.sh）。URL 可用性・キャッシュ状態に依存させないため（exp_023.5 と同方針）。
export THETA0=/workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth

STAGE="${1:-all}"

run_debug () {
  echo "==== [exp_027] デバッグ実行（Objects365 リプレイ経路の確認） ===="
  python experiments/exp_027/debug_o365.py
  echo "==== [exp_027] デバッグ実行 完了 ===="
}

case "$STAGE" in
  debug)  run_debug ;;
  fullft)
    echo "==== [exp_027-1] 逐次FT + リプレイ（条件A＝全モジュール）開始 ===="
    bash experiments/exp_027/run_sequential_fullft_replay.sh
    ;;
  condB)
    echo "==== [exp_027-2] 逐次FT + リプレイ（条件B＝特徴抽出+融合）開始 ===="
    bash experiments/exp_027/run_sequential_condB_replay.sh
    ;;
  all)
    run_debug                                                # 先に経路を確認してから
    bash experiments/exp_027/run_sequential_fullft_replay.sh
    bash experiments/exp_027/run_sequential_condB_replay.sh
    ;;
  *) echo "usage: $0 [debug|fullft|condB|all]"; exit 1 ;;
esac

echo "==== exp_027 train_val.sh ($STAGE) 完了 ===="
