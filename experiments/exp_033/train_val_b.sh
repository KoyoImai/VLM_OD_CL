#!/bin/bash
# =============================================================================
# exp_033_b train_val.sh — 条件A + L2 蒸留 + リプレイ / **λ = 2.0 固定**
#   note07 の分割に対応: 蒸留D・蒸留E
#   段階（$1 / 既定 all）: D | E | all
# =============================================================================
set -euo pipefail
cd /workspace/kouyou/mmdetection

export HF_HOME=/tmp/hf
export TRANSFORMERS_CACHE=/tmp/hf/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p /tmp/hf/hub

export PORT="${PORT:-29770}"
export THETA0=/workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth

run () { bash experiments/exp_028/run_sequential_kd.sh exp_033 condA l2w20 "$1"; }

STAGE="${1:-all}"
case "$STAGE" in
  D) run D ;;
  E) run E ;;
  all) for k in D E; do run "$k"; done ;;
  *) echo "usage: $0 [D|E|all]"; exit 1 ;;
esac

echo "==== exp_033_b train_val ($STAGE) 完了 ===="
