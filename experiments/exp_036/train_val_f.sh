#!/bin/bash
# =============================================================================
# exp_036_f train_val.sh — 条件A + L2 蒸留 + リプレイ / **λ = 5.0 固定**
#   note07 の分割に対応: 蒸留D（融合後）+ 蒸留E（3項平均）
#   段階（$1 / 既定 all）: D|E | all
# =============================================================================
set -euo pipefail
cd /workspace/kouyou/mmdetection

export HF_HOME=/tmp/hf
export TRANSFORMERS_CACHE=/tmp/hf/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p /tmp/hf/hub

export PORT="${PORT:-29890}"
export THETA0=/workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth

# ★ λ はここで明示する。ドライバ run_sequential_kd.sh は
#   lam="${LAMBDA:-1.0}" を --cfg-options model.kd.loss_weight に渡すため、
#   LAMBDA を設定しないと config の loss_weight=5.0 が 1.0 で上書きされる。
#   （exp_033 はこれが原因で全12条件が実効 λ=1.0 で学習されていた。2026-08-06 検出）
export LAMBDA=5.0

# 起動時に config と LAMBDA の一致を確認する（食い違ったら学習を始めない）
python - <<'PY'
import os
from mmengine.config import Config
w = Config.fromfile(
    'experiments/exp_036/configs/kdD_condA_l2w50_underwater.py').model.kd.loss_weight
lam = float(os.environ['LAMBDA'])
assert w == lam, f'config の loss_weight={w} と LAMBDA={lam} が食い違っています'
print(f'[exp_036_f] 実効 λ = {lam}（config と一致）')
PY

run () { bash experiments/exp_028/run_sequential_kd.sh exp_036 condA l2w50 "$1"; }

STAGE="${1:-all}"
case "$STAGE" in
  D) run D ;;
  E) run E ;;
  all) for k in D E; do run "$k"; done ;;
  *) echo "usage: $0 [D|E|all]"; exit 1 ;;
esac

echo "==== exp_036_f train_val ($STAGE) 完了 ===="
