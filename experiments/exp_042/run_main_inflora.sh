#!/usr/bin/env bash
# =============================================================================
# exp_042 本走: InfLoRA の ODinW-13 13 タスク逐次学習（design.md §2.5）
#
#   各タスク t で（公式実装の手続き。projects/lora_cl/README.md）:
#     1. 設計: θ_{t-1} で入力共分散を収集し、DualGPM メモリで射影して A を設計・固定
#        （inflora_prepare.py。t=1 はメモリ無し = 生の共分散の SVD）。
#     2. 学習（3000 iter、B のみ）。load_from は θ_{t-1}、design_path を渡す。
#     3. メモリ更新: 学習済み ckpt で共分散を再収集し DualGPM を更新
#        （inflora_update_memory.py。閾値 lamb=0.95 → lame=1.0 線形、公式 DomainNet 設定）。
#     4. マージ: θ_t = θ_{t-1} + B·A（merge_lora.py。alpha=r → scaling=1）。
#     5. 評価: θ_t で学習済みタスク 1..t ＋ ZCOCO（exp_038 の plain 評価 config）。
#
#   途中で止まっても同じコマンドで再開できる（θ_t があるタスクはスキップ）。
#
# 使い方:
#   bash experiments/exp_042/run_main_inflora.sh
#   GPU=1 bash experiments/exp_042/run_main_inflora.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

EXP=experiments/exp_042
CFG="$EXP/configs"
E38="experiments/exp_038/configs"
DESIGNDIR="$EXP/inflora_designs"
MEMDIR="$EXP/inflora_memory"
THETADIR="$EXP/inflora_theta"
EVALDIR="$EXP/eval_inflora"
THETA0='https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
LAMB=0.95; LAME=1.0; TOTAL=13
mkdir -p "$DESIGNDIR" "$MEMDIR" "$THETADIR" "$EVALDIR"

mapfile -t ORDER < <(python -c "
import sys; sys.path.insert(0, 'experiments/exp_034')
from odinw_official_tasks import task_order
print('\n'.join(task_order(42)))")

echo "==================== exp_042 InfLoRA 本走 ===================="

prev="$THETA0"   # θ_{t-1}（t=1 は θ0）
prev_mem=""      # DualGPM メモリ（t=1 は無し）

for i in "${!ORDER[@]}"; do
  t=$((i + 1)); tt=$(printf '%02d' "$t"); task="${ORDER[$i]}"
  WD="$EXP/inflora_t${tt}_${task}_work_dir"
  design="$DESIGNDIR/design_t${tt}.pth"
  mem="$MEMDIR/memory_t${tt}.pth"
  theta="$THETADIR/theta_t${tt}_${task}.pth"

  echo "==================== [t=$t/13] $task ===================="
  if [ -f "$theta" ]; then
    echo "[t=$t] $theta があるためスキップ"
  else
    POPT=(--out "$design" --max-samples 1000 --seed 0 --task-name "$task")
    [ -n "$prev_mem" ] && POPT+=(--memory "$prev_mem")
    python projects/lora_cl/inflora_prepare.py "$CFG/inflora_${task}.py" \
      "$prev" "${POPT[@]}"

    TOPT=("model.inflora.design_path=$design")
    [ "$t" -gt 1 ] && TOPT+=("load_from=$prev")
    python tools/train.py "$CFG/inflora_${task}.py" --work-dir "$WD" \
      --cfg-options "${TOPT[@]}"
    ckpt="$WD/iter_3000.pth"

    MOPT=(--out "$mem" --task-index "$i" --total "$TOTAL"
          --lamb "$LAMB" --lame "$LAME" --max-samples 1000 --seed 0
          --task-name "$task")
    [ -n "$prev_mem" ] && MOPT+=(--memory "$prev_mem")
    python projects/lora_cl/inflora_update_memory.py "$CFG/inflora_${task}.py" \
      "$ckpt" "${MOPT[@]}"

    python projects/lora_cl/merge_lora.py "$ckpt" "$theta"
  fi
  prev="$theta"; prev_mem="$mem"

  echo "-------------------- [t=$t] ODinW 評価（$t タスク）--------------------"
  [ -d "$EVALDIR/odinw_after_t${tt}" ] || \
    python tools/test.py "$E38/_eval_after_t${tt}.py" "$prev" \
      --work-dir "$EVALDIR/odinw_after_t${tt}"
  echo "-------------------- [t=$t] ZCOCO 評価 --------------------"
  [ -d "$EVALDIR/zcoco_after_t${tt}" ] || \
    python tools/test.py "$E38/zcoco_eval.py" "$prev" \
      --work-dir "$EVALDIR/zcoco_after_t${tt}"
done

echo "==================== exp_042 InfLoRA 本走 完了（最終: $prev）===================="
