#!/usr/bin/env bash
# =============================================================================
# exp_042 本走: EWC の ODinW-13 13 タスク逐次学習（design.md §2.5）
#
#   各タスク t で:
#     1. 学習（3000 iter）。load_from は θ_{t-1}（t=1 は config の θ0）。
#        --cfg-options で λ と EWC 状態（2 バッファ）を渡す。t=1 は状態無し。
#     2. Fisher 推定 → 状態更新（projects/ewc_cl/estimate_fisher.py、上限 1,000 枚）。
#     3. 評価: 学習済みタスク 1..t ＋ ZCOCO（exp_038 の plain 評価 config を流用。
#        EWC の ckpt は素の GroundingDINO 構造なのでそのまま読める）。
#
#   λ は 3 水準 {100, 1000, 10000} を別 GPU で並行して走らせる（design.md §2.4-2.5、
#   2026-08-15 変更）。出力（work_dir・状態・評価）は λ ごとに分離する。
#   途中で止まっても同じコマンドで再開できる（状態ファイルがあるタスクはスキップ）。
#
# 使い方（3 本並行）:
#   GPU=0 LAM=100   bash experiments/exp_042/run_main_ewc.sh
#   GPU=1 LAM=1000  bash experiments/exp_042/run_main_ewc.sh
#   GPU=2 LAM=10000 bash experiments/exp_042/run_main_ewc.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
LAM="${LAM:?λ を LAM= で指定（100 / 1000 / 10000）}"

EXP=experiments/exp_042
CFG="$EXP/configs"
E38="experiments/exp_038/configs"
STATEDIR="$EXP/ewc_lam${LAM}_states"
EVALDIR="$EXP/eval_ewc_lam${LAM}"
mkdir -p "$STATEDIR" "$EVALDIR"

mapfile -t ORDER < <(python -c "
import sys; sys.path.insert(0, 'experiments/exp_034')
from odinw_official_tasks import task_order
print('\n'.join(task_order(42)))")

echo "==================== exp_042 EWC 本走 λ=$LAM ===================="

prev=""        # θ_{t-1}（空なら config の θ0）
prev_state=""  # 2 バッファ（空なら無し = ペナルティ不活性）

for i in "${!ORDER[@]}"; do
  t=$((i + 1)); tt=$(printf '%02d' "$t"); task="${ORDER[$i]}"
  WD="$EXP/ewc_lam${LAM}_t${tt}_${task}_work_dir"
  ckpt="$WD/iter_3000.pth"
  state="$STATEDIR/state_t${tt}.pth"

  echo "==================== [t=$t/13] $task 学習 ===================="
  if [ -f "$state" ]; then
    echo "[t=$t] $state があるため学習と Fisher をスキップ"
  else
    OPTS=("model.ewc.lam=$LAM")
    [ -n "$prev" ] && OPTS+=("load_from=$prev")
    [ -n "$prev_state" ] && OPTS+=("model.ewc.state_path=$prev_state")
    python tools/train.py "$CFG/ewc_${task}.py" --work-dir "$WD" \
      --cfg-options "${OPTS[@]}"
    FOPTS=(--out "$state" --max-samples 1000 --seed 0 --task-name "$task")
    [ -n "$prev_state" ] && FOPTS+=(--prev-state "$prev_state")
    python projects/ewc_cl/estimate_fisher.py "$CFG/ewc_${task}.py" "$ckpt" \
      "${FOPTS[@]}"
  fi
  prev="$ckpt"; prev_state="$state"

  echo "-------------------- [t=$t] ODinW 評価（$t タスク）--------------------"
  [ -d "$EVALDIR/odinw_after_t${tt}" ] || \
    python tools/test.py "$E38/_eval_after_t${tt}.py" "$prev" \
      --work-dir "$EVALDIR/odinw_after_t${tt}"
  echo "-------------------- [t=$t] ZCOCO 評価 --------------------"
  [ -d "$EVALDIR/zcoco_after_t${tt}" ] || \
    python tools/test.py "$E38/zcoco_eval.py" "$prev" \
      --work-dir "$EVALDIR/zcoco_after_t${tt}"
done

echo "==================== exp_042 EWC 本走 完了（最終: $prev）===================="
