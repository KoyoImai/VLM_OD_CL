#!/usr/bin/env bash
# =============================================================================
# exp_034: ZiRa の ODinW-13 逐次学習ドライバ（design.md 承認済み・2026-08-05）
#
#   各タスク t で:
#     1. 学習（2000 iter, batch 2, 1 GPU）。load_from は前タスクの Rep+ 融合済み ckpt。
#        t=1 は config の θ0。
#     2. Rep+: merge_hlrb.py で LLRB へ融合（HLRB→1e-8, s→0.1, LLRB 保持）。
#     3. 評価: 融合後 ckpt で「学習済みタスク（1..t）」と ZCOCO を評価（design.md §5）。
#
#   学習・評価とも 1 GPU（並列実行しない）。使う GPU は環境変数 GPU で指定（既定 0）。
#   途中で止まった場合は同じコマンドで再開できる（完了済みタスクはスキップする）。
#
# 使い方:
#   bash experiments/exp_034/run_sequential_zira_odinw13.sh          # seed 42 の順序
#   GPU=1 bash experiments/exp_034/run_sequential_zira_odinw13.sh 42
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # experiments/exp_034/ -> リポジトリルート

SEED="${1:-42}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

EXP=experiments/exp_034
CFG="$EXP/configs"
MERGE=experiments/exp_020/merge_hlrb.py
MERGEDIR="$EXP/zira_s${SEED}_merged"
EVALDIR="$EXP/eval"
mkdir -p "$MERGEDIR" "$EVALDIR"

# 逐次順（design.md §4: odinw13 config の記載順を random.Random(seed).shuffle）
mapfile -t ORDER < <(python -c "
import sys; sys.path.insert(0, '$EXP')
from odinw_official_tasks import task_order
print('\n'.join(task_order($SEED)))")

echo "==================== exp_034 ZiRa / seed $SEED ===================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "順序: ${ORDER[*]}"

LEARNED=()
prev_merged=""

for i in "${!ORDER[@]}"; do
  t=$((i + 1))
  tt=$(printf '%02d' "$t")
  task="${ORDER[$i]}"
  WD="$EXP/zira_s${SEED}_t${tt}_${task}_work_dir"
  merged="$MERGEDIR/merged_after_t${tt}_${task}.pth"

  echo "==================== [t=$t/13] $task 学習 ===================="
  if [ -f "$merged" ]; then
    echo "[t=$t] $merged が既にあるため学習と融合をスキップ"
  else
    if [ -z "$prev_merged" ]; then
      python tools/train.py "$CFG/zira_odinw13_${task}.py" --work-dir "$WD"
    else
      python tools/train.py "$CFG/zira_odinw13_${task}.py" --work-dir "$WD" \
        --cfg-options load_from="$prev_merged"
    fi
    ckpt="$(cat "$WD/last_checkpoint")"
    echo "[t=$t] last ckpt: $ckpt"
    python "$MERGE" "$ckpt" "$merged"
  fi
  prev_merged="$merged"
  LEARNED+=("$task")

  # --- 評価: 学習済みタスク（1..t）---
  subset="$CFG/_eval_after_t${tt}.py"
  python "$EXP/make_eval_subset.py" "$subset" "${LEARNED[@]}"
  echo "-------------------- [t=$t] ODinW 評価（${#LEARNED[@]} タスク） --------------------"
  python tools/test.py "$subset" "$merged" \
    --work-dir "$EVALDIR/odinw_after_t${tt}"

  # --- 評価: ZCOCO ---
  echo "-------------------- [t=$t] ZCOCO 評価 --------------------"
  python tools/test.py "$CFG/zcoco_eval.py" "$merged" \
    --work-dir "$EVALDIR/zcoco_after_t${tt}"
done

echo "==================== exp_034 ZiRa / seed $SEED 完了 ===================="
