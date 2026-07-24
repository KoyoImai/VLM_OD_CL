#!/usr/bin/env bash
# =============================================================================
# exp_023 ZiRa 逐次ドライバ（replay-free / +replay 共通。第1引数で切替）
#   ZiRa は RDB（HLRB=高lr / LLRB=低lr）を持ち、タスクごとに Rep+ で
#   HLRB を LLRB へ融合して蓄積する（LLRB=過去累積 / HLRB=新タスク）。
#   各ドメイン t で:
#     1. 学習（load_from = 前ドメインの Rep+ 融合済み ckpt。t=1 は config の θ0）。
#     2. 評価: 現在 + 全過去ドメイン + ZCOCO を ZiRa 評価 config（全和 forward）で。
#        評価は融合前の last ckpt で行う（全和 forward は Rep+ 前後で数学的に等価）。
#     3. Rep+: merge_hlrb.py で LLRB へ融合（HLRB→1e-8, s→0.1, LLRB保持）した ckpt を
#        作り、次ドメインの load_from にする。
#   ※ merge_hlrb.py は forward 保存を検証済み。モデル内 rep_merge() のバグ（LLRB 消去）は
#     修正済みだが、本ドライバは checkpoint 版 merge_hlrb.py を用いる。
#   ※ 実行は design.md 承認後（行動原理3）。
# 使い方: bash experiments/exp_023/run_sequential_zira.sh [replayfree|replay]
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # experiments/exp_023/ -> リポジトリルート

VARIANT="${1:-replayfree}"
case "$VARIANT" in
  replayfree|replay) ;;
  *) echo "usage: $0 [replayfree|replay]"; exit 1;;
esac

GPUS=4
CFG=experiments/exp_023/configs
OUT=experiments/exp_023
MERGE=experiments/exp_020/merge_hlrb.py
MERGEDIR="$OUT/zira_${VARIANT}_merged"
mkdir -p "$MERGEDIR"

ORDER=(underwater electromagnetic videogames)
PAST=()
prev_merged=""   # 前ドメインの Rep+ 融合済み ckpt（空なら config の θ0）

for dom in "${ORDER[@]}"; do
  WD="$OUT/zira_${VARIANT}_${dom}_work_dir"
  echo "==================== [ZiRa/$VARIANT] $dom 学習 ===================="
  if [ -z "$prev_merged" ]; then
    bash tools/dist_train.sh "$CFG/zira_${VARIANT}_${dom}.py" "$GPUS" --work-dir "$WD"
  else
    bash tools/dist_train.sh "$CFG/zira_${VARIANT}_${dom}.py" "$GPUS" --work-dir "$WD" \
      --cfg-options load_from="$prev_merged"
  fi
  ckpt="$(cat "$WD/last_checkpoint")"
  echo "[$dom] last ckpt: $ckpt"

  # --- 評価（現在 + 全過去ドメイン + ZCOCO、融合前 last ckpt・全和 forward）---
  for evd in "$dom" "${PAST[@]}"; do
    echo "-------------------- [$dom] 評価 on $evd --------------------"
    bash tools/dist_test.sh "$CFG/zira_eval_${evd}.py" "$ckpt" "$GPUS" \
      --work-dir "$OUT/zira_${VARIANT}_eval_after_${dom}/on_${evd}"
  done
  echo "-------------------- [$dom] ZCOCO --------------------"
  bash tools/dist_test.sh "$CFG/zira_eval_zcoco.py" "$ckpt" "$GPUS" \
    --work-dir "$OUT/zira_${VARIANT}_eval_after_${dom}/zcoco"

  # --- Rep+ 融合（次ドメインの load_from 用）---
  merged="$MERGEDIR/merged_after_${dom}.pth"
  python "$MERGE" "$ckpt" "$merged"
  echo "[$dom] Rep+ 融合: $merged"
  prev_merged="$merged"

  PAST+=("$dom")
done

echo "==================== ZiRa/$VARIANT 逐次 完了 ===================="
