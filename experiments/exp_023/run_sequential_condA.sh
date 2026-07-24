#!/usr/bin/env bash
# =============================================================================
# exp_023 条件A + リプレイ 逐次ドライバ
#   条件A = 特徴抽出+融合のみ学習・下流凍結（paramwise lr_mult。custom detector 不要）。
#   単一の GroundingDINO モデルなので、full FT+replay と同型の逐次（重みのみ継続）:
#     1. 学習（load_from = 前ドメインの last。t=1 は config の θ0）。
#     2. 現ドメイン + 全過去ドメイン + ZCOCO を評価（素の eval_{domain}.py / eval_base_coco.py）。
#     3. この last を次ドメインの load_from にして次へ。
#   ※ 実行は design.md 承認後（行動原理3）。既存プログラムは無変更。
# 使い方: bash experiments/exp_023/run_sequential_condA.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # experiments/exp_023/ -> リポジトリルート

GPUS=4
CFG=experiments/exp_023/configs
OUT=experiments/exp_023
ZCOCO_CFG=configs/mm_grounding_dino/eval_base_coco.py

ORDER=(underwater electromagnetic videogames)
PAST=()
prev_ckpt=""   # 空なら config の load_from(θ0) を使う（t=1）

for dom in "${ORDER[@]}"; do
  WD="$OUT/condA_replay_${dom}_work_dir"
  echo "==================== [条件A/replay] $dom 学習 ===================="
  if [ -z "$prev_ckpt" ]; then
    bash tools/dist_train.sh "$CFG/condA_replay_${dom}.py" "$GPUS" --work-dir "$WD"
  else
    bash tools/dist_train.sh "$CFG/condA_replay_${dom}.py" "$GPUS" --work-dir "$WD" \
      --cfg-options load_from="$prev_ckpt"
  fi
  last="$(cat "$WD/last_checkpoint")"
  echo "[$dom] last ckpt: $last"

  # 現ドメイン + 全過去ドメイン（素の GroundingDINO 評価 config）
  for evd in "$dom" "${PAST[@]}"; do
    echo "-------------------- [$dom] 評価 on $evd --------------------"
    bash tools/dist_test.sh "$CFG/eval_${evd}.py" "$last" "$GPUS" \
      --work-dir "$OUT/condA_replay_eval_after_${dom}/on_${evd}"
  done

  # ZCOCO
  echo "-------------------- [$dom] ZCOCO --------------------"
  bash tools/dist_test.sh "$ZCOCO_CFG" "$last" "$GPUS" \
    --work-dir "$OUT/condA_replay_eval_after_${dom}/zcoco"

  prev_ckpt="$last"
  PAST+=("$dom")
done

echo "==================== 条件A + リプレイ 逐次 完了 ===================="
