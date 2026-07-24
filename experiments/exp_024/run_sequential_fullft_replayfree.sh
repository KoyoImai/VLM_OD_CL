#!/usr/bin/env bash
# =============================================================================
# exp_024 逐次ドライバ: リプレイ無し逐次ファインチューニング（条件A＝全モジュール）
#   逐次順 underwater -> electromagnetic -> videogames を順に学習・評価する。
#   各ドメイン t で:
#     1. 前ドメインの last ckpt から load_from（t=1 は config の load_from=θ0）
#        ※ load_from は重みのみ読む。LR スケジュール・optimizer 状態はリセットされる。
#     2. dist_train（4GPU）で学習（リプレイ無し・現在ドメインのみ・batch_size=4/GPU）
#     3. last ckpt を特定（適応 mAP は last を採用）
#     4. 現ドメイン + それまでの全過去ドメイン + ZCOCO を評価
#     5. この last を次ドメインの load_from にして次へ
#
#   評価 config は exp_023 のものを流用（無変更。design.md §7）。
#   ※ 実行は design.md 承認かつ exp_023.5 完了後（行動原理3・着手条件）。既存ファイルは無変更。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # experiments/exp_024/ -> リポジトリルート

GPUS=4
export PORT="${PORT:-29540}"          # 他ジョブ（学習29500 / per-epoch評価29520）と衝突回避
CFG=experiments/exp_024/configs
EVALCFG=experiments/exp_023/configs   # eval_{domain}.py を流用
OUT=experiments/exp_024
ZCOCO_CFG=configs/mm_grounding_dino/eval_base_coco.py

ORDER=(underwater electromagnetic videogames)
PAST=()
prev_ckpt=""   # 空なら t=1（θ0 から開始）

# θ0 の与え方（exp_023.5 design の方針に合わせる）:
#   THETA0 が設定されていれば、その明示パスを t=1 の load_from に使う（URL 可用性や
#   torch hub キャッシュの状態に依存せず、どのノードでも同一の θ0 を確定的に読ませる）。
#   未設定なら config の load_from（openmmlab の URL）をそのまま使う（本環境での既定）。
THETA0="${THETA0:-}"

for dom in "${ORDER[@]}"; do
  WD="$OUT/fullft_replayfree_${dom}_work_dir"
  echo "==================== [$dom] 学習（条件A・リプレイ無し） ===================="
  if [ -z "$prev_ckpt" ] && [ -n "$THETA0" ]; then
    echo "[$dom] t=1: θ0 を明示パスで指定: $THETA0"
    bash tools/dist_train.sh "$CFG/fullft_replayfree_${dom}.py" "$GPUS" --work-dir "$WD" \
      --cfg-options load_from="$THETA0"
  elif [ -z "$prev_ckpt" ]; then
    echo "[$dom] t=1: config の load_from（θ0 URL）を使用"
    bash tools/dist_train.sh "$CFG/fullft_replayfree_${dom}.py" "$GPUS" --work-dir "$WD"
  else
    bash tools/dist_train.sh "$CFG/fullft_replayfree_${dom}.py" "$GPUS" --work-dir "$WD" \
      --cfg-options load_from="$prev_ckpt"
  fi

  # last ckpt（mmengine が work_dir に last_checkpoint を書く）
  last="$(cat "$WD/last_checkpoint")"
  echo "[$dom] last ckpt: $last"

  # 現ドメイン + 全過去ドメインの検出性能（ドメイン別・平均は取らない）
  for evd in "$dom" "${PAST[@]}"; do
    echo "-------------------- [$dom] 評価 on $evd --------------------"
    bash tools/dist_test.sh "$EVALCFG/eval_${evd}.py" "$last" "$GPUS" \
      --work-dir "$OUT/eval_after_${dom}/on_${evd}"
  done

  # ZCOCO（COCO2017-val zero-shot）
  echo "-------------------- [$dom] ZCOCO --------------------"
  bash tools/dist_test.sh "$ZCOCO_CFG" "$last" "$GPUS" \
    --work-dir "$OUT/eval_after_${dom}/zcoco"

  prev_ckpt="$last"
  PAST+=("$dom")
done

echo "==================== exp_024 リプレイ無し逐次FT（条件A）完了 ===================="
