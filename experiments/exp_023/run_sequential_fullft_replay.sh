#!/usr/bin/env bash
# =============================================================================
# exp_023 逐次ドライバ: full finetuning + リプレイ（= 素朴リプレイ）
#   逐次順 underwater -> electromagnetic -> videogames を順に学習・評価する。
#   各ドメイン t で:
#     1. 前ドメインの last ckpt から load_from（t=1 は config の load_from=θ0）
#     2. dist_train（4GPU）で学習（LR/optimizer は毎ドメインでリセット、重みのみ継続）
#     3. last ckpt を特定（適応 mAP は last を採用）
#     4. 現ドメイン + それまでの全過去ドメイン + ZCOCO を評価
#     5. この last を次ドメインの load_from にして次へ
#   ※ 実行は design.md 承認後（行動原理3）。既存ファイルは無変更。
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
  WD="$OUT/full_ft_replay_${dom}_work_dir"
  echo "==================== [$dom] 学習 ===================="
  if [ -z "$prev_ckpt" ]; then
    bash tools/dist_train.sh "$CFG/fullft_replay_${dom}.py" "$GPUS" --work-dir "$WD"
  else
    bash tools/dist_train.sh "$CFG/fullft_replay_${dom}.py" "$GPUS" --work-dir "$WD" \
      --cfg-options load_from="$prev_ckpt"
  fi

  # last ckpt（mmengine が work_dir に last_checkpoint を書く）
  last="$(cat "$WD/last_checkpoint")"
  echo "[$dom] last ckpt: $last"

  # 現ドメイン + 全過去ドメインの検出性能（ドメイン別）
  for evd in "$dom" "${PAST[@]}"; do
    echo "-------------------- [$dom] 評価 on $evd --------------------"
    bash tools/dist_test.sh "$CFG/eval_${evd}.py" "$last" "$GPUS" \
      --work-dir "$OUT/eval_after_${dom}/on_${evd}"
  done

  # ZCOCO（COCO2017-val zero-shot）
  echo "-------------------- [$dom] ZCOCO --------------------"
  bash tools/dist_test.sh "$ZCOCO_CFG" "$last" "$GPUS" \
    --work-dir "$OUT/eval_after_${dom}/zcoco"

  prev_ckpt="$last"
  PAST+=("$dom")
done

echo "==================== exp_023 full finetuning + リプレイ 完了 ===================="
