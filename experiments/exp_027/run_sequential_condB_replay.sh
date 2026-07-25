#!/usr/bin/env bash
# =============================================================================
# exp_027-2 逐次ドライバ: 逐次FT + リプレイ（条件B＝特徴抽出+融合）
#   逐次順 underwater -> electromagnetic -> videogames を順に学習・評価する。
#   各ドメイン t で:
#     1. 前ドメインの last ckpt から load_from（t=1 は θ0）
#        ※ load_from は重みのみ読む。LR スケジュール・optimizer 状態はリセットされる。
#     2. dist_train（4GPU）で学習。リプレイ混合は config 側で定義:
#          t=1   [現在, 参照]        source_ratio [2,1]  batch 6 -> 現在4・参照2
#          t>=2  [現在, 参照, 過去]  source_ratio [4,1,1] batch 6 -> 現在4・参照1・過去1
#     3. last ckpt を特定（適応 mAP は last を採用）
#     4. 現ドメイン + それまでの全過去ドメイン + ZCOCO を評価
#     5. この last を次ドメインの load_from にして次へ
#
#   config は exp_023 の condA_replay_*.py（旧呼称。中身は特徴抽出+融合）を継承したエイリアス。
#   下流（decoder / bbox_head / query・dn・memory_trans 系）は lr_mult=0.0 で凍結される。
#   評価 config は exp_023 のものを流用（無変更）。
#   ※ 実行は design.md の着手条件を満たしてから（行動原理3）。既存ファイルは無変更。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # experiments/exp_027/ -> リポジトリルート

GPUS=4
export PORT="${PORT:-29640}"          # exp_024(29540)/exp_025(29560)/exp_026(29580) と衝突回避
CFG=experiments/exp_027/configs
EVALCFG=experiments/exp_023/configs   # eval_{domain}.py を流用
OUT=experiments/exp_027
ZCOCO_CFG=configs/mm_grounding_dino/eval_base_coco.py

ORDER=(underwater electromagnetic videogames)
PAST=()
prev_ckpt=""   # 空なら t=1（θ0 から開始）

# θ0 の与え方（exp_023.5 以来の方針）:
#   THETA0 が設定されていれば、その明示パスを t=1 の load_from に使う。
#   未設定なら config の load_from（openmmlab の URL）をそのまま使う（本環境での既定）。
THETA0="${THETA0:-}"

# backbone の init_cfg（SwinT の ImageNet 重み URL）を無効化する。
#   load_from が backbone 187 パラメータを全て上書きするため最終重みに影響しない
#   （exp_024 design.md §7 で全 908 テンソルの一致を実測確認済み）。exp_024/025/026 と揃える。
INITCFG_OPT=(model.backbone.init_cfg=None)

for dom in "${ORDER[@]}"; do
  WD="$OUT/condB_replay_${dom}_work_dir"
  echo "==================== [027-2][$dom] 学習（条件B・リプレイあり） ===================="
  if [ -z "$prev_ckpt" ] && [ -n "$THETA0" ]; then
    echo "[$dom] t=1: θ0 を明示パスで指定: $THETA0"
    bash tools/dist_train.sh "$CFG/condB_replay_${dom}.py" "$GPUS" --work-dir "$WD" \
      --cfg-options load_from="$THETA0" "${INITCFG_OPT[@]}"
  elif [ -z "$prev_ckpt" ]; then
    echo "[$dom] t=1: config の load_from（θ0 URL）を使用"
    bash tools/dist_train.sh "$CFG/condB_replay_${dom}.py" "$GPUS" --work-dir "$WD" \
      --cfg-options "${INITCFG_OPT[@]}"
  else
    bash tools/dist_train.sh "$CFG/condB_replay_${dom}.py" "$GPUS" --work-dir "$WD" \
      --cfg-options load_from="$prev_ckpt" "${INITCFG_OPT[@]}"
  fi

  last="$(cat "$WD/last_checkpoint")"
  echo "[027-2][$dom] last ckpt: $last"

  # 現ドメイン + 全過去ドメイン（ドメイン別。平均は取らない）
  for evd in "$dom" "${PAST[@]}"; do
    echo "-------------------- [027-2][$dom] 評価 on $evd --------------------"
    bash tools/dist_test.sh "$EVALCFG/eval_${evd}.py" "$last" "$GPUS" \
      --work-dir "$OUT/condB_replay_eval_after_${dom}/on_${evd}"
  done

  # ZCOCO（COCO2017-val zero-shot）
  echo "-------------------- [027-2][$dom] ZCOCO --------------------"
  bash tools/dist_test.sh "$ZCOCO_CFG" "$last" "$GPUS" \
    --work-dir "$OUT/condB_replay_eval_after_${dom}/zcoco"

  prev_ckpt="$last"
  PAST+=("$dom")
done

echo "==================== exp_027-2 逐次FT + リプレイ（条件B）完了 ===================="
