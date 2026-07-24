#!/usr/bin/env bash
# =============================================================================
# exp_026-4 ドライバ: オラクル（条件A＝全モジュール）/ 3ドメイン同時学習
#   underwater + electromagnetic + videogames を ConcatDataset で結合し1モデルを学習。
#   Objects365 は含めない（note04 の指定）。均衡化なし（design.md §4 の決定）。
#   学習後、3ドメインそれぞれ ＋ ZCOCO を評価する。
#
#   使い方: bash experiments/exp_026/run_oracle.sh
#   ※ 実行は design.md 承認かつ exp_023.5 完了後（着手条件・行動原理3）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

GPUS=4
export PORT="${PORT:-29590}"
OUT=experiments/exp_026
CFG=experiments/exp_026/configs/oracle_condA_joint.py
EVALCFG=experiments/exp_023/configs
ZCOCO_CFG=configs/mm_grounding_dino/eval_base_coco.py
WD="$OUT/oracle_condA_work_dir"
THETA0="${THETA0:-}"

echo "==================== [オラクル/条件A] 3ドメイン同時学習 ===================="
if [ -n "$THETA0" ]; then
  bash tools/dist_train.sh "$CFG" "$GPUS" --work-dir "$WD" --cfg-options load_from="$THETA0"
else
  bash tools/dist_train.sh "$CFG" "$GPUS" --work-dir "$WD"
fi

last="$(cat "$WD/last_checkpoint")"
echo "[オラクル] last ckpt: $last"

# 3ドメインそれぞれを個別に評価（平均は取らない）
for dom in underwater electromagnetic videogames; do
  echo "-------------------- [オラクル] 評価 on $dom --------------------"
  bash tools/dist_test.sh "$EVALCFG/eval_${dom}.py" "$last" "$GPUS" \
    --work-dir "$OUT/oracle_condA_eval/on_${dom}"
done

echo "-------------------- [オラクル] ZCOCO --------------------"
bash tools/dist_test.sh "$ZCOCO_CFG" "$last" "$GPUS" \
  --work-dir "$OUT/oracle_condA_eval/zcoco"

echo "==================== exp_026 オラクル（条件A）完了 ===================="
