#!/usr/bin/env bash
# =============================================================================
# exp_053: 蒸留E × 学習可能モジュールアブレーション・後半 3 ドメイン逐次ドライバ
#          （design.md の承認後に実行する。2026-08-25 承認済み）
#
#   exp_052（前半 3 ドメイン）の続き。各ドメイン t（4..6）で:
#     1. 学習（20 epoch）。load_from・教師 θ_{t-1} は前ドメイン epoch_20。
#        t=4 は exp_052 の**同一条件**の videogames epoch_20（START で差し替え可）。
#     2. 評価: 学習済みドメイン 1..t と ZCOCO（plain 評価 config）。
#
#   再開: epoch_20.pth のあるドメインは学習をスキップ。評価はディレクトリ有無で
#   スキップ（json の無い評価ディレクトリは消してから再投入すること）。
#
# 使い方:
#   COND=swinNeck bash experiments/exp_053/run_sequential.sh
#   COND は swinNeck|bertTfm|enhancer|qsel|decoder。GPUS=4 が既定（クラスタ）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

COND="${COND:?COND=swinNeck|bertTfm|enhancer|qsel|decoder を指定}"
case "$COND" in swinNeck|bertTfm|enhancer|qsel|decoder) ;; \
  *) echo "COND は swinNeck|bertTfm|enhancer|qsel|decoder"; exit 1;; esac
GPUS="${GPUS:-4}"

# t=4 の初期重み = exp_052 同一条件の t=3 ckpt（design.md §1.2）
START="${START:-experiments/exp_052/${COND}_videogames_work_dir/epoch_20.pth}"
[ -f "$START" ] || { echo "exp_052 の t=3 ckpt が無い: $START（exp_052 の完了を確認）"; exit 1; }

EXP=experiments/exp_053
CFG="$EXP/configs"
EVALDIR="$EXP/eval_${COND}"
mkdir -p "$EVALDIR"

ORDER=(aerial microscopic documents)                          # t=4,5,6
LEARNED=(underwater electromagnetic videogames)               # exp_052 で学習済み

eval_cfg() {
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_023/configs/eval_$1.py" ;;
    aerial|microscopic|documents)          echo "experiments/exp_026/configs/eval_$1.py" ;;
  esac
}

echo "==================== exp_053 $COND ===================="
echo "GPUS=$GPUS / START: $START / 出力: $EVALDIR"

prev="$START"

for i in "${!ORDER[@]}"; do
  t=$((i + 4))
  dom="${ORDER[$i]}"
  WD="$EXP/${COND}_${dom}_work_dir"
  ckpt="$WD/epoch_20.pth"

  echo "==================== [t=$t/6] $dom 学習 ===================="
  if [ -f "$ckpt" ]; then
    echo "[t=$t] $ckpt が既にあるため学習をスキップ"
  else
    echo "[t=$t] 教師 θ^T = $prev"
    bash tools/dist_train.sh "$CFG/${COND}_${dom}.py" "$GPUS" \
      --work-dir "$WD" --cfg-options \
      "model.teacher_ckpt=$prev" "load_from=$prev"
  fi
  prev="$ckpt"
  LEARNED+=("$dom")

  for d in "${LEARNED[@]}"; do
    echo "-------------------- [t=$t] $d 評価 --------------------"
    [ -d "$EVALDIR/t${t}_on_${d}" ] || \
      bash tools/dist_test.sh "$(eval_cfg "$d")" "$prev" "$GPUS" \
        --work-dir "$EVALDIR/t${t}_on_${d}"
  done
  echo "-------------------- [t=$t] ZCOCO 評価 --------------------"
  [ -d "$EVALDIR/t${t}_zcoco" ] || \
    bash tools/dist_test.sh configs/mm_grounding_dino/eval_base_coco.py "$prev" \
      "$GPUS" --work-dir "$EVALDIR/t${t}_zcoco"
done

echo "==================== exp_053 $COND 完了（最終: $prev）===================="
