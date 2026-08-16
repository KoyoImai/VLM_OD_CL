#!/usr/bin/env bash
# =============================================================================
# exp_045 EWC: RF100 6 ドメイン逐次（バッファ不使用）ドライバ
#              （design.md の承認後に実行する）
#
#   各ドメイン t（1..6）で:
#     1. 学習（20 epoch、4 GPU）。load_from は θ_{t-1}（t=1 は θ0 の実パス）。
#        --cfg-options で λ と EWC 状態を毎回明示（t=1 は状態無し = ペナルティ不活性）。
#     2. Fisher 推定 → 2 バッファ状態を蓄積（projects/ewc_cl/estimate_fisher.py、
#        上限 1,000 枚・seed 0）。
#     3. 評価: 学習済みドメイン 1..t と ZCOCO（plain 評価 config。EWC の ckpt は
#        素の GroundingDINO 構造なのでそのまま読める）。
#
#   途中で止まった場合は同じコマンドで再開できる（状態ファイルがあるタスクはスキップ）。
#
# 使い方:
#   LAM=1000 bash experiments/exp_045/run_ewc.sh      # GPUS=4 が既定（クラスタ）
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

GPUS="${GPUS:-4}"
LAM="${LAM:-1000}"     # design.md §2.2（2026-08-16 確定: λ=10^3）

if [ -z "${THETA0:-}" ]; then
  for p in \
    grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    /workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth; do
    [ -f "$p" ] && THETA0="$p" && break
  done
fi
[ -n "${THETA0:-}" ] || { echo "θ0 が見つからない（THETA0= で指定可）"; exit 1; }

EXP=experiments/exp_045
CFG="$EXP/configs"
STATEDIR="$EXP/ewc_states"
EVALDIR="$EXP/eval_ewc"
mkdir -p "$STATEDIR" "$EVALDIR"

ORDER=(underwater electromagnetic videogames aerial microscopic documents)

eval_cfg() {
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_023/configs/eval_$1.py" ;;
    aerial|microscopic|documents)          echo "experiments/exp_026/configs/eval_$1.py" ;;
  esac
}

echo "==================== exp_045 EWC λ=$LAM ===================="
echo "GPUS=$GPUS / θ0: $THETA0 / 出力: $STATEDIR, $EVALDIR"

prev=""; prev_state=""
LEARNED=()

for i in "${!ORDER[@]}"; do
  t=$((i + 1))
  dom="${ORDER[$i]}"
  WD="$EXP/ewc_${dom}_work_dir"
  ckpt="$WD/epoch_20.pth"
  state="$STATEDIR/state_t${t}.pth"

  echo "==================== [t=$t/6] $dom 学習 ===================="
  if [ -f "$state" ]; then
    echo "[t=$t] $state があるため学習と Fisher をスキップ"
  else
    OPTS=("model.ewc.lam=$LAM" "load_from=${prev:-$THETA0}")
    [ -n "$prev_state" ] && OPTS+=("model.ewc.state_path=$prev_state")
    bash tools/dist_train.sh "$CFG/ewc_${dom}.py" "$GPUS" \
      --work-dir "$WD" --cfg-options "${OPTS[@]}"
    FOPTS=(--out "$state" --max-samples 1000 --seed 0 --task-name "$dom")
    [ -n "$prev_state" ] && FOPTS+=(--prev-state "$prev_state")
    python projects/ewc_cl/estimate_fisher.py "$CFG/ewc_${dom}.py" "$ckpt" \
      "${FOPTS[@]}"
  fi
  prev="$ckpt"; prev_state="$state"
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

echo "==================== exp_045 EWC 完了（最終: $prev）===================="
