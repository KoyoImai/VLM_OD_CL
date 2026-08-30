#!/usr/bin/env bash
# =============================================================================
# exp_054 EWC: seed 変更版（exp_045 の run_ewc.sh を踏襲。λ=1000）
#   変更点は randomness.seed / Fisher 推定の --seed / 出力先 / last のみ保存だけ。
# 使い方: SEED=1 bash experiments/exp_054/run_ewc.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
SEED="${SEED:?SEED=1|2 を指定}"
GPUS="${GPUS:-4}"
LAM="${LAM:-1000}"
if [ -z "${THETA0:-}" ]; then
  for p in \
    grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    /workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth; do
    [ -f "$p" ] && THETA0="$p" && break
  done
fi
[ -n "${THETA0:-}" ] || { echo "θ0 が見つからない"; exit 1; }

EXP=experiments/exp_054
CFG=experiments/exp_045/configs
STATEDIR="$EXP/ewc_s${SEED}_states"
EVALDIR="$EXP/eval_ewc_s${SEED}"
mkdir -p "$STATEDIR" "$EVALDIR"
ORDER=(underwater electromagnetic videogames aerial microscopic documents)
eval_cfg() {
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_023/configs/eval_$1.py" ;;
    *) echo "experiments/exp_026/configs/eval_$1.py" ;;
  esac
}
CKPT_OPTS=("default_hooks.checkpoint.interval=20" "default_hooks.checkpoint.save_optimizer=False" \
           "default_hooks.checkpoint.save_param_scheduler=False")

echo "==================== exp_054 EWC λ=$LAM seed=$SEED ===================="
prev=""; prev_state=""; LEARNED=()
for i in "${!ORDER[@]}"; do
  t=$((i + 1)); dom="${ORDER[$i]}"
  WD="$EXP/ewc_s${SEED}_${dom}_work_dir"; ckpt="$WD/epoch_20.pth"
  state="$STATEDIR/state_t${t}.pth"
  echo "==================== [t=$t/6] $dom ===================="
  if [ -f "$state" ]; then
    echo "[t=$t] $state があるためスキップ"
  else
    OPTS=("randomness.seed=$SEED" "model.ewc.lam=$LAM" "load_from=${prev:-$THETA0}" "${CKPT_OPTS[@]}")
    [ -n "$prev_state" ] && OPTS+=("model.ewc.state_path=$prev_state")
    bash tools/dist_train.sh "$CFG/ewc_${dom}.py" "$GPUS" --work-dir "$WD" --cfg-options "${OPTS[@]}"
    FOPTS=(--out "$state" --max-samples 1000 --seed "$SEED" --task-name "$dom")
    [ -n "$prev_state" ] && FOPTS+=(--prev-state "$prev_state")
    python projects/ewc_cl/estimate_fisher.py "$CFG/ewc_${dom}.py" "$ckpt" "${FOPTS[@]}"
  fi
  prev="$ckpt"; prev_state="$state"; LEARNED+=("$dom")
  for d in "${LEARNED[@]}"; do
    [ -d "$EVALDIR/t${t}_on_${d}" ] || \
      bash tools/dist_test.sh "$(eval_cfg "$d")" "$prev" "$GPUS" --work-dir "$EVALDIR/t${t}_on_${d}"
  done
  [ -d "$EVALDIR/t${t}_zcoco" ] || \
    bash tools/dist_test.sh configs/mm_grounding_dino/eval_base_coco.py "$prev" "$GPUS" \
      --work-dir "$EVALDIR/t${t}_zcoco"
done
echo "==================== exp_054 EWC seed=$SEED 完了 ===================="
