#!/usr/bin/env bash
# =============================================================================
# exp_054 InfLoRA: seed 変更版（exp_045 の run_inflora.sh を踏襲）
#   変更点は randomness.seed / prepare・update の --seed / 出力先 / last のみ保存だけ。
# 使い方: SEED=1 bash experiments/exp_054/run_inflora.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
SEED="${SEED:?SEED=1|2 を指定}"
GPUS="${GPUS:-4}"
LAMB=0.95; LAME=1.0; TOTAL=6
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
DESIGNDIR="$EXP/inflora_s${SEED}_designs"
MEMDIR="$EXP/inflora_s${SEED}_memory"
THETADIR="$EXP/inflora_s${SEED}_theta"
EVALDIR="$EXP/eval_inflora_s${SEED}"
mkdir -p "$DESIGNDIR" "$MEMDIR" "$THETADIR" "$EVALDIR"
ORDER=(underwater electromagnetic videogames aerial microscopic documents)
eval_cfg() {
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_023/configs/eval_$1.py" ;;
    *) echo "experiments/exp_026/configs/eval_$1.py" ;;
  esac
}
CKPT_OPTS=("default_hooks.checkpoint.interval=20" "default_hooks.checkpoint.save_optimizer=False" \
           "default_hooks.checkpoint.save_param_scheduler=False")

echo "==================== exp_054 InfLoRA seed=$SEED ===================="
prev="$THETA0"; prev_mem=""; LEARNED=()
for i in "${!ORDER[@]}"; do
  t=$((i + 1)); dom="${ORDER[$i]}"
  WD="$EXP/inflora_s${SEED}_${dom}_work_dir"
  design="$DESIGNDIR/design_t${t}.pth"
  mem="$MEMDIR/memory_t${t}.pth"
  theta="$THETADIR/theta_t${t}_${dom}.pth"
  echo "==================== [t=$t/6] $dom ===================="
  if [ -f "$theta" ]; then
    echo "[t=$t] $theta があるためスキップ"
  else
    POPT=(--out "$design" --max-samples 1000 --seed "$SEED" --task-name "$dom")
    [ -n "$prev_mem" ] && POPT+=(--memory "$prev_mem")
    python projects/lora_cl/inflora_prepare.py "$CFG/inflora_${dom}.py" "$prev" "${POPT[@]}"
    bash tools/dist_train.sh "$CFG/inflora_${dom}.py" "$GPUS" --work-dir "$WD" \
      --cfg-options "randomness.seed=$SEED" "model.inflora.design_path=$design" \
      "load_from=$prev" "${CKPT_OPTS[@]}"
    ckpt="$WD/epoch_20.pth"
    MOPT=(--out "$mem" --task-index "$i" --total "$TOTAL"
          --lamb "$LAMB" --lame "$LAME" --max-samples 1000 --seed "$SEED"
          --task-name "$dom")
    [ -n "$prev_mem" ] && MOPT+=(--memory "$prev_mem")
    python projects/lora_cl/inflora_update_memory.py "$CFG/inflora_${dom}.py" "$ckpt" "${MOPT[@]}"
    python projects/lora_cl/merge_lora.py "$ckpt" "$theta"
  fi
  prev="$theta"; prev_mem="$mem"; LEARNED+=("$dom")
  for d in "${LEARNED[@]}"; do
    [ -d "$EVALDIR/t${t}_on_${d}" ] || \
      bash tools/dist_test.sh "$(eval_cfg "$d")" "$prev" "$GPUS" --work-dir "$EVALDIR/t${t}_on_${d}"
  done
  [ -d "$EVALDIR/t${t}_zcoco" ] || \
    bash tools/dist_test.sh configs/mm_grounding_dino/eval_base_coco.py "$prev" "$GPUS" \
      --work-dir "$EVALDIR/t${t}_zcoco"
done
echo "==================== exp_054 InfLoRA seed=$SEED 完了 ===================="
