#!/usr/bin/env bash
# =============================================================================
# exp_054 Ours（蒸留E λ=10・旧バッチ）: seed 変更版の 6 ドメイン逐次ドライバ
#   config は exp_035（t1-3）/ exp_040（t4-6）の kdE をそのまま使い、
#   上書きは randomness.seed と ckpt 保存方針（last のみ）だけ（design.md §1）。
# 使い方: SEED=1 bash experiments/exp_054/run_ours.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
SEED="${SEED:?SEED=1|2 を指定}"
GPUS="${GPUS:-4}"
if [ -z "${THETA0:-}" ]; then
  for p in \
    grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    /workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth; do
    [ -f "$p" ] && THETA0="$p" && break
  done
fi
[ -n "${THETA0:-}" ] || { echo "θ0 が見つからない"; exit 1; }

EXP=experiments/exp_054
EVALDIR="$EXP/eval_ours_s${SEED}"
mkdir -p "$EVALDIR"
ORDER=(underwater electromagnetic videogames aerial microscopic documents)

train_cfg() {
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_035/configs/kdE_condA_l2w100_$1.py" ;;
    *) echo "experiments/exp_040/configs/kdE_$1.py" ;;
  esac
}
eval_cfg() {
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_023/configs/eval_$1.py" ;;
    *) echo "experiments/exp_026/configs/eval_$1.py" ;;
  esac
}
CKPT_OPTS=("default_hooks.checkpoint.interval=20" "default_hooks.checkpoint.save_optimizer=False" \
           "default_hooks.checkpoint.save_param_scheduler=False")

echo "==================== exp_054 Ours seed=$SEED ===================="
prev=""; LEARNED=()
for i in "${!ORDER[@]}"; do
  t=$((i + 1)); dom="${ORDER[$i]}"
  WD="$EXP/ours_s${SEED}_${dom}_work_dir"; ckpt="$WD/epoch_20.pth"
  echo "==================== [t=$t/6] $dom 学習 ===================="
  if [ -f "$ckpt" ]; then
    echo "[t=$t] $ckpt があるためスキップ"
  else
    teacher="${prev:-$THETA0}"
    bash tools/dist_train.sh "$(train_cfg "$dom")" "$GPUS" --work-dir "$WD" \
      --cfg-options "randomness.seed=$SEED" "model.teacher_ckpt=$teacher" \
      "load_from=$teacher" "${CKPT_OPTS[@]}"
  fi
  prev="$ckpt"; LEARNED+=("$dom")
  for d in "${LEARNED[@]}"; do
    [ -d "$EVALDIR/t${t}_on_${d}" ] || \
      bash tools/dist_test.sh "$(eval_cfg "$d")" "$prev" "$GPUS" --work-dir "$EVALDIR/t${t}_on_${d}"
  done
  [ -d "$EVALDIR/t${t}_zcoco" ] || \
    bash tools/dist_test.sh configs/mm_grounding_dino/eval_base_coco.py "$prev" "$GPUS" \
      --work-dir "$EVALDIR/t${t}_zcoco"
done
echo "==================== exp_054 Ours seed=$SEED 完了 ===================="
