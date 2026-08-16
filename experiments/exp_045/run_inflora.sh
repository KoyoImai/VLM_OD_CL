#!/usr/bin/env bash
# =============================================================================
# exp_045 InfLoRA: RF100 6 ドメイン逐次（バッファ不使用）ドライバ
#                  （design.md の承認後に実行する）
#
#   各ドメイン t（1..6）で（公式実装の手続き。projects/lora_cl/README.md）:
#     1. 設計: θ_{t-1} で入力共分散を収集し DualGPM メモリで射影して A を設計・固定
#        （inflora_prepare.py。t=1 はメモリ無し。上限 1,000 枚・seed 0）。
#     2. 学習（20 epoch、4 GPU、B のみ）。load_from は θ_{t-1}、design_path を渡す。
#     3. メモリ更新: 学習済み ckpt で共分散を再収集し DualGPM を更新
#        （inflora_update_memory.py。閾値 0.95→1.0 線形、**総タスク数 T=6**）。
#     4. マージ: θ_t = θ_{t-1} + B·A（merge_lora.py。alpha=r → scaling=1）。
#     5. 評価: θ_t で学習済み 1..t ＋ ZCOCO（plain 評価 config）。
#
#   途中で止まっても同じコマンドで再開できる（θ_t があるタスクはスキップ）。
#
# 使い方:
#   bash experiments/exp_045/run_inflora.sh      # GPUS=4 が既定（クラスタ）
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

GPUS="${GPUS:-4}"
LAMB=0.95; LAME=1.0; TOTAL=6

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
DESIGNDIR="$EXP/inflora_designs"
MEMDIR="$EXP/inflora_memory"
THETADIR="$EXP/inflora_theta"
EVALDIR="$EXP/eval_inflora"
mkdir -p "$DESIGNDIR" "$MEMDIR" "$THETADIR" "$EVALDIR"

ORDER=(underwater electromagnetic videogames aerial microscopic documents)

eval_cfg() {
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_023/configs/eval_$1.py" ;;
    aerial|microscopic|documents)          echo "experiments/exp_026/configs/eval_$1.py" ;;
  esac
}

echo "==================== exp_045 InfLoRA ===================="
echo "GPUS=$GPUS / θ0: $THETA0 / 出力: $THETADIR, $EVALDIR"

prev="$THETA0"; prev_mem=""
LEARNED=()

for i in "${!ORDER[@]}"; do
  t=$((i + 1))
  dom="${ORDER[$i]}"
  WD="$EXP/inflora_${dom}_work_dir"
  design="$DESIGNDIR/design_t${t}.pth"
  mem="$MEMDIR/memory_t${t}.pth"
  theta="$THETADIR/theta_t${t}_${dom}.pth"

  echo "==================== [t=$t/6] $dom ===================="
  if [ -f "$theta" ]; then
    echo "[t=$t] $theta があるためスキップ"
  else
    POPT=(--out "$design" --max-samples 1000 --seed 0 --task-name "$dom")
    [ -n "$prev_mem" ] && POPT+=(--memory "$prev_mem")
    python projects/lora_cl/inflora_prepare.py "$CFG/inflora_${dom}.py" \
      "$prev" "${POPT[@]}"

    bash tools/dist_train.sh "$CFG/inflora_${dom}.py" "$GPUS" \
      --work-dir "$WD" \
      --cfg-options "model.inflora.design_path=$design" "load_from=$prev"
    ckpt="$WD/epoch_20.pth"

    MOPT=(--out "$mem" --task-index "$i" --total "$TOTAL"
          --lamb "$LAMB" --lame "$LAME" --max-samples 1000 --seed 0
          --task-name "$dom")
    [ -n "$prev_mem" ] && MOPT+=(--memory "$prev_mem")
    python projects/lora_cl/inflora_update_memory.py "$CFG/inflora_${dom}.py" \
      "$ckpt" "${MOPT[@]}"

    python projects/lora_cl/merge_lora.py "$ckpt" "$theta"
  fi
  prev="$theta"; prev_mem="$mem"
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

echo "==================== exp_045 InfLoRA 完了（最終: $prev）===================="
