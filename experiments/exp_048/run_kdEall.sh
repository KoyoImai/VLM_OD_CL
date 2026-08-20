#!/usr/bin/env bash
# =============================================================================
# exp_048 条件2 kdEall: リプレイ[4,1,1]×6・全6枚に蒸留E の 6 ドメイン逐次ドライバ
#          （design.md の承認後に実行する。2026-08-20 承認済み）
#
#   各ドメイン t（1..6）で:
#     1. 学習（20 epoch）。load_from は θ_{t-1}（t=1 は θ0 の実パス）。
#        教師 θ_{t-1} を model.teacher_ckpt に渡す（t=1 は θ0。**model. を付ける**）。
#     2. 評価: 学習済みドメイン 1..t と ZCOCO（plain 評価 config）。
#
#   途中で止まった場合は同じコマンドで再開できる（epoch_20.pth があるドメインはスキップ）。
#
# 使い方:
#   bash experiments/exp_048/run_kdEall.sh      # GPUS=4 が既定（クラスタ）
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

GPUS="${GPUS:-4}"

if [ -z "${THETA0:-}" ]; then
  for p in \
    grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    /workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth; do
    [ -f "$p" ] && THETA0="$p" && break
  done
fi
[ -n "${THETA0:-}" ] || { echo "θ0 が見つからない（THETA0= で指定可）"; exit 1; }

EXP=experiments/exp_048
CFG="$EXP/configs"
EVALDIR="$EXP/eval_kdEall"
mkdir -p "$EVALDIR"

ORDER=(underwater electromagnetic videogames aerial microscopic documents)

eval_cfg() {
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_023/configs/eval_$1.py" ;;
    aerial|microscopic|documents)          echo "experiments/exp_026/configs/eval_$1.py" ;;
  esac
}

echo "==================== exp_048 kdEall ===================="
echo "GPUS=$GPUS / θ0: $THETA0 / 出力: $EVALDIR"

prev=""
LEARNED=()

for i in "${!ORDER[@]}"; do
  t=$((i + 1))
  dom="${ORDER[$i]}"
  WD="$EXP/kdEall_${dom}_work_dir"
  ckpt="$WD/epoch_20.pth"

  echo "==================== [t=$t/6] $dom 学習 ===================="
  if [ -f "$ckpt" ]; then
    echo "[t=$t] $ckpt が既にあるため学習をスキップ"
  else
    # t=1 は load_from にも θ0 の実パスを渡す（exp_028 と同じ規約。学生と教師が
    # 同一ファイル由来でビット単位に一致し、config の URL 解決にも依存しない）
    teacher="${prev:-$THETA0}"
    echo "[t=$t] 教師 θ^T = $teacher"
    OPTS=("model.teacher_ckpt=$teacher" "load_from=$teacher")
    bash tools/dist_train.sh "$CFG/kdEall_${dom}.py" "$GPUS" \
      --work-dir "$WD" --cfg-options "${OPTS[@]}"
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

echo "==================== exp_048 kdEall 完了（最終: $prev）===================="
