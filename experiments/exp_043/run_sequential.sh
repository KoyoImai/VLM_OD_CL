#!/usr/bin/env bash
# =============================================================================
# exp_043: リプレイ／蒸留E＋リプレイ（バッチ [4,2,2]×8）の 6 ドメイン逐次ドライバ
#          （design.md の承認後に実行する）
#
#   各ドメイン t（1..6）で:
#     1. 学習（20 epoch）。load_from は θ_{t-1}（t=1 は config の θ0）。
#        kdE は教師 θ_{t-1} を model.teacher_ckpt に渡す（t=1 は θ0 の実パス。
#        **model. を付ける**。付けないとトップレベルに行って届かない）。
#     2. 評価: 学習済みドメイン 1..t と ZCOCO。
#        評価 config は前半3ドメインが exp_023、後半3ドメインが exp_026。
#
#   途中で止まった場合は同じコマンドで再開できる（epoch_20.pth があるドメインはスキップ）。
#
# 使い方:
#   COND=replay bash experiments/exp_043/run_sequential.sh
#   COND=kdE    bash experiments/exp_043/run_sequential.sh
#   GPUS=4 が既定（クラスタ）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

COND="${COND:?COND=replay|kdE を指定}"
GPUS="${GPUS:-4}"

# θ0（kdE の t=1 の教師）。本環境はリポジトリ直下、クラスタは /workspace/kouyou/ckpt の bind
if [ -z "${THETA0:-}" ]; then
  for p in \
    grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    /workspace/kouyou/ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth; do
    [ -f "$p" ] && THETA0="$p" && break
  done
fi
[ -n "${THETA0:-}" ] || { echo "θ0 が見つからない（THETA0= で指定可）"; exit 1; }

EXP=experiments/exp_043
CFG="$EXP/configs"
EVALDIR="$EXP/eval_${COND}"
mkdir -p "$EVALDIR"

ORDER=(underwater electromagnetic videogames aerial microscopic documents)

eval_cfg() {
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_023/configs/eval_$1.py" ;;
    aerial|microscopic|documents)          echo "experiments/exp_026/configs/eval_$1.py" ;;
  esac
}

echo "==================== exp_043 $COND ===================="
echo "GPUS=$GPUS / θ0: $THETA0 / 出力: $EVALDIR"

prev=""            # θ_{t-1}（空なら config の θ0）
LEARNED=()

for i in "${!ORDER[@]}"; do
  t=$((i + 1))
  dom="${ORDER[$i]}"
  WD="$EXP/${COND}_${dom}_work_dir"
  ckpt="$WD/epoch_20.pth"

  echo "==================== [t=$t/6] $dom 学習 ===================="
  if [ -f "$ckpt" ]; then
    echo "[t=$t] $ckpt が既にあるため学習をスキップ"
  else
    OPTS=()
    [ -n "$prev" ] && OPTS+=("load_from=$prev")
    if [ "$COND" = "kdE" ]; then
      teacher="${prev:-$THETA0}"
      OPTS+=("model.teacher_ckpt=$teacher")
      echo "[t=$t] 教師 θ^T = $teacher"
    fi
    if [ ${#OPTS[@]} -gt 0 ]; then
      bash tools/dist_train.sh "$CFG/${COND}_${dom}.py" "$GPUS" \
        --work-dir "$WD" --cfg-options "${OPTS[@]}"
    else
      bash tools/dist_train.sh "$CFG/${COND}_${dom}.py" "$GPUS" --work-dir "$WD"
    fi
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

echo "==================== exp_043 $COND 完了（最終: $prev）===================="
