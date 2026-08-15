#!/usr/bin/env bash
# =============================================================================
# exp_041: ZiRa / DitHub の後半3ドメイン逐次ドライバ（design.md の承認後に実行する）
#
#   exp_039 の 3 タスク目（videogames）終了時の融合済みパラメータを初期値として、
#   aerial -> microscopic -> documents を逐次学習する。
#
#   各ドメイン t（4->5->6）で:
#     1. 学習（20 epoch）。load_from は前タスクの融合済み ckpt（t=4 は §2.1 の初期パラメータ）。
#     2. タスク境界処理:
#          ZiRa   : experiments/exp_020/merge_hlrb.py で Rep+
#                   （W_llrb += s*W_hlrb、HLRB を 1e-8・s を 0.1 に再初期化）
#          DitHub : experiments/exp_021/merge_dithub.py で式4（B の λ_B=0.7 融合）
#                   ＋ クラス別 A の和集合。**t=4 から常に前ライブラリと融合する**
#                   （exp_039 の t=1 スキップは初期ライブラリが無いための処置で、
#                   本実験は初期パラメータがライブラリそのもの）。
#     3. 評価: 学習済み 1..t ドメインと ZCOCO。
#          ZiRa   : 前半3ドメインと ZCOCO は exp_023 の zira_eval_*、後半は exp_041。
#          DitHub : 全評価を exp_041 の 260 クラス和集合版で行う（design.md §4）。
#
#   途中で止まった場合は同じコマンドで再開できる（融合済み ckpt があるドメインはスキップ）。
#
# 使い方:
#   METHOD=zira   REPLAY=replayfree bash experiments/exp_041/run_sequential.sh
#   METHOD=dithub REPLAY=replay     bash experiments/exp_041/run_sequential.sh
#   GPUS=4 が既定（クラスタ）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

METHOD="${METHOD:?METHOD=zira|dithub を指定}"
REPLAY="${REPLAY:?REPLAY=replay|replayfree を指定}"
GPUS="${GPUS:-4}"

case "$METHOD" in
  zira)   MERGE=experiments/exp_020/merge_hlrb.py ;;
  dithub) MERGE=experiments/exp_021/merge_dithub.py ;;
  *) echo "METHOD は zira か dithub"; exit 1 ;;
esac

EXP=experiments/exp_041
CFG="$EXP/configs"
TAG="${METHOD}_${REPLAY}"

# 初期パラメータ = exp_039 の t=3 融合済み ckpt（design.md §2.1）。
# git 管理外なので、クラスタでは既存クローンからのコピーが必要（README.md §3）。
INIT=experiments/exp_039/${TAG}_merged/merged_after_t3_videogames.pth
[ -f "$INIT" ] || { echo "初期パラメータが無い: $INIT（README.md §3 のコピーを実施する）"; exit 1; }

MERGEDIR="$EXP/${TAG}_merged"
EVALDIR="$EXP/eval_${TAG}"
mkdir -p "$MERGEDIR" "$EVALDIR"

ORDER=(aerial microscopic documents)                 # 後半3ドメイン（t=4,5,6）
LEARNED=(underwater electromagnetic videogames)      # 前半3ドメイン（exp_039 で学習済み）

# 評価 config の在り処（design.md §4）
eval_cfg() {  # $1 = ドメイン名 or zcoco
  if [ "$METHOD" = "zira" ]; then
    case "$1" in
      underwater|electromagnetic|videogames|zcoco)
        echo "experiments/exp_023/configs/zira_eval_$1.py" ;;
      *) echo "$CFG/zira_eval_$1.py" ;;
    esac
  else
    echo "$CFG/dithub_eval_$1.py"
  fi
}

echo "==================== exp_041 $METHOD / $REPLAY ===================="
echo "GPUS=$GPUS / 初期: $INIT / 融合: $MERGE / 出力: $MERGEDIR, $EVALDIR"

prev="$INIT"

for i in "${!ORDER[@]}"; do
  t=$((4 + i))
  dom="${ORDER[$i]}"
  WD="$EXP/${TAG}_${dom}_work_dir"
  merged="$MERGEDIR/merged_after_t${t}_${dom}.pth"

  echo "==================== [t=$t/6] $dom 学習 ===================="
  if [ -f "$merged" ]; then
    echo "[t=$t] $merged が既にあるため学習と融合をスキップ"
  else
    bash tools/dist_train.sh "$CFG/${TAG}_${dom}.py" "$GPUS" \
      --work-dir "$WD" --cfg-options load_from="$prev"
    ckpt="$(cat "$WD/last_checkpoint")"
    echo "[t=$t] last ckpt: $ckpt"
    if [ "$METHOD" = "dithub" ]; then
      python "$MERGE" "$prev" "$ckpt" "$merged"
      echo "[t=$t] library 更新: $merged（式4 + A 和集合）"
    else
      python "$MERGE" "$ckpt" "$merged"
      echo "[t=$t] Rep+: $merged"
    fi
  fi
  prev="$merged"
  LEARNED+=("$dom")

  for d in "${LEARNED[@]}"; do
    echo "-------------------- [t=$t] $d 評価 --------------------"
    bash tools/dist_test.sh "$(eval_cfg "$d")" "$prev" "$GPUS" \
      --work-dir "$EVALDIR/t${t}_on_${d}"
  done

  echo "-------------------- [t=$t] ZCOCO 評価 --------------------"
  bash tools/dist_test.sh "$(eval_cfg zcoco)" "$prev" "$GPUS" \
    --work-dir "$EVALDIR/t${t}_zcoco"
done

echo "==================== exp_041 $METHOD / $REPLAY 完了（最終: $prev）===================="
