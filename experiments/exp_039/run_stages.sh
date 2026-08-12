#!/usr/bin/env bash
# =============================================================================
# exp_039: コンテナ内の実行本体。sbatch_{zira,dithub}.sh から呼ばれる。
#
#   引数: $1 = zira | dithub
#         $2 = all | replayfree | replay   （既定 all＝リプレイ無し→有りを直列）
#
#   逐次学習と評価そのものは run_sequential.sh に委譲する。本スクリプトは
#   「どの条件をどの順で流すか」だけを決める（クラスタ側の器と実験ロジックの分離）。
# =============================================================================
set -e
cd /workspace/kouyou/mmdetection

METHOD="${1:?zira か dithub}"
STAGE="${2:-all}"

run_one() {
  echo "############################################################"
  echo "# exp_039  $METHOD / $1   開始: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "############################################################"
  METHOD="$METHOD" REPLAY="$1" GPUS=4 \
    bash experiments/exp_039/run_sequential.sh
  echo "# exp_039  $METHOD / $1   終了: $(date '+%Y-%m-%d %H:%M:%S')"
}

case "$STAGE" in
  replayfree) run_one replayfree ;;
  replay)     run_one replay ;;
  all)        run_one replayfree; run_one replay ;;
  *) echo "STAGE は all / replayfree / replay"; exit 1 ;;
esac

echo "==================== exp_039 $METHOD ($STAGE) 完了 ===================="
