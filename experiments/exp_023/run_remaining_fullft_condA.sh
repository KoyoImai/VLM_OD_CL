#!/usr/bin/env bash
# =============================================================================
# exp_023 残り実行：DitHub を外し、full FT+リプレイ と 条件A+リプレイ を回す。
#   手法1・2（ZiRa なし・あり）完了、手法3（DitHub なし）完了、DitHub は次以降の実験へ。
#   ここでは非DitHub の残り2手法を順に学習・評価する。
#   順序: 5 full FT+replay → 6 条件A+replay。
# 使い方: bash experiments/exp_023/run_remaining_fullft_condA.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # experiments/exp_023/ -> リポジトリルート
D=experiments/exp_023

echo "########## 5/6: full finetuning + リプレイ ##########"
bash "$D/run_sequential_fullft_replay.sh"

echo "########## 6/6: 条件A + リプレイ ##########"
bash "$D/run_sequential_condA.sh"

echo "########## exp_023 残り（full FT・条件A、DitHub除く）完了 ##########"
