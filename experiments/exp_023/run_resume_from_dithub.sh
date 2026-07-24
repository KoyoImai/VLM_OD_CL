#!/usr/bin/env bash
# =============================================================================
# exp_023 再開ドライバ：DitHub（手法3）から実行する。
#   手法1・2（ZiRa なし・あり）は完了済み・有効なのでスキップ。
#   手法3の DitHub は loss() の ODVG 対応修正（DitHubODVGGroundingDINO）済み。
#   順序: 3 DitHub なし → 4 DitHub あり → 5 full FT+replay → 6 条件A+replay。
# 使い方: bash experiments/exp_023/run_resume_from_dithub.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # experiments/exp_023/ -> リポジトリルート
D=experiments/exp_023

echo "########## 3/6: DitHub リプレイなし ##########"
bash "$D/run_sequential_dithub.sh" replayfree

echo "########## 4/6: DitHub リプレイあり ##########"
bash "$D/run_sequential_dithub.sh" replay

echo "########## 5/6: full finetuning + リプレイ ##########"
bash "$D/run_sequential_fullft_replay.sh"

echo "########## 6/6: 条件A + リプレイ ##########"
bash "$D/run_sequential_condA.sh"

echo "########## exp_023 手法3〜6（再開分）完了 ##########"
