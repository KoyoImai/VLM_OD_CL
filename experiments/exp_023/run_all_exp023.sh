#!/usr/bin/env bash
# =============================================================================
# exp_023 全比較手法の逐次実行オーケストレーション
#   実行順（ユーザー指定 2026-07-22。手法ごとに replay 有無を隣接）:
#     1. ZiRa   リプレイなし
#     2. ZiRa   リプレイあり
#     3. DitHub リプレイなし
#     4. DitHub リプレイあり
#     5. full finetuning + リプレイ
#     6. 条件A + リプレイ
#   各ドライバが「ドメイン逐次の学習＋評価（現在＋全過去＋ZCOCO）」を内部で行う。
#   ※ 実行は design.md 承認後（行動原理3）。既存プログラムは無変更。
# 使い方: bash experiments/exp_023/run_all_exp023.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # experiments/exp_023/ -> リポジトリルート
D=experiments/exp_023

echo "########## 1/6: ZiRa リプレイなし ##########"
bash "$D/run_sequential_zira.sh" replayfree

echo "########## 2/6: ZiRa リプレイあり ##########"
bash "$D/run_sequential_zira.sh" replay

echo "########## 3/6: DitHub リプレイなし ##########"
bash "$D/run_sequential_dithub.sh" replayfree

echo "########## 4/6: DitHub リプレイあり ##########"
bash "$D/run_sequential_dithub.sh" replay

echo "########## 5/6: full finetuning + リプレイ ##########"
bash "$D/run_sequential_fullft_replay.sh"

echo "########## 6/6: 条件A + リプレイ ##########"
bash "$D/run_sequential_condA.sh"

echo "########## exp_023 全手法 完了 ##########"
