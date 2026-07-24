#!/usr/bin/env bash
# =============================================================================
# exp_026-1/2/3 ドライバ: 個別チューニング（条件A・条件B）／個別 ZiRa
#   各ドメインを **θ0 から独立に** 学習する（逐次ではないので前ドメインを引き継がない）。
#   学習後の評価は「自ドメインの適応 mAP」＋「ZCOCO」のみ（過去ドメインは存在しない）。
#
#   config は新規に作らず、逐次実験のものをそのまま流用する。こうすることで
#   「個別 vs 逐次」の差が **初期値（θ0 か前ドメインか）だけ** に閉じ、比較が厳密になる:
#     条件A → exp_024 の fullft_replayfree_{dom}.py
#     条件B → exp_025 の condB_replayfree_{dom}.py
#     ZiRa  → exp_023 の zira_replayfree_{dom}.py（論文準拠 2000 iter / lr 1e-3）
#            ※ exp_023 の config は backbone.init_cfg が URL のままなので、exp_026 の
#              決定に合わせて --cfg-options で None に落とす（重みへの影響は無し）。
#   ZiRa の評価は融合前の生 last ckpt を使う（exp_023 の逐次ドライバと同一。融合は
#   次ドメイン用の処理であり、個別学習では不要）。
#
#   使い方: bash experiments/exp_026/run_indiv.sh [condA|condB|zira|all]
#   ※ 実行は design.md 承認かつ exp_023.5 完了後（着手条件・行動原理3）。既存ファイルは無変更。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # experiments/exp_026/ -> リポジトリルート

GPUS=4
export PORT="${PORT:-29580}"
OUT=experiments/exp_026
EVALCFG=experiments/exp_023/configs
ZCOCO_CFG=configs/mm_grounding_dino/eval_base_coco.py
ORDER=(underwater electromagnetic videogames)
WHICH="${1:-all}"

# θ0（未設定なら config の load_from=θ0 URL を使う）
THETA0="${THETA0:-}"

run_method () {
  local method="$1"
  for dom in "${ORDER[@]}"; do
    local cfg extra evalcfg zcococfg
    case "$method" in
      condA) cfg="experiments/exp_024/configs/fullft_replayfree_${dom}.py"; extra=""
             evalcfg="$EVALCFG/eval_${dom}.py"; zcococfg="$ZCOCO_CFG" ;;
      condB) cfg="experiments/exp_025/configs/condB_replayfree_${dom}.py"; extra=""
             evalcfg="$EVALCFG/eval_${dom}.py"; zcococfg="$ZCOCO_CFG" ;;
      zira)  cfg="$EVALCFG/zira_replayfree_${dom}.py"; extra="model.backbone.init_cfg=None"
             evalcfg="$EVALCFG/zira_eval_${dom}.py"; zcococfg="$EVALCFG/zira_eval_zcoco.py" ;;
      *) echo "unknown method: $method"; exit 1 ;;
    esac

    local WD="$OUT/indiv_${method}_${dom}_work_dir"
    echo "==================== [個別/$method] $dom 学習（θ0 から独立） ===================="
    # 個別学習: 常に θ0 から開始（前ドメインを引き継がない）
    local opts=()
    [ -n "$extra" ] && opts+=("$extra")
    [ -n "$THETA0" ] && opts+=("load_from=$THETA0")
    if [ ${#opts[@]} -gt 0 ]; then
      bash tools/dist_train.sh "$cfg" "$GPUS" --work-dir "$WD" --cfg-options "${opts[@]}"
    else
      bash tools/dist_train.sh "$cfg" "$GPUS" --work-dir "$WD"
    fi

    local last; last="$(cat "$WD/last_checkpoint")"
    echo "[個別/$method] $dom last ckpt: $last"

    echo "-------------------- [個別/$method] $dom 適応評価 --------------------"
    bash tools/dist_test.sh "$evalcfg" "$last" "$GPUS" \
      --work-dir "$OUT/indiv_${method}_eval/${dom}/on_${dom}"
    echo "-------------------- [個別/$method] $dom ZCOCO --------------------"
    bash tools/dist_test.sh "$zcococfg" "$last" "$GPUS" \
      --work-dir "$OUT/indiv_${method}_eval/${dom}/zcoco"
  done
}

case "$WHICH" in
  condA|condB|zira) run_method "$WHICH" ;;
  all) run_method condA; run_method condB; run_method zira ;;
  *) echo "usage: $0 [condA|condB|zira|all]"; exit 1 ;;
esac

echo "==================== exp_026 個別チューニング（$WHICH）完了 ===================="
