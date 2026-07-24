#!/usr/bin/env bash
# =============================================================================
# exp_026-5 ドライバ: ゼロショット再確認（学習なし・評価のみ）
#   θ0（事前学習済み MM-Grounding DINO）を凍結したまま、RF100 **6ドメイン全て** ＋ COCO を
#   評価する。exp_001 の実測値が再現されるかの確認が目的。
#
#   判定の参照値（exp_001 実測）:
#     COCO 0.504 / underwater 0.051 / aerial 0.034 / electromagnetic 0.024 /
#     videogames 0.016 / documents 0.006 / microscopic 0.002
#   学習を伴わず評価は決定的なので、ほぼ一致するはず。乖離があれば θ0・データ・評価設定の
#   いずれかに差があることを示すため、原因を切り分ける。
#
#   評価 config: underwater/electromagnetic/videogames は exp_023 のものを流用、
#                aerial/microscopic/documents は exp_026 で新規作成したものを使う。
#   使い方: bash experiments/exp_026/run_zeroshot.sh
#   ※ 実行は design.md 承認後（着手条件・行動原理3）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

GPUS=4
export PORT="${PORT:-29600}"
OUT=experiments/exp_026/zeroshot_eval
ZCOCO_CFG=configs/mm_grounding_dino/eval_base_coco.py

# θ0。明示パスが無ければ torch hub キャッシュの既定位置を使う。
THETA0="${THETA0:-$HOME/.cache/torch/hub/checkpoints/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth}"
if [ ! -f "$THETA0" ]; then
  echo "ERROR: θ0 が見つかりません: $THETA0"
  echo "       THETA0=<path> を指定して再実行してください。"
  exit 1
fi
echo "θ0 = $THETA0"

# 評価 config の所在（ドメインごとに参照先が異なる）
cfg_for () {
  case "$1" in
    underwater|electromagnetic|videogames) echo "experiments/exp_023/configs/eval_$1.py" ;;
    aerial|microscopic|documents)          echo "experiments/exp_026/configs/eval_$1.py" ;;
    *) echo "unknown domain: $1" >&2; exit 1 ;;
  esac
}

for dom in underwater aerial videogames microscopic documents electromagnetic; do
  echo "-------------------- [ゼロショット] $dom --------------------"
  bash tools/dist_test.sh "$(cfg_for "$dom")" "$THETA0" "$GPUS" \
    --work-dir "$OUT/on_${dom}"
done

echo "-------------------- [ゼロショット] COCO --------------------"
bash tools/dist_test.sh "$ZCOCO_CFG" "$THETA0" "$GPUS" --work-dir "$OUT/zcoco"

echo "==================== exp_026 ゼロショット再確認 完了 ===================="
echo "参照値: COCO 0.504 / underwater 0.051 / aerial 0.034 / electromagnetic 0.024 /"
echo "        videogames 0.016 / documents 0.006 / microscopic 0.002"
