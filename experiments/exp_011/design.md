# exp_011: 残り5ドメインの単独FT（Image/Text Backbone も 0.1×学習, unfrozen×seed=0）

## 位置づけ・目的
exp_010(underwater) で得た **「backbone を 0.1×学習させると適応・COCO保持が共に改善（Pareto改善）」** が、
**他ドメインでも一般に成り立つか・ドメイン距離で効果が変わるか**を確認する。
exp_010 の unfrozen×seed=0 方針を残り5ドメインへ展開。

- 能力軸: 本実験も **C（適応）と A（事前学習知識保持=ZCOCO）**。B（過去ドメイン保持）は単独学習のため対象外。

## 対象ドメイン
aerial / microscopic / videogames / documents / electromagnetic（real_world は除外）。
※ underwater は exp_010 で取得済み。

## 設定（exp_010 unfrozen と完全同一・ドメインのみ差し替え）
- config: `experiments/exp_011/configs/<domain>_finetune_unfrozen.py`
  - 標準ドメインFT config を継承し、**backbone・language_model の lr_mult=0.1** に上書き、**seed=0** を設定するのみ。
  - videogames(87クラス) は標準configが max_text_len=512 を持つため継承で対応。
- 据え置き: base lr=1e-4 / 実効64 / 20ep[15] / AdamW wd=1e-4 / clip_grad(0.1) / 評価(800,1333) / **seed=0**。
- 初期重み: 事前学習済み。実行: 4GPU 分散、5ドメイン順次。

## 評価・比較
各ドメインで **適応 mAP（自ドメイン）** と **ZCOCO（COCO保持）** を取得し、
**frozen×seed=0（exp_005 の他5ドメイン）** と並べて **Pareto改善の一般性**を検証。

| 比較軸 | frozen×seed=0 | unfrozen×seed=0 |
|---|---|---|
| 5ドメイン | exp_005（流用） | **exp_011（本実験）** |
| underwater | exp_010 frozen | exp_010 unfrozen |

## 実行
```bash
for d in aerial microscopic videogames documents electromagnetic; do
  bash tools/dist_train.sh experiments/exp_011/configs/${d}_finetune_unfrozen.py 4 \
    --work-dir ./experiments/exp_011/${d}_unfrozen_work_dir
done
# 各 best ckpt で 適応評価（自ドメイン config）＋ COCO評価（eval_base_coco.py）
```

## 想定される懸念
1. **BERT 学習の安定性**: lr_mult=0.1（実効1e-5）。grad_norm を監視、発散時は当該ドメインを止め見直す。
2. **videogames 87クラス**: max_text_len=512 対応済み（標準config継承）。
3. OOM: per-GPU8 で実績内。

## 成功基準
- 5ドメインの適応 mAP と ZCOCO を seed=0 で取得。
- frozen×seed=0（exp_005）と比較し、**「backbone 0.1×学習の Pareto 改善」が一般的か**を判定。
- ドメイン距離（exp_005/007 の知見）と効果の関係を考察。

## 関連
- 起点: [[../exp_010/results/underwater_frozen_vs_unfrozen]] / [[../exp_010/design]]
- frozen基準: exp_005（他5ドメイン, seed=0）/ 能力軸: [[../exp_009/minutes]]
- seed方針: 全実験 seed=0 固定。
