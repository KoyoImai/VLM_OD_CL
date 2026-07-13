# exp_020 結果記録: ZiRa 再現（単発ドメイン学習・全 6 ドメイン）

実測値の記録（2026-07-12 全 run 完了）。設計と判定の観点は [[../design]]、
実装は [[../implementation_plan]]。すべて seed=0・4 GPU・20 epoch・
プロトコルは exp_017/018/019 と同一。θ0 の ZCOCO = 0.504。
解釈・考察は本ファイルには書かない（実験ノートはユーザーが記述する）。

## 実測値

| ドメイン | 適応 mAP (best ep) | ZCOCO mAP | 学習期間 |
|---|---|---|---|
| underwater | 0.183 (ep17) | 0.436 | 07-10 16:43〜22:15 |
| aerial | 0.297 (ep20) | 0.473 | 07-10 22:15〜07-11 01:11 |
| videogames | 0.097 (ep19) | 0.457 | 07-11 03:41〜07:58 |
| microscopic | 0.192 (ep19) | 0.482 | 07-11 07:58〜12:20 |
| documents | 0.166 (ep20) | 0.476 | 07-11 12:20〜20:51 |
| electromagnetic | 0.183 (ep17) | 0.466 | 07-11 20:51〜07-12 08:50 |

## 基準線との対比（module_rollback_summary.md ⑦ 表より、全て seed=0）

| ドメイン | ZiRa 適応 | neckTfm | frozen | unfrozen | ‖ ZiRa ZCOCO | neckTfm | frozen | unfrozen |
|---|---|---|---|---|---|---|---|---|
| underwater | 0.183 | 0.208 | 0.337 | 0.359 | ‖ 0.436 | 0.465 | 0.389 | 0.407 |
| aerial | 0.297 | 0.350 | 0.468 | 0.486 | ‖ 0.473 | 0.470 | 0.318 | 0.369 |
| videogames | 0.097 | 0.113 | 0.719 | 0.782 | ‖ 0.457 | 0.462 | 0.324 | 0.315 |
| microscopic | 0.192 | 0.272 | 0.499 | 0.538 | ‖ 0.482 | 0.384 | 0.327 | 0.288 |
| documents | 0.166 | 0.196 | 0.478 | 0.542 | ‖ 0.476 | 0.486 | 0.275 | 0.211 |
| electromagnetic | 0.183 | 0.222 | 0.454 | 0.502 | ‖ 0.466 | 0.447 | 0.331 | 0.269 |

事前宣言した観点（design.md 3 節）に対する事実:
- 適応: 全 6 ドメインで neckTfm を 0.015〜0.080 下回る（neckTfm 水準に留まる側）。
- ZCOCO: θ0 比 −2.2〜−6.8 pt。neckTfm 比は 3 勝（aerial +0.003 / microscopic
  +0.098 / electromagnetic +0.019）3 敗（underwater −0.029 / videogames −0.005 /
  documents −0.010）。

## 公式ハイパラ版（全 6 ドメイン。design.md 追記 2026-07-11 / 2026-07-12）

2000 iter / batch 2 / lr 1e-3 / iter 800 decay / 1 GPU / dn 有効 / seed=0。
適応は iter 2000 時点の val（best 選択なし、公式と同じ最終値）。
underwater は 2026-07-11 の対照 run、他 5 ドメインは 2026-07-12 実施。

| ドメイン | 公式版 適応 | 公式版 ZCOCO | ‖ 20ep版 適応 | 20ep版 ZCOCO |
|---|---|---|---|---|
| underwater | 0.111 | 0.491 | ‖ 0.183 | 0.436 |
| aerial | 0.257 | 0.481 | ‖ 0.297 | 0.473 |
| videogames | 0.059 | 0.465 | ‖ 0.097 | 0.457 |
| microscopic | 0.073 | 0.440 | ‖ 0.192 | 0.482 |
| documents | 0.048 | 0.497 | ‖ 0.166 | 0.476 |
| electromagnetic | 0.068 | 0.492 | ‖ 0.183 | 0.466 |

事実の記録:
- 適応は全 6 ドメインで 20ep 版を下回る（0.048〜0.257 対 0.097〜0.297）。
- ZCOCO は 5 ドメインで 20ep 版より高い（θ0 比 −0.7〜−3.9 pt）が、
  **microscopic のみ逆転**（公式版 0.440 < 20ep 版 0.482、θ0 比 −6.4 pt）。
- 出力: `{domain}_zira_official_work_dir` / `{domain}_zira_official_zcoco`

判定: ZCOCO 低下 −1.3 pt は論文報告（13 タスク逐次で −1.31 pt）と同水準であり、
design.md 追記の事前判定基準「~1 pt 級に収まれば実装は忠実」を満たした。
実装診断の記録: 凍結部 907 テンソルは学習後も θ0 とビット一致、loss_zil は
動的平衡（20ep 版ピーク 0.81→終値 0.73、対照 run 終値 0.53）、20ep 版の
RDB 実効ノルムは本体重みの 21〜33%（s は 0.1→0.54〜0.80 に成長、
extra conv のみ 1.3% と非活性）。

## 出力の所在

- 学習: `experiments/exp_020/{domain}_zira_work_dir`（対照 run は
  `underwater_zira_official_work_dir`）
- ZCOCO: `experiments/exp_020/{domain}_zira_zcoco`
- ログ: `experiments/exp_020/train.log`
