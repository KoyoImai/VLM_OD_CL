# exp_022 結果記録（条件A の細分化：extraction/fusion のどこを学習すれば適応するか）

- 完了: 2026-07-21 18:43（train.log「exp_022 完了／全ドメイン×条件 正常終了」）。
- 全 15 run（3ドメイン × 5条件）、seed=0、プロトコルは exp_017/018/019 と同一。
- 本ファイルは事実記録（CLAUDE.md 行動原理8）。解釈・考察は含めない。

## 条件（学習対象モジュール群）

| 条件 | 学習対象 | 学習パラメータ数 |
|---|---|---:|
| a1_img | 画像抽出（Swin + neck） | 29.64M |
| a3_fus | 融合（encoder + level_embed） | 21.91M |
| a4_imgtxt | 画像抽出 + テキスト抽出 | 138.73M |
| a5_imgfus | 画像抽出 + 融合 | 51.54M |
| a6_txtfus | テキスト抽出 + 融合 | 131.00M |

downstream（decoder + query選択 + bbox head、12.29M）は全条件で凍結。

## 実測値（適応 mAP / ZCOCO mAP、θ0 の ZCOCO = 0.504）

| 条件 | ドメイン | 適応 best (ep) | 適応 last | ZCOCO |
|---|---|---|---|---|
| a1_img | underwater | 0.286 (ep20) | 0.286 | 0.401 |
| a3_fus | underwater | 0.323 (ep19) | 0.322 | 0.369 |
| a4_imgtxt | underwater | 0.302 (ep17) | 0.302 | 0.022 |
| a5_imgfus | underwater | 0.343 (ep20) | 0.343 | 0.394 |
| a6_txtfus | underwater | 0.334 (ep19) | 0.332 | 0.393 |
| a1_img | videogames | 0.187 (ep20) | 0.187 | 0.091 |
| a3_fus | videogames | 0.747 (ep19) | 0.746 | 0.278 |
| a4_imgtxt | videogames | 0.696 (ep19) | 0.695 | 0.359 |
| a5_imgfus | videogames | 0.757 (ep19) | 0.755 | 0.300 |
| a6_txtfus | videogames | 0.758 (ep20) | 0.758 | 0.294 |
| a1_img | electromagnetic | 0.315 (ep19) | 0.315 | 0.186 |
| a3_fus | electromagnetic | 0.461 (ep17) | 0.456 | 0.236 |
| a4_imgtxt | electromagnetic | 0.414 (ep18) | 0.414 | 0.359 |
| a5_imgfus | electromagnetic | 0.483 (ep18) | 0.483 | 0.294 |
| a6_txtfus | electromagnetic | 0.470 (ep16) | 0.464 | 0.206 |

## 基準値（同一プロトコル）

| ドメイン | 条件3 / unfrozen（適応） | 足切り閾値（×0.9） | 条件A（exp_019 適応 / ZCOCO） | A2（exp_018 適応 / ZCOCO） |
|---|---|---|---|---|
| underwater | 0.359 | 0.3231 | 0.355 / 0.411 | 0.185 / 0.489 |
| videogames | 0.782 | 0.7038 | 0.777 / 0.328 | 0.315 / 0.443 |
| electromagnetic | 0.502 | 0.4518 | 0.485 / 0.279 | 0.215 / 0.443 |

## 足切り判定（全3ドメインで閾値以上の条件のみ通過）

適応 best を閾値と機械的に照合。

| 条件 | underwater (≥0.3231) | videogames (≥0.7038) | electromagnetic (≥0.4518) | 全通過 |
|---|---|---|---|---|
| a1_img | 0.286 ✗ | 0.187 ✗ | 0.315 ✗ | ✗ |
| a3_fus | 0.323 ✗（境界） | 0.747 ✓ | 0.461 ✓ | ✗ |
| a4_imgtxt | 0.302 ✗ | 0.696 ✗ | 0.414 ✗ | ✗ |
| a5_imgfus | 0.343 ✓ | 0.757 ✓ | 0.483 ✓ | **✓** |
| a6_txtfus | 0.334 ✓ | 0.758 ✓ | 0.470 ✓ | **✓** |

- **全3ドメイン通過: a5_imgfus（画像+融合）、a6_txtfus（テキスト+融合）。**
- 境界事例: a3_fus の underwater は 0.323 で閾値 0.3231 を 0.0001 下回る（丸め前も 0.323）。videogames・electromagnetic は通過。
