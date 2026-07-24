# exp_022 実行前検証の記録（param assignment）

実行日 2026-07-15。検証スクリプト: `check_param_assignment.py`（結果 ALL OK）。
手続きは exp_019 と同じ（design.md §5）。

## 検証 1・2・4: param group の実構築（underwater、5 条件）

optimizer の param group を実構築し、lr>0 の集合・実効 lr・帳尻を確認した。

| 条件 | 学習 (M) | 凍結 (M) | 合計 (M) | 学習集合 | 実効 lr |
|---|---|---|---|---|---|
| a1_img | 29.64 | 143.28 | 172.92 | backbone, neck | OK |
| a3_fus | 21.91 | 151.01 | 172.92 | encoder, level_embed | OK |
| a4_imgtxt | 138.73 | 34.19 | 172.92 | backbone, language_model, neck, text_feat_map | OK |
| a5_imgfus | 51.54 | 121.38 | 172.92 | backbone, neck, encoder, level_embed | OK |
| a6_txtfus | 131.00 | 41.92 | 172.92 | language_model, text_feat_map, encoder, level_embed | OK |

- 学習集合は design.md §2 の定義と完全一致（誤凍結・凍結漏れなし）。
- 実効 lr は条件3 と同一: backbone / language_model = 1e-5（lr_mult 0.1）、
  neck / text_feat_map / encoder / level_embed = 1e-4（lr_mult 1.0）、凍結 = 0。
- 学習 + 凍結 = 総パラメータ 172.92 M（全条件で帳尻一致）。
- a5_imgfus の 51.54 M は design.md の 51.55 M と丸め差（29.64+21.91 の丸め）。

## 検証 3: ドメイン間の分割定義の一致

videogames / electromagnetic の 10 config について、`custom_keys` が underwater と
辞書として同一であることを確認した（5 条件すべて一致）。分割の定義は
3 ドメインで共通である。

## 未検証（学習開始時に確認する事項）

- DDP 実行時の挙動（凍結は lr_mult=0.0 方式のため find_unused_parameters は不要。
  exp_017/018/019 と同じ）
- 学習開始直後の loss が正常値であること
