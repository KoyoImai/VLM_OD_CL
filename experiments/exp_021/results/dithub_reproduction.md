# exp_021 結果記録: DitHub 再現（適応 mAP / ZCOCO mAP）

本ファイルは事実の記録である（CLAUDE.md 行動原理8）。解釈・考察は含めない。
設計は [[../design]]、実装と裁定は [[../implementation_plan]]、手法の正本は
[[../../../papers/DitHub_implementation_notes]]。全 run seed=0。

## 1. 測定条件

- 適応 mAP: 各ドメインの valid（COCO 形式）での bbox_mAP。
  - 標準版: specialization 期（epoch 11〜20）の best epoch を採用（裁定④）。
  - 公式準拠版: 最終 iterate（iter 3000）を採用（公式と同じ）。
- ZCOCO mAP: COCO2017 val のゼロショット bbox_mAP。θ0（事前学習）= 0.504。
  - 学習した LoRA ライブラリのうち、COCO クラス名と一致するクラスのモジュールのみ
    推論時に適用される（名前ベースのクラス選択）。
- 4 GPU 評価、画像スケール (800,1333)。

## 2. DitHub 標準版（20 epoch, warmup10+spec10, lr 1e-4, wd 1e-4）

| ドメイン | 適応 mAP | (best epoch) | ZCOCO mAP | ZCOCO 対 θ0 |
|---|---|---|---|---|
| underwater | 0.220 | ep19 | 0.504 | +0.000 |
| aerial | 0.377 | ep15 | 0.346 | −0.158 |
| videogames | 0.137 | ep14 | 0.193 | −0.311 |
| microscopic | 0.281 | ep12 | 0.504 | +0.000 |
| documents | 0.260 | ep20 | 0.310 | −0.194 |
| electromagnetic | 0.261 | ep20 | 0.286 | −0.218 |

- documents（ep11〜20: 0.256→0.260）と electromagnetic（0.245→0.261）は
  specialization 期の変動が 0.01 前後で、best が ep20 でも近傍は平坦。

## 3. DitHub 公式準拠版（3000 iter, warmup1500+spec1500, batch2, lr 1e-3, wd 1e-2, 1 GPU）

| ドメイン | 適応 mAP | ZCOCO mAP | ZCOCO 対 θ0 |
|---|---|---|---|
| underwater | 0.161 | 0.504 | +0.000 |
| aerial | 0.330 | 0.421 | −0.083 |
| videogames | 0.094 | 0.286 | −0.218 |
| microscopic | 0.191 | 0.504 | +0.000 |
| documents | 0.162 | 0.320 | −0.184 |
| electromagnetic | 0.113 | 0.375 | −0.129 |

## 4. ZCOCO と COCO クラス重複の対応（検証7の事前調査）

| ドメイン | COCO 重複クラス数 | 標準版 ZCOCO | 公式版 ZCOCO |
|---|---|---|---|
| underwater | 0 | 0.504 | 0.504 |
| microscopic | 0 | 0.504 | 0.504 |
| aerial | 1 (cow) | 0.346 | 0.421 |
| documents | 1 (fork) | 0.310 | 0.320 |
| videogames | 3 | 0.193 | 0.286 |
| electromagnetic | 4 | 0.286 | 0.375 |

- 重複 0 の 2 ドメインは両版とも ZCOCO=θ0=0.504（design.md の事前予測どおり）。
- 重複を持つ 4 ドメインのみ ZCOCO が低下。標準版（学習量 約60倍）は公式版より
  低下が大きい（例 videogames 0.193 < 0.286、electromagnetic 0.286 < 0.375）。

## 5. 既存手法・基準線との比較（適応 mAP / ZCOCO mAP）

基準線は experiments/module_rollback_summary.md（frozen / unfrozen / neckTfm）と
exp_020（ZiRa 20ep版）。全て seed=0・同一評価プロトコル。

### 5.1 適応 mAP

| ドメイン | frozen | unfrozen | neckTfm | ZiRa | DitHub標準 | DitHub公式 |
|---|---|---|---|---|---|---|
| underwater | 0.337 | 0.359 | 0.208 | 0.183 | 0.220 | 0.161 |
| aerial | 0.468 | 0.486 | 0.350 | 0.297 | 0.377 | 0.330 |
| videogames | 0.719 | 0.782 | 0.113 | 0.097 | 0.137 | 0.094 |
| microscopic | 0.499 | 0.538 | 0.272 | 0.192 | 0.281 | 0.191 |
| documents | 0.478 | 0.542 | 0.196 | 0.166 | 0.260 | 0.162 |
| electromagnetic | 0.454 | 0.502 | 0.222 | 0.183 | 0.261 | 0.113 |

- DitHub標準 は ZiRa を全 6 ドメインで上回る（+0.037/+0.080/+0.040/+0.089/+0.094/+0.078）。
- DitHub標準 は neckTfm を全 6 ドメインで上回る。
- DitHub標準・ZiRa とも frozen / unfrozen には全ドメインで届かない。

### 5.2 ZCOCO mAP（θ0 = 0.504）

| ドメイン | frozen | unfrozen | neckTfm | ZiRa | DitHub標準 | DitHub公式 |
|---|---|---|---|---|---|---|
| underwater | 0.389 | 0.407 | 0.465 | 0.436 | 0.504 | 0.504 |
| aerial | 0.318 | 0.369 | 0.470 | 0.473 | 0.346 | 0.421 |
| videogames | 0.324 | 0.315 | 0.462 | 0.457 | 0.193 | 0.286 |
| microscopic | 0.327 | 0.288 | 0.384 | 0.482 | 0.504 | 0.504 |
| documents | 0.275 | 0.211 | 0.486 | 0.476 | 0.310 | 0.320 |
| electromagnetic | 0.331 | 0.269 | 0.447 | 0.466 | 0.286 | 0.375 |

- DitHub の ZCOCO は重複 0 の 2 ドメインで最高（0.504）、重複を持つ 4 ドメインでは
  ZiRa / neckTfm を下回る。ZiRa は全ドメインで 0.436〜0.482 の狭い範囲。

## 6. 成果物の所在

- 標準版: `experiments/exp_021/{domain}_dithub_work_dir`（学習）,
  `{domain}_dithub_zcoco`（ZCOCO 評価）
- 公式準拠版: `{domain}_dithub_official_work_dir`, `{domain}_dithub_official_zcoco`
- 実装検証: `check_dithub_setup.py`（10/10）, `merge_dithub.py`（式4 の実モデル往復）
