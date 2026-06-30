# exp_010+exp_011 結果まとめ：frozen vs unfrozen（全6ドメイン）

作成日: 2026-06-30。**全て seed=0・同一config系統（標準 finetune config 継承）・発散なし**。
設計: [[../design.md]] / underwater 起点: [[../../exp_010/results/underwater_frozen_vs_unfrozen]]。
アーキ: [[../../../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]]。

## 設定
- 各ドメイン単独FT（事前学習済みモデルから）。能力軸 = **C(適応) と A(事前学習知識保持=ZCOCO)**。B(過去ドメイン保持) は単独学習のため対象外。
- 唯一の差: backbone(Swin)・language_model(BERT) の `lr_mult`。**frozen=0.0 / unfrozen=0.1**（実効1e-5相当）。
- 据え置き: base lr=1e-4 / 実効64 / 20ep[15] / AdamW wd=1e-4 / clip_grad(0.1) / 評価(800,1333) / **seed=0**。
- frozen: underwater=exp_010、他5=exp_005（**いずれも標準config継承・seed=0 を実ログで確認済み**＝同一系統）。
- unfrozen: underwater=exp_010、他5=exp_011。
- ZCOCO: `eval_base_coco.py` による COCO val mAP。

## 結果（適応=自ドメインmAP / ZCOCO=COCO保持mAP）

| ドメイン | frozen 適応 | unfrozen 適応 | Δ適応 | frozen ZCOCO | unfrozen ZCOCO | ΔZCOCO | 判定 |
|---|---|---|---|---|---|---|---|
| underwater | 0.337 | 0.359 | +0.022 | 0.389 | 0.407 | +0.018 | ✅ Pareto改善 |
| aerial | 0.468 | 0.486 | +0.018 | 0.318 | 0.369 | +0.051 | ✅ Pareto改善 |
| microscopic | 0.499 | 0.538 | +0.039 | 0.327 | 0.288 | −0.039 | ⚠️ トレードオフ |
| videogames | 0.719 | 0.782 | +0.063 | 0.324 | 0.315 | −0.009 | ⚠️ ほぼ横ばい/微減 |
| documents | 0.478 | 0.542 | +0.064 | 0.275 | 0.211 | −0.064 | ⚠️ トレードオフ |
| electromagnetic | 0.454 | 0.502 | +0.048 | 0.331 | 0.269 | −0.062 | ⚠️ トレードオフ |

参考: COCO zero-shot（学習前）= 0.504。
best epoch: unfrozen = aerial19/micro18/vg20/doc16/em19、frozen = aerial17/micro20/vg20/doc19/em18。

## 主要な所見

1. **適応(C)は全6ドメインで unfrozen が改善**（+0.018〜+0.064、例外なし）。
   Swin/BERT を 0.1×学習させると、どのドメインでも自ドメイン性能は上がる。

2. **「Pareto改善」は一般化しなかった**。exp_010(underwater, n=1) の「適応も忘却も両方改善」は
   **underwater・aerial の2ドメインのみ**。残り4ドメインは **unfreeze で ZCOCO 悪化＝忘却増**のトレードオフ。
   → 前回の「backbone学習は Pareto改善」仮説は **否定**。実態は **ドメイン依存**。

3. **ドメイン距離との対応（仮説）**。ZCOCO が改善する underwater/aerial は **自然画像系**（COCOに近い）。
   悪化が大きい documents(−0.064)/electromagnetic(−0.062) は **非自然画像系**（COCOから遠い）。
   解釈: **近ドメインは backbone適応がCOCO特徴と両立 → Pareto改善。遠ドメインは backbone適応がCOCO特徴を破壊 → 忘却増**。
   →「適応を backbone に吸収させる」戦略は **ドメイン距離が近いときだけ有効**。

## 手法設計への示唆
- backbone をどこまで動かすかは **ドメイン距離に応じて変えるべき**（一律 unfreeze は遠ドメインで忘却を招く）。
- 遠ドメインでは「適応をどこに吸収させるか」（LoRA挿入箇所・凍結範囲）が忘却制御の鍵になる、という exp_004/exp_010 の示唆と整合。

## 残る検証（要検討）
- ドメイン距離を **定量化**（例: 各ドメイン画像のbackbone特徴とCOCO特徴の分布距離、または zero-shot ZCOCO 低下幅）し、ΔZCOCO との相関を確認。
- 系列学習（B: 過去ドメイン保持）での frozen/unfrozen 比較は本実験の対象外。別途。

## 関連
- [[../design.md]] / [[../../exp_010/results/underwater_frozen_vs_unfrozen]]
- 機序の起点: [[../../exp_004/design.md]] / 能力軸の議論: [[../../exp_009/minutes]]
- frozen基準: underwater=exp_010・他5=exp_005（標準config・seed=0）
