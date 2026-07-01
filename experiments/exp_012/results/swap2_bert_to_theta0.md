# exp_012 第2 swap 結果：unfrozen の BERT を θ0 に戻す（Swin と対称）

作成日: 2026-06-30。設計: [[../design.md]]。生成: `make_hybrid_bert_theta0.py`（学習なし・重み合成のみ）。
評価: ZCOCO=`eval_base_coco.py` / 適応=自ドメインFT config、いずれも (800,1333) keep_ratio・4GPU。

## ハイブリッド H^BERT_d の定義
**H^BERT_d = θ1_d^U（unfrozen）の `language_model.*`（BERT, 共通197 params）だけを θ0 に置換**。
**Swin(`backbone.`) は unfrozen のまま（ドメイン適応済み）**、neck/enc/dec/head も unfrozen。
- θ0 は language_model.* が198、unfrozen は197（θ0のみ `...embeddings.position_ids`＝定数バッファ）。共通197を置換。
- swap 検証済み: 197 params すべて unfrozen で θ0 と異なる（BERT は lr_mult=0.1 で完全に学習されていた）→ swap は非自明。

## ① ZCOCO（COCO 保持）— Swin との比較

| ドメイン | θ0 | frozen | unfrozen | H^Swin | H^Swin−unf | H^BERT | H^BERT−unf |
|---|---|---|---|---|---|---|---|
| underwater | 0.504 | 0.389 | 0.407 | 0.432 | +0.025 | 0.408 | +0.001 |
| aerial | 0.504 | 0.318 | 0.369 | 0.404 | +0.035 | 0.377 | +0.008 |
| microscopic | 0.504 | 0.327 | 0.288 | 0.323 | +0.035 | 0.298 | +0.010 |
| videogames | 0.504 | 0.324 | 0.315 | 0.345 | +0.030 | 0.324 | +0.009 |
| documents | 0.504 | 0.275 | 0.211 | 0.258 | +0.047 | 0.219 | +0.008 |
| electromagnetic | 0.504 | 0.331 | 0.269 | 0.321 | +0.052 | 0.294 | +0.025 |

**所見**: 低下4ドメインで BERT 回復も**正（クリーン）だが Swin の 1/3〜1/5**。
→ **COCO 忘却への backbone 寄与は Swin(画像)が支配的、BERT(テキスト)は軽微**。両者とも COCO を害するが Swin ≫ BERT。

## ② 適応（自ドメイン）— BERT 戻しは壊滅、ただし共適応の産物

| ドメイン | frozen | unfrozen | H^Swin | H^BERT | unf−H^BERT |
|---|---|---|---|---|---|
| underwater | 0.337 | 0.359 | 0.287 | 0.220 | 0.139 |
| aerial | 0.468 | 0.486 | 0.438 | 0.398 | 0.088 |
| microscopic | 0.499 | 0.538 | 0.420 | 0.218 | 0.320 |
| videogames | 0.719 | 0.782 | 0.667 | 0.182 | 0.600 |
| documents | 0.478 | 0.542 | 0.312 | 0.216 | 0.326 |
| electromagnetic | 0.454 | 0.502 | 0.411 | 0.189 | 0.313 |

**所見**: BERT を戻すと適応が **frozen すら大きく下回り壊滅**（videogames 0.782→0.182）。
だが「BERT が適応知識を握る」ことの証拠ではない：

- **frozen は BERT=θ0 のまま適応 0.337〜0.719 を達成** → θ0 BERT でも適応は十分可能。BERT 訓練は適応に不要。
- H^BERT が壊滅するのは、**検出経路（分類）が「ドメイン調整後 BERT のテキスト埋め込み」に強く整合**しているため。
  MM-GDINO の分類は **query·text 内積（ContrastiveEmbed・クラス固有重みなし）**ゆえ、BERT を θ0 に戻すと
  照合先テキストベクトルが変わり整合が壊れて分類が崩壊する。
- → **この壊滅は共適応（アーキ結合）の産物であって BERT 適応の重要性ではない**。videogames(87クラス)で最大なのは
  テキスト依存が最も強いため。

## まとめ（画像 vs テキスト backbone）

| | 忘却(ZCOCO)への寄与 | 適応への寄与 |
|---|---|---|
| Swin(画像) | 主（clean 回復 +0.025〜0.052） | 中（swap drop は不整合込み） |
| BERT(テキスト) | 軽微（clean 回復 +0.001〜0.025） | 訓練は不要（frozen で十分）。swap 壊滅は contrastive 結合の産物 |
| 検出経路 | 依然 最大の主役 | 最大の主役 |

## 重要な訂正（swap1 の誤ラベル）
`swap1_swin_to_theta0.md` で「unfrozen−frozen＝Swin 寄与」と記したが**誤り**。frozen は
**Swin と BERT の両方**を凍結（lr_mult=0.0）しているため、**unfrozen−frozen は Swin+BERT 合算の寄与**。
Swin 単独へは既存モデルからは分離不可（要「Swin だけ unfrozen」run）。swap1 側を訂正済み。

## 成果物
- ハイブリッド: `hybrids/{domain}_bert_theta0.pth`（6本）
- 評価: `{domain}_bert_theta0_coco/`（ZCOCO）, `{domain}_bert_theta0_adapt/`（適応）

## 関連
- 第1 swap（Swin）: [[swap1_swin_to_theta0]]
- アーキ（ContrastiveEmbed=query·text）: [[../../../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]]
- 現象の出所: [[../../exp_011/results/frozen_vs_unfrozen_6domains]]
