# exp_016 結果：Cross-Modality Decoder 内部処理の4分解ロールバック — 実 mAP

作成日: 2026-07-01。設計: [[../design]]。生成: `make_hybrid_decoder.py`（学習なし・重み合成のみ、24 hybrid）。
評価: ZCOCO=`eval_base_coco.py` / 適応=自ドメインFT config、(800,1333) keep_ratio・4GPU。
全48評価で **mAP 全数取得（欠損 0・FAIL 0）**。

## ハイブリッド定義（各処理＋直後の post-norm 同梱、全6層合計）
- **dec_selfattn**（36）= クエリ↔クエリ self-attn ＋ norms.0
- **dec_crossattn_text**（36）= クエリ→テキスト cross-attn ＋ norms.1
- **dec_crossattn_img**（60）= クエリ→画像 deformable cross-attn ＋ norms.2
- **dec_ffn**（36）= FFN ＋ norms.3
- 4グループ = `decoder.layers.*`（168）。ref_point_head/final_norm(6) は unfrozen 据え置き。
- whole（decoder全体174）は [[../../exp_015/results/downstream_rollback]] の `decoder` 群で取得済み。

## ① ZCOCO（COCO 保持 mAP。θ0=0.504、unfrozen が忘却状態）

| ドメイン | unfrozen | decoder(whole) | dec_selfattn | dec_crossattn_text | dec_crossattn_img | dec_ffn |
|---|---|---|---|---|---|---|
| underwater | 0.407 | 0.417 | 0.413 | 0.409 | 0.411 | 0.416 |
| aerial | 0.369 | 0.396 | 0.379 | 0.374 | 0.377 | 0.383 |
| microscopic | 0.288 | 0.339 | 0.299 | 0.300 | 0.308 | 0.305 |
| videogames | 0.315 | 0.345 | 0.330 | 0.314 | 0.326 | 0.334 |
| documents | 0.211 | 0.237 | 0.219 | 0.223 | 0.219 | 0.221 |
| electromagnetic | 0.269 | 0.302 | 0.279 | 0.273 | 0.279 | 0.287 |

## ② 適応（自ドメイン valid mAP）

| ドメイン | unfrozen | decoder(whole) | dec_selfattn | dec_crossattn_text | dec_crossattn_img | dec_ffn |
|---|---|---|---|---|---|---|
| underwater | 0.359 | 0.352 | 0.359 | 0.358 | 0.360 | 0.357 |
| aerial | 0.486 | 0.458 | 0.483 | 0.483 | 0.477 | 0.484 |
| microscopic | 0.538 | 0.508 | 0.536 | 0.537 | 0.533 | 0.535 |
| videogames | 0.782 | 0.749 | 0.780 | 0.776 | 0.773 | 0.779 |
| documents | 0.542 | 0.494 | 0.526 | 0.524 | 0.521 | 0.520 |
| electromagnetic | 0.502 | 0.472 | 0.499 | 0.493 | 0.493 | 0.492 |

## 所見（実 mAP で読む）— 主仮説は否定

### ZCOCO（忘却）
1. **主仮説（画像 cross-attn が忘却源として突出）は否定**。4処理の ZCOCO 回復（vs unfrozen）は
   いずれも小さく（+0.00〜+0.02）、**特定処理に集中せず 4処理にほぼ均等に分散**する。
   むしろ **dec_ffn が最大回復のドメインが 6中4**（underwater/aerial/videogames/electromagnetic）、
   dec_crossattn_img が最大は microscopic のみ。差はいずれも ~0.01 で拮抗。
2. **4処理の回復の和 ≈ decoder(whole) の回復**（例 microscopic Σ+0.060 vs whole +0.051、
   electromagnetic Σ+0.042 vs +0.033）＝ほぼ加法的で、突出した単独処理は無い。
3. → **encoder では画像 self-attn が忘却源として支配的だったが、decoder の画像 cross-attn は支配的でない**。

### 適応
- 4処理いずれも θ0 ロールバックで適応はほぼ不変（unfrozen 比 −0.00〜−0.02）。
  decoder(whole) の毀損（−0.03〜−0.05）すら、単独処理では現れない＝decoder の適応寄与も特定処理に集中しない。

## 考察：なぜ decoder の画像注意は encoder と違うのか
- **encoder の画像 self-attn** は 4スケールの**全画像トークン**を相互更新し、COCO が依存する空間特徴分布そのものを
  作り替える → 適応で動くと COCO を大きく破壊（忘却の主座）。
- **decoder の画像 cross-attn** は 900クエリが参照点まわりを**疎にサンプルして読むだけ**で、画像表現自体は書き換えない。
  介入範囲が限定的なため、単独ロールバックの効果は小さく分散する。
- → 「忘却源＝画像の空間的注意」は **self-attn over 全特徴マップ（encoder）に固有**であり、
  cross-attn from queries（decoder）には当てはまらない。仮説の適用範囲が精密化された。

## exp_012〜016 の統合的結論
- **COCO 忘却の主座は Feature Enhancer の画像 self-attn**（enc:whole で 0.211→0.445 等）。次点で Cross-Modality Decoder 全体（中程度）。
- **decoder 内部は self/text/img/ffn いずれも小さく均等**＝下流に明確な単一忘却源は無い。
- head(cls/reg)・qsel・投影(neck/text_feat_map)・テキスト系は忘却・適応ともほぼ無関与。
- → 継続学習の忘却緩和は **encoder 画像 self-attn を主対象**とし、decoder は全体を緩く扱えば足りる（内部を細分する利得は小さい）。

## 補足（評価の健全性）
- FAIL 0。全48評価が正常終了（exit 0）。

## 成果物
- ハイブリッド: `experiments/exp_016/hybrids/{domain}_{dec_selfattn|dec_crossattn_text|dec_crossattn_img|dec_ffn}_theta0.pth`（24本）
- 評価: `experiments/exp_016/{domain}_{group}_theta0_{coco|adapt}/`（48本）

## 関連
- 前段: [[../../exp_014/results/feature_enhancer_rollback]]（encoder分解＝画像self-attnが主座）/ [[../../exp_015/results/downstream_rollback]]（decoder=whole）
- 統合表: [[../../module_rollback_summary]]
- アーキ（Decoder §2.6 の3注意＋FFN）: [[../../../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]]
