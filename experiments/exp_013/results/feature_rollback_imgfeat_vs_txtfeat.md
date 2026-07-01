# exp_013 結果：特徴抽出部（投影アダプタ含む）のロールバック — 画像 vs テキスト

作成日: 2026-07-01。設計: [[../design]]。生成: `make_hybrid_rollback.py`（学習なし・重み合成のみ）。
評価: ZCOCO=`eval_base_coco.py` / 適応=自ドメインFT config、いずれも (800,1333) keep_ratio・4GPU。
全24評価（12 hybrid × ZCOCO/適応）**正常終了・FAIL 0**。

## ハイブリッド定義（exp_012 の backbone 単体を投影アダプタまで拡張）
- **H^imgfeat_d** = θ1_d^U の `backbone.*`(Swin 187) ＋ `neck.*`(ChannelMapper 16) = **203 params** を θ0 に置換。
- **H^txtfeat_d** = θ1_d^U の `language_model.*`(BERT 197) ＋ `text_feat_map.*`(Linear 768→256, 2) = **199 params** を θ0 に置換。
- テキスト側/画像側・検出経路（encoder/decoder/bbox_head）は unfrozen のまま。
- **exp_012 との関係**: imgfeat = H^Swin ＋ neck も θ0 化 ／ txtfeat = H^BERT ＋ text_feat_map も θ0 化。
  → **差分 = 投影アダプタ（neck / text_feat_map）の限界寄与**（交差検証で確認済み：imgfeat と H^Swin の差は neck.* のみ）。

## ① ZCOCO（COCO 保持 mAP）

| ドメイン | θ0 | frozen | unfrozen | H^Swin<br>(bb) | **H^imgfeat**<br>(bb+neck) | H^BERT<br>(lm) | **H^txtfeat**<br>(lm+tfm) |
|---|---|---|---|---|---|---|---|
| underwater | 0.504 | 0.389 | 0.407 | 0.432 | **0.435** | 0.408 | **0.408** |
| aerial | 0.504 | 0.318 | 0.369 | 0.404 | **0.407** | 0.377 | **0.379** |
| microscopic | 0.504 | 0.327 | 0.288 | 0.323 | **0.325** | 0.298 | **0.300** |
| videogames | 0.504 | 0.324 | 0.315 | 0.345 | **0.349** | 0.324 | **0.326** |
| documents | 0.504 | 0.275 | 0.211 | 0.258 | **0.265** | 0.219 | **0.220** |
| electromagnetic | 0.504 | 0.331 | 0.269 | 0.321 | **0.327** | 0.294 | **0.300** |

### 回復量（vs unfrozen）と投影アダプタの限界寄与

| ドメイン | 画像側<br>H^imgfeat−unf | うち neck 限界<br>(imgfeat−H^Swin) | テキスト側<br>H^txtfeat−unf | うち tfm 限界<br>(txtfeat−H^BERT) |
|---|---|---|---|---|
| underwater | +0.028 | +0.003 | +0.001 | +0.000 |
| aerial | +0.038 | +0.003 | +0.010 | +0.002 |
| microscopic | +0.037 | +0.002 | +0.012 | +0.002 |
| videogames | +0.034 | +0.004 | +0.011 | +0.002 |
| documents | +0.054 | +0.007 | +0.009 | +0.001 |
| electromagnetic | +0.058 | +0.006 | +0.031 | +0.006 |

### ZCOCO 所見（符号ルール：低下4ドメイン micro/vg/doc/em が主証拠）
1. **画像特徴抽出部の回復 ≫ テキスト特徴抽出部**（imgfeat +0.028〜+0.058 vs txtfeat +0.001〜+0.031）。
   exp_012（Swin ≫ BERT）の結論が投影アダプタを含めても**不変**＝**COCO 忘却は画像ストリームが支配的**。
2. **neck の限界寄与は小さいが一貫して正**（+0.002〜+0.007）。**text_feat_map も同様に微小で正**（0〜+0.006）。
   → 投影アダプタの適応も COCO を僅かに害する（backbone/BERT と同方向）が、寄与は backbone 本体の **1割前後**。
3. **限界寄与は遠ドメイン（documents/electromagnetic）で最大**（neck: doc +0.007, em +0.006 ／ tfm: em +0.006）。
   backbone の傾向（遠ドメインで COCO を強く害する）と整合。
4. **回復は依然 partial**（imgfeat 最大 0.435 ≪ θ0 0.504）。特徴抽出部を丸ごと戻しても残る低下は
   unfrozen のまま残した**共有検出経路（encoder/decoder/head）**由来 ＝ exp_012 と同じ結論。

## ② 適応（自ドメイン valid mAP）

| ドメイン | frozen | unfrozen | H^Swin | **H^imgfeat** | H^BERT | **H^txtfeat** |
|---|---|---|---|---|---|---|
| underwater | 0.337 | 0.359 | 0.287 | **0.263** | 0.220 | **0.214** |
| aerial | 0.468 | 0.486 | 0.438 | **0.421** | 0.398 | **0.386** |
| microscopic | 0.499 | 0.538 | 0.420 | **0.385** | 0.218 | **0.202** |
| videogames | 0.719 | 0.782 | 0.667 | **0.570** | 0.182 | **0.173** |
| documents | 0.478 | 0.542 | 0.312 | **0.275** | 0.216 | **0.201** |
| electromagnetic | 0.454 | 0.502 | 0.411 | **0.373** | 0.189 | **0.177** |

### 適応 所見（符号は非クリーン＝上限として読む）
1. **txtfeat 適応は壊滅**（0.17〜0.39、frozen すら大きく下回る）。design.md の予想どおり、
   text_feat_map＋BERT を θ0 に戻すと **ContrastiveEmbed（query·text 内積）が照合する
   テキスト埋め込みが変わり、ドメイン調整済み検出経路との整合が崩壊**する（exp_012 H^BERT と同機序の産物）。
   videogames(87クラス) が最小 0.173 ＝テキスト依存が最強。**「テキスト特徴抽出が適応知識を握る」証拠ではない**。
2. **投影アダプタ追加ロールバックで適応はさらに低下**（imgfeat−H^Swin: −0.017〜−0.097 ／ txtfeat−H^BERT: −0.006〜−0.016）。
   ただし ZCOCO と異なり適応では「アダプタ適応寄与の消失」と「共適応の不整合」が同符号のため、
   この落差は**寄与の上限**（不整合込み）であり単独では帰属不能。
3. **クリーンな適応寄与は frozen vs unfrozen で読む**（exp_012 の結論を踏襲）：適応の大半は検出経路で達成、
   特徴抽出部の学習は top-up。

## まとめ（exp_012 + exp_013：画像 vs テキスト特徴抽出ストリーム）

| モジュール | 忘却(ZCOCO)への寄与（clean 回復） | 適応への寄与 |
|---|---|---|
| Swin(画像backbone) | 主 +0.025〜0.052 | 中（swap drop は不整合込み） |
| neck(画像投影) | 微小 +0.002〜0.007（遠ドメインで大） | 小（top-up） |
| BERT(テキストbackbone) | 軽微 +0.001〜0.025 | 訓練不要（frozen で十分）。swap 壊滅は contrastive 産物 |
| text_feat_map(テキスト投影) | 微小 0〜0.006 | 同上 |
| 検出経路(encoder/decoder/head) | **依然 最大の主役** | **最大の主役** |

→ **画像特徴抽出ストリーム（Swin+neck）が COCO 忘却の主因、テキスト側（BERT+text_feat_map）は軽微**。
投影アダプタは backbone 本体に約1割を上乗せする補助部品。**忘却・適応いずれも主役は共有検出経路**（exp_012/exp_004 と一貫）。
次段 exp_014 で検出経路の入口＝Feature Enhancer(encoder) を切り分ける。

## 追加考察（2026-07-01）：ドリフト分析 — 忘却は「どこが動いたか」で決まる

特徴抽出部の各モジュールが学習でどれだけ動いたか、相対ドリフト ‖θ1_d^U−θ0‖/‖θ0‖ を測定
（**整数バッファ `relative_position_index`/`position_ids` は分母から除外**。含めると backbone が
巨大な整数バッファに希釈され誤って ≈0 に見える。要注意点）。

### 相対ドリフト（浮動小数キーのみ）
| ドメイン | backbone(Swin) | neck | language_model(BERT) | text_feat_map |
|---|---|---|---|---|
| underwater | 0.017 | 0.064 | 0.015 | 0.039 |
| aerial | 0.012 | 0.042 | 0.010 | 0.028 |
| microscopic | 0.015 | 0.052 | 0.014 | 0.033 |
| videogames | 0.013 | 0.050 | 0.014 | 0.032 |
| documents | 0.020 | 0.075 | 0.020 | 0.045 |
| electromagnetic | 0.024 | 0.082 | 0.021 | 0.054 |

**観察**: **投影アダプタ（neck 0.04〜0.08 / text_feat_map 0.03〜0.05）は backbone/BERT（0.01〜0.02）より
大きく動く**。小さく可塑的な射影層は速く適応し、巨大な事前学習 backbone はゆっくりしか動かない。

### 回復量 / ドリフト（動いた量あたりの ZCOCO 忘却寄与、全モジュール統合：exp_012/013/014）
| モジュール | 相対ドリフト | ZCOCO回復 | 回復/ドリフト | 区分 |
|---|---|---|---|---|
| **backbone Swin（画像 self-attn）** | 0.017 | 0.037 | **2.20** | ★load-bearing |
| **encoder image（画像 deformable self-attn）** | 0.068 | 0.129 | **1.90** | ★load-bearing |
| encoder fusion（cross-modal） | 0.068 | 0.051 | 0.74 | inert |
| language_model BERT | 0.016 | 0.010 | 0.65 | inert |
| encoder text（text self-attn） | 0.045 | 0.025 | 0.55 | inert |
| **neck（画像 projection）** | 0.061 | 0.004 | **0.07** | inert |
| **text_feat_map（text projection）** | 0.038 | 0.002 | **0.06** | inert |

### 解釈（本プロジェクトの中核的知見）
1. **忘却は「動いた量」ではなく「動いた場所」で決まる**。回復/ドリフトは明快な2クラスタに分かれる：
   - **★load-bearing（≈2.0）= 画像の空間的自己注意**：backbone Swin と encoder image。**この2つだけが**
     動いた量あたりの COCO 破壊力が突出。
   - **inert（≤0.74）= それ以外全て**：cross-modal fusion・BERT・text self-attn・そして射影層（neck/text_feat_map）。
2. **neck / text_feat_map は最も動くのに最も無害**（回復/ドリフト 0.06〜0.07 で全モジュール最小）。
   1×1 conv/Linear の**チャネル射影は空間表現を壊さない**ため、大きく再パラメータ化されても COCO を害さない。
3. → exp_014 の結論（画像 self-attn が load-bearing）を上流にも拡張。**忘却源は "画像の空間的注意" に一貫して局在**し、
   射影・テキスト系・cross-modal 融合は「動いても無害」。**継続学習の忘却緩和は画像自己注意
   （`backbone.stages.*.attn` と `encoder.layers.*.self_attn`）に限定投下すればよい**という設計指針を強化する。

## 成果物
- ハイブリッド: `experiments/exp_013/hybrids/{domain}_{imgfeat|txtfeat}_theta0.pth`（12本）
- 評価: `experiments/exp_013/{domain}_{imgfeat|txtfeat}_theta0_{coco|adapt}/`（24本）
- ログ: `zcoco_eval.log` / `adapt_eval.log` / `make_hybrid.log`

## 関連
- 前段（backbone単体）: [[../../exp_012/results/swap1_swin_to_theta0]] / [[../../exp_012/results/swap2_bert_to_theta0]]
- 次段（Feature Enhancer=encoder）: [[../../exp_014/design]]
- アーキ（neck=ChannelMapper, text_feat_map, ContrastiveEmbed=query·text）: [[../../../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]]
- 現象の出所: [[../../exp_011/results/frozen_vs_unfrozen_6domains]]
