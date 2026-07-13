# 全モジュール ロールバック 統合結果（実 mAP）— exp_012〜015

各 unfrozen モデル θ1_d^U の**特定モジュールだけを θ0（事前学習）に戻したハイブリッド**の実 mAP を、
モジュール横断で一覧化したもの。表形式は [[exp_015/results/downstream_rollback]] の ①ZCOCO / ②適応 に準拠。
評価はすべて ZCOCO=`eval_base_coco.py` / 適応=自ドメインFT config、(800,1333) keep_ratio・4GPU。

## 列（ロールバック対象モジュール）と出典
| 列 | 論文表記 / 内容 | 接頭辞 | params | 出典 |
|---|---|---|---|---|
| θ0 | 事前学習（ロールバックの上限基準） | 全体 | — | — |
| frozen | backbone/BERT 凍結で学習した θ1^F | — | — | exp_005/010/011 |
| unfrozen | 全層学習した θ1^U（ロールバックの起点） | — | — | exp_010/011 |
| **Swin** | Image Backbone のみ | `backbone.` | 187 | exp_012 |
| **BERT** | Text Backbone のみ | `language_model.` | 197 | exp_012 |
| imgfeat | Swin＋neck（画像特徴抽出部） | `backbone.`+`neck.` | 203 | exp_013 |
| txtfeat | BERT＋text_feat_map（テキスト特徴抽出部） | `language_model.`+`text_feat_map.` | 199 | exp_013 |
| **enc:image** | Feature Enhancer 画像 self-attn | `encoder.layers.` | 96 | exp_014 |
| **enc:text** | Feature Enhancer テキスト self-attn | `encoder.text_layers.` | 72 | exp_014 |
| **enc:fusion** | Feature Enhancer cross-modal | `encoder.fusion_layers.` | 108 | exp_014 |
| **enc:whole** | Feature Enhancer 全体 | `encoder.` | 276 | exp_014 |
| **decoder** | Cross-Modality Decoder 全体 | `decoder.` | 174 | exp_015 |
| cls | Contrastive classification head | `bbox_head.cls_branches.` | 7 | exp_015 |
| reg | Box regression head | `bbox_head.reg_branches.` | 42 | exp_015 |
| qsel | Language-guided Query Selection | pre_decoder 射影・埋込 | 6 | exp_015 |
| down:whole | 下流全体（decoder+head+qsel） | — | 229 | exp_015 |

> 注: neck 単独・text_feat_map 単独は未分離（imgfeat/txtfeat の複合として計測）。
> Cross-Modality Decoder の内部4分解（self/text/img/ffn）は exp_016 で計測中（完了後に追記）。

---

## ① ZCOCO（COCO 保持 mAP。θ0=0.504、高いほど忘却が少ない）

| ドメイン | θ0 | frozen | unfrozen | Swin | BERT | imgfeat | txtfeat | enc:image | enc:text | enc:fusion | **enc:whole** | decoder | cls | reg | qsel | **down:whole** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| underwater | 0.504 | 0.389 | 0.407 | 0.432 | 0.408 | 0.435 | 0.408 | 0.460 | 0.410 | 0.426 | **0.473** | 0.417 | 0.407 | 0.408 | 0.409 | **0.420** |
| aerial | 0.504 | 0.318 | 0.369 | 0.404 | 0.377 | 0.407 | 0.379 | 0.461 | 0.392 | 0.411 | **0.475** | 0.396 | 0.369 | 0.370 | 0.371 | **0.399** |
| microscopic | 0.504 | 0.327 | 0.288 | 0.323 | 0.298 | 0.325 | 0.300 | 0.438 | 0.325 | 0.340 | **0.466** | 0.339 | 0.288 | 0.291 | 0.292 | **0.344** |
| videogames | 0.504 | 0.324 | 0.315 | 0.345 | 0.324 | 0.349 | 0.326 | 0.453 | 0.338 | 0.371 | **0.471** | 0.345 | 0.315 | 0.316 | 0.319 | **0.352** |
| documents | 0.504 | 0.275 | 0.211 | 0.258 | 0.219 | 0.265 | 0.220 | 0.413 | 0.246 | 0.274 | **0.445** | 0.237 | 0.211 | 0.216 | 0.222 | **0.248** |
| electromagnetic | 0.504 | 0.331 | 0.269 | 0.321 | 0.294 | 0.327 | 0.300 | 0.405 | 0.295 | 0.340 | **0.453** | 0.302 | 0.269 | 0.269 | 0.277 | **0.309** |

**読み方**: unfrozen より高い＝そのモジュールを戻すと COCO が回復＝そのモジュールの適応が忘却の一因（符号ルールでクリーン）。

---

## ② 適応（自ドメイン valid mAP。高いほど適応良好）

| ドメイン | frozen | unfrozen | Swin | BERT | imgfeat | txtfeat | enc:image | enc:text | enc:fusion | **enc:whole** | decoder | cls | reg | qsel | **down:whole** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| underwater | 0.337 | 0.359 | 0.287 | 0.220 | 0.263 | 0.214 | 0.298 | 0.341 | 0.342 | **0.225** | 0.352 | 0.359 | 0.358 | 0.358 | **0.351** |
| aerial | 0.468 | 0.486 | 0.438 | 0.398 | 0.421 | 0.386 | 0.381 | 0.465 | 0.464 | **0.295** | 0.458 | 0.486 | 0.485 | 0.485 | **0.454** |
| microscopic | 0.499 | 0.538 | 0.420 | 0.218 | 0.385 | 0.202 | 0.369 | 0.528 | 0.520 | **0.290** | 0.508 | 0.538 | 0.538 | 0.538 | **0.505** |
| videogames | 0.719 | 0.782 | 0.667 | 0.182 | 0.570 | 0.173 | 0.337 | 0.768 | 0.762 | **0.196** | 0.749 | 0.782 | 0.781 | 0.783 | **0.748** |
| documents | 0.478 | 0.542 | 0.312 | 0.216 | 0.275 | 0.201 | 0.208 | 0.484 | 0.503 | **0.108** | 0.494 | 0.542 | 0.522 | 0.523 | **0.484** |
| electromagnetic | 0.454 | 0.502 | 0.411 | 0.189 | 0.373 | 0.177 | 0.209 | 0.482 | 0.480 | **0.104** | 0.472 | 0.500 | 0.500 | 0.502 | **0.467** |

**読み方**: unfrozen より低い＝そのモジュールを戻すと適応が落ちる。ただし符号は非クリーン（共適応の不整合込み＝寄与の上限）。クリーンな適応寄与は frozen vs unfrozen で読む。

---

---

## ③ Cross-Modality Decoder 内部4分解（exp_016）
decoder(whole) をさらに内部処理に分解（各処理＋直後 post-norm 同梱）。unfrozen・decoder(whole) を基準に併記。

### ZCOCO（COCO 保持 mAP）
| ドメイン | unfrozen | decoder(whole) | dec:self | dec:x_text | dec:x_img | dec:ffn |
|---|---|---|---|---|---|---|
| underwater | 0.407 | 0.417 | 0.413 | 0.409 | 0.411 | 0.416 |
| aerial | 0.369 | 0.396 | 0.379 | 0.374 | 0.377 | 0.383 |
| microscopic | 0.288 | 0.339 | 0.299 | 0.300 | 0.308 | 0.305 |
| videogames | 0.315 | 0.345 | 0.330 | 0.314 | 0.326 | 0.334 |
| documents | 0.211 | 0.237 | 0.219 | 0.223 | 0.219 | 0.221 |
| electromagnetic | 0.269 | 0.302 | 0.279 | 0.273 | 0.279 | 0.287 |

### 適応（自ドメイン valid mAP）
| ドメイン | unfrozen | decoder(whole) | dec:self | dec:x_text | dec:x_img | dec:ffn |
|---|---|---|---|---|---|---|
| underwater | 0.359 | 0.352 | 0.359 | 0.358 | 0.360 | 0.357 |
| aerial | 0.486 | 0.458 | 0.483 | 0.483 | 0.477 | 0.484 |
| microscopic | 0.538 | 0.508 | 0.536 | 0.537 | 0.533 | 0.535 |
| videogames | 0.782 | 0.749 | 0.780 | 0.776 | 0.773 | 0.779 |
| documents | 0.542 | 0.494 | 0.526 | 0.524 | 0.521 | 0.520 |
| electromagnetic | 0.502 | 0.472 | 0.499 | 0.493 | 0.493 | 0.492 |

> decoder 内部は self/x_text/x_img/ffn いずれも小さく均等（各処理の ZCOCO 回復 +0.00〜+0.02、最大は dec:ffn が6中4ドメイン）。
> **画像 cross-attn は突出せず**、encoder の画像 self-attn が支配的だったのとは対照的。詳細: [[exp_016/results/decoder_internal_rollback]]。

---

## ④ neckTfm-only（射影アダプタ neck＋text_feat_map の18 params のみ学習。exp_017）
ロールバック（重み合成）ではなく**実学習**。画像/テキスト backbone・Feature Enhancer 以降を全凍結し、射影のみ適応。
frozen / unfrozen / θ0 と同基準（ZCOCO=`eval_base_coco.py`・適応=自ドメインFT config・(800,1333)・4GPU）で対比。

| ドメイン | θ0(ZCOCO) | ZCOCO: frozen | unfrozen | **neckTfm** | ‖ 適応: frozen | unfrozen | **neckTfm** | 対unfrozen |
|---|---|---|---|---|---|---|---|---|
| underwater | 0.504 | 0.389 | 0.407 | **0.465** | ‖ 0.337 | 0.359 | **0.208** | −0.151 |
| aerial | 0.504 | 0.318 | 0.369 | **0.470** | ‖ 0.468 | 0.486 | **0.350** | −0.136 |
| microscopic | 0.504 | 0.327 | 0.288 | **0.384** | ‖ 0.499 | 0.538 | **0.272** | −0.266 |
| videogames | 0.504 | 0.324 | 0.315 | **0.462** | ‖ 0.719 | 0.782 | **0.113** | −0.669 |
| documents | 0.504 | 0.275 | 0.211 | **0.486** | ‖ 0.478 | 0.542 | **0.196** | −0.346 |
| electromagnetic | 0.504 | 0.331 | 0.269 | **0.447** | ‖ 0.454 | 0.502 | **0.222** | −0.280 |

> **忘却はほぼ完全に防げる**（neckTfm ZCOCO 0.384〜0.486 で全ドメイン frozen/unfrozen を上回り θ0=0.504 に肉薄）。
> **代償として適応はほぼ捨てる**（unfrozen比 −0.14〜−0.67）。クラス数の多い/遠いドメイン（videogames, microscopic, documents, electromagnetic）ほど毀損大。
> ＝検出経路（Feature Enhancer 以降）を凍結すると忘却は止まるが遠いドメインへ適応できない、という**トレードオフ点**を1つ与える。詳細: [[exp_017/design]]。

## ⑤ ZiRa 再現（exp_020。RDB を neck＋text_feat_map に並列挿入、ZiL λ=0.1、本体全凍結）

frozen / unfrozen / neckTfm と同基準（ZCOCO=`eval_base_coco.py` 同等・適応=自ドメインFT config・seed=0・4GPU・20ep）。
詳細と対照 run（公式ハイパラ）: [[exp_020/results/zira_reproduction]]。

| ドメイン | θ0(ZCOCO) | ZCOCO: neckTfm | **ZiRa** | ‖ 適応: neckTfm | **ZiRa** | 対neckTfm |
|---|---|---|---|---|---|---|
| underwater | 0.504 | 0.465 | **0.436** | ‖ 0.208 | **0.183** | −0.025 |
| aerial | 0.504 | 0.470 | **0.473** | ‖ 0.350 | **0.297** | −0.053 |
| microscopic | 0.504 | 0.384 | **0.482** | ‖ 0.272 | **0.192** | −0.080 |
| videogames | 0.504 | 0.462 | **0.457** | ‖ 0.113 | **0.097** | −0.016 |
| documents | 0.504 | 0.486 | **0.476** | ‖ 0.196 | **0.166** | −0.030 |
| electromagnetic | 0.504 | 0.447 | **0.466** | ‖ 0.222 | **0.183** | −0.039 |

> 既存手法 ZiRa は挿入位置が neckTfm と同じ（学習形態は並列枝＋ZiL）。適応は全ドメインで neckTfm を下回り、
> ZCOCO は θ0 比 −2.2〜−6.8 pt（neckTfm 比 3勝3敗）。公式ハイパラ（2000 iter）では uw 適応 0.111 / ZCOCO 0.491（−1.3 pt、論文水準）。

## 要約（実 mAP で見た各モジュールの位置づけ）
- **忘却（ZCOCO）の最大回復は enc:whole（0.445〜0.475、θ0=0.504 目前）**、次いで enc:image（0.405〜0.460）。
  → COCO 忘却の主座は **Feature Enhancer、特に画像 self-attn**。
- **decoder は中程度**（down:whole ≈ decoder。0.237〜0.417）。**cls/reg/qsel は unfrozen とほぼ同値＝忘却に無関与**。
- **Swin/BERT/imgfeat/txtfeat（上流）は小さな回復**（Swin ≫ BERT、投影 neck/tfm は複合で微増）。
- **適応で最も毀損するのも enc:image / enc:whole**（videogames 0.782→0.337/0.196）＝画像 self-attn は**保持と適応が最も競合**。
  cls/reg/qsel/enc:text/enc:fusion は unfrozen 近傍＝適応にほぼ不要。

→ **忘却源・適応の要はいずれも「画像の空間的自己注意」に局在**（backbone Swin と encoder image）。
Cross-Modality Decoder の画像 cross-attn は突出せず（exp_016）、「画像の空間的注意＝忘却源」は encoder の self-attn に固有と精密化された。
- **構成的検証（exp_017・④）**: 検出経路を全凍結し射影18 params のみ学習すると **ZCOCO≈θ0（忘却ほぼ皆無）だが適応は壊滅**（unfrozen比 −0.14〜−0.67）。
  忘却源（encoder画像self-attn 等）を触らなければ COCO は守れるが、その凍結が適応も同時に殺す＝**保持と適応が同じ経路で競合**するロールバック分析の結論を実学習でも裏付け。

## 関連
- [[exp_012/results/swap1_swin_to_theta0]] / [[exp_012/results/swap2_bert_to_theta0]]
- [[exp_013/results/feature_rollback_imgfeat_vs_txtfeat]]
- [[exp_014/results/feature_enhancer_rollback]] / [[exp_015/results/downstream_rollback]]
- アーキ: [[../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]]
