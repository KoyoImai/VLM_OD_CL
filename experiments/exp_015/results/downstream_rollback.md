# exp_015 結果：下流（Query Selection＋Decoder＋Head）ロールバック — 実 mAP

作成日: 2026-07-01。設計: [[../design]]。生成: `make_hybrid_downstream.py`（学習なし・重み合成のみ、30 hybrid）。
評価: ZCOCO=`eval_base_coco.py` / 適応=自ドメインFT config、いずれも (800,1333) keep_ratio・4GPU。
全60評価で **mAP 全数取得（欠損 0）**。適応で終了時 segfault の FAIL 1件（documents/qsel）が出たが、
mAP 算出後のクラッシュで値はログから回収済み。ZCOCO は FAIL 0。

## ハイブリッド定義（論文表記）
θ1_d^U の下流を θ0 に置換（上流 backbone/neck/language_model/text_feat_map/encoder は unfrozen のまま）。
- **whole**(229) = 下記4つの和
- **decoder** = Cross-Modality Decoder（`decoder.*` 174）
- **cls** = Contrastive classification head（`bbox_head.cls_branches.*` 7）
- **reg** = Box regression head（`bbox_head.reg_branches.*` 42）
- **qsel** = Language-guided Query Selection（`memory_trans_fc/norm`+`level_embed`+`query_embedding` 6）
- 除外: `dn_query_generator.label_embedding`（学習専用・評価不使用・shape不一致）。

## ① ZCOCO（COCO 保持 mAP。θ0=0.504）

| ドメイン | θ0 | frozen | unfrozen | **whole** | **decoder** | **cls** | **reg** | **qsel** |
|---|---|---|---|---|---|---|---|---|
| underwater | 0.504 | 0.389 | 0.407 | 0.420 | 0.417 | 0.407 | 0.408 | 0.409 |
| aerial | 0.504 | 0.318 | 0.369 | 0.399 | 0.396 | 0.369 | 0.370 | 0.371 |
| microscopic | 0.504 | 0.327 | 0.288 | 0.344 | 0.339 | 0.288 | 0.291 | 0.292 |
| videogames | 0.504 | 0.324 | 0.315 | 0.352 | 0.345 | 0.315 | 0.316 | 0.319 |
| documents | 0.504 | 0.275 | 0.211 | 0.248 | 0.237 | 0.211 | 0.216 | 0.222 |
| electromagnetic | 0.504 | 0.331 | 0.269 | 0.309 | 0.302 | 0.269 | 0.269 | 0.277 |

## ② 適応（自ドメイン valid mAP）

| ドメイン | frozen | unfrozen | **whole** | **decoder** | **cls** | **reg** | **qsel** |
|---|---|---|---|---|---|---|---|
| underwater | 0.337 | 0.359 | 0.351 | 0.352 | 0.359 | 0.358 | 0.358 |
| aerial | 0.468 | 0.486 | 0.454 | 0.458 | 0.486 | 0.485 | 0.485 |
| microscopic | 0.499 | 0.538 | 0.505 | 0.508 | 0.538 | 0.538 | 0.538 |
| videogames | 0.719 | 0.782 | 0.748 | 0.749 | 0.782 | 0.781 | 0.783 |
| documents | 0.478 | 0.542 | 0.484 | 0.494 | 0.542 | 0.522 | 0.523 |
| electromagnetic | 0.454 | 0.502 | 0.467 | 0.472 | 0.502 | 0.500 | 0.502 |

## 所見（実 mAP で読む）

### ZCOCO（忘却）
1. **下流内では Cross-Modality Decoder だけが忘却に効く**。whole と decoder はほぼ同値
   （例 microscopic whole 0.344 / decoder 0.339、documents 0.248 / 0.237）＝下流の回復は**ほぼ全て decoder 由来**。
2. **cls / reg / qsel は unfrozen とほぼ同じ**（cls は完全に横ばい：underwater 0.407=unf、microscopic 0.288=unf）。
   → **分類ヘッド（ContrastiveEmbed の log_scale/bias）・box 回帰・クエリ選択の適応は、COCO 忘却にほぼ無関与**。
   exp_004 で候補に挙げた「reg ドリフト・log_scale/bias」は、実 mAP では**忘却源ではない**（reg 回復は最大 +0.005）。
3. **下流全体の回復は encoder より遥かに小さい**。whole の unfrozen からの回復は +0.013〜+0.056 に留まる
   （documents 0.211→0.248）。exp_014 の encoder は 0.211→0.445（+0.234）だった。
   → **忘却源は encoder（特に画像 self-attn）＞＞ decoder ＞ head/qsel≈0**。

### 適応
1. **cls / reg / qsel を θ0 に戻しても適応はほぼ不変**（cls は unfrozen と同値：videogames 0.782=unf）。
   → 分類ヘッド・box 回帰・クエリ選択の学習は、ドメイン適応にもほぼ不要。
2. **decoder / whole を戻すと適応は緩やかに低下**（videogames 0.782→0.749、documents 0.542→0.484）。
   ただし encoder 画像層の毀損（videogames 0.782→0.337）に比べ遥かに軽微。
   → decoder の適応寄与は中程度。head/qsel はほぼ無寄与。

## exp_012〜015 統合：全909 params の忘却・適応マップ（実 mAP の水準）
| 段 | モジュール（論文表記） | ZCOCO 忘却への効き | 適応への効き |
|---|---|---|---|
| exp_012/013 | Image Backbone(Swin) | 小〜中 | 中 |
| exp_013 | neck / text_feat_map（射影） | ほぼ無 | ほぼ無 |
| exp_012/013 | Text Backbone(BERT) | 軽微 | 訓練不要 |
| **exp_014** | **Feature Enhancer image（画像 deform self-attn）** | **最大** | **最大（保持と競合）** |
| exp_014 | Feature Enhancer fusion / text | 小〜中／小 | ほぼ無 |
| **exp_015** | **Cross-Modality Decoder** | **中（下流で最大）** | 中 |
| exp_015 | Contrastive cls / Box reg / Query Selection | ほぼ無 | ほぼ無 |

→ **忘却の主座は Feature Enhancer の画像自己注意、次いで Cross-Modality Decoder**。
分類・回帰ヘッド、クエリ選択、各種射影、テキスト系は、実 mAP の水準で忘却にも適応にもほぼ効かない。
継続学習の忘却緩和は **encoder 画像 self-attn ＋ decoder** に絞れば足りる、という設計指針が全モデルで確定した。

## 補足（評価の健全性）
- 適応 FAIL 1件（documents/qsel）は mAP 算出後の終了時 segfault（exitcode −11）で値は正常記録済み。再評価不要。

## 成果物
- ハイブリッド: `experiments/exp_015/hybrids/{domain}_{whole|decoder|cls|reg|qsel}_theta0.pth`（30本）
- 評価: `experiments/exp_015/{domain}_{group}_theta0_{coco|adapt}/`（60本）
- ログ: `zcoco_eval.log` / `adapt_eval.log` / `make_hybrid.log`

## 関連
- 前段: [[../../exp_012/results/swap1_swin_to_theta0]] / [[../../exp_013/results/feature_rollback_imgfeat_vs_txtfeat]] / [[../../exp_014/results/feature_enhancer_rollback]]
- アーキ（Query Selection §2.5, Decoder §2.6, Head §2.7, ContrastiveEmbed §3）: [[../../../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]]
- 忘却源候補（reg ドリフト・log_scale/bias）→本実験で否定: [[../../exp_004/design]]
