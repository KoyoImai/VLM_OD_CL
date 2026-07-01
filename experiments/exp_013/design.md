# exp_013: 特徴抽出部（投影アダプタ含む）のロールバック — 画像特徴抽出 vs テキスト特徴抽出

## 目的
exp_012 は backbone 単体（Swin / BERT）を θ0 に戻した。exp_013 は**投影アダプタまで含めた
「特徴抽出部」を丸ごと**θ0 に戻し、ZCOCO 変化・適応を **画像特徴抽出部 vs テキスト特徴抽出部**に帰属する。
exp_012（backbone単体）との差分で、**投影アダプタ（neck / text_feat_map）の寄与**も切り分ける。

- 学習は一切行わない（重み合成のみ）。能力軸: A（ZCOCO保持）と C（適応）。
- 検出経路（encoder/decoder/bbox_head 等）へ進む前段。

## モジュール定義（接頭辞・実体確認済み）
- **画像特徴抽出部** = `backbone.`(Swin, 187) + `neck.`(ChannelMapper: Swin特徴→256, 16) = **203 params**
- **テキスト特徴抽出部** = `language_model.`(BERT, 共通197) + `text_feat_map.`(Linear 768→256, 2) = **199 params**
- neck/text_feat_map とも全パラメータが unfrozen で変化（swap 非自明）。`language_model` は θ0 のみ
  `...position_ids`（定数バッファ）あり→共通197を置換。

## ハイブリッド定義（各ドメイン2本）
- **H^imgfeat_d** = θ1_d^U の `backbone.*`＋`neck.*` を θ0 に置換。テキスト側・検出経路は unfrozen のまま。
- **H^txtfeat_d** = θ1_d^U の `language_model.*`＋`text_feat_map.*` を θ0 に置換。画像側・検出経路は unfrozen のまま。

### 生成手順（学習なし）
`experiments/exp_013/make_hybrid_rollback.py`（接頭辞グループを指定して合成、サニティ assert 込み）。
出力: `experiments/exp_013/hybrids/{domain}_{imgfeat|txtfeat}_theta0.pth`。

## 使用する重み（exp_012 と同一）
- θ0: `grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`
- θ1_d^U: underwater=exp_010 unfrozen best / 他5=exp_011 unfrozen best。

## 評価
- ZCOCO=`eval_base_coco.py`、適応=自ドメインFT config。いずれも (800,1333) keep_ratio・4GPU。
- 出力: `{domain}_{imgfeat|txtfeat}_theta0_coco/`, `{domain}_{imgfeat|txtfeat}_theta0_adapt/`。

## 比較・解釈
- **基準**: θ0(0.504) / frozen / unfrozen / exp_012 の H^Swin・H^BERT。
- **ZCOCO（符号ルール・合意済み）**: 低下4ドメインで H が**回復**すれば、その特徴抽出部の適応が忘却の一因（クリーン）。
  - **画像側の限界寄与** = (H^imgfeat 回復) − (H^Swin 回復) ≒ neck の寄与。
  - **テキスト側の限界寄与** = (H^txtfeat 回復) − (H^BERT 回復) ≒ text_feat_map の寄与。
- **適応**: 同符号で曖昧（不整合と分離不可）。クリーンな適応寄与は frozen vs unfrozen 側で読む。
  H^txtfeat は contrastive 結合（query·text）により壊滅的に落ちる想定（exp_012 BERT と同様、産物）。

## 方法論的注意（neck/text_feat_map を含めても不整合は消えない）
- 投影アダプタを含めると、Swin→neck（あるいは BERT→text_feat_map）の界面は内部で θ0 整合になるが、
  **neck→encoder（text_feat_map→encoder）の界面に不整合が移る**だけで、共適応は完全には消えない。
  ただし「特徴抽出ストリーム全体を θ0 にする」という解釈上クリーンな切り口になる。
- よって ZCOCO の**回復**は依然クリーン（不整合は下げ方向）。適応の落差は上限のまま。

## 段階的方針
1. H^imgfeat・H^txtfeat（6ドメイン×2）の ZCOCO＋適応を測る。
2. exp_012 と差分を取り、neck / text_feat_map の限界寄与を読む。
3. その後に検出経路の swap（共適応の扱いが最難）を別途設計。

## コスト
重み合成（CPU, 12本）＋ 評価（forwardのみ・新規学習ゼロ）。

## 承認ゲート
行動原理①②に従い、本 design.md の承認後に生成・評価を実行する。

## 関連
- 前段: [[../exp_012/results/swap1_swin_to_theta0]] / [[../exp_012/results/swap2_bert_to_theta0]] / [[../exp_012/design]]
- アーキ（neck=ChannelMapper, text_feat_map, ContrastiveEmbed=query·text）: [[../../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]]
- 現象の出所: [[../exp_011/results/frozen_vs_unfrozen_6domains]]
