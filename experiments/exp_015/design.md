# exp_015: 下流（Query Selection＋Decoder＋Head）のロールバック — 全体＋4分解

## 目的
exp_012（Image/Text Backbone）・exp_013（両特徴射影 neck/text_feat_map）・exp_014（Feature Enhancer=encoder）で
上流を切り分けた。exp_015 は**残りの下流全て**を θ0 に戻し、ZCOCO 変化・適応を
**Language-guided Query Selection ／ Cross-Modality Decoder ／ Contrastive 分類 ／ Box 回帰**に帰属する。
これで exp_012〜015 が MM-Grounding DINO 全 909 params を分割完了する。

- 学習は一切行わない（重み合成のみ）。能力軸: A（ZCOCO保持）と C（適応）。
- exp_014 の発見（忘却源は encoder の画像 self-attn）に対し、下流各段の寄与を確定する。

## モジュール定義（論文表記 ↔ 接頭辞、実体確認済み）
| グループ | 論文表記 | state_dict 接頭辞 | params |
|---|---|---|---|
| **decoder** | Cross-Modality Decoder（query→text / query→image cross-attn, self-attn, ref_point_head） | `decoder.*` | 174 |
| **cls** | Contrastive classification head（ContrastiveEmbed = query·text, log_scale＋bias） | `bbox_head.cls_branches.*` | 7 |
| **reg** | Box regression head（reg_branch MLP → box(cx,cy,w,h)） | `bbox_head.reg_branches.*` | 42 |
| **qsel** | Language-guided Query Selection（pre_decoder 射影・埋め込み） | `memory_trans_fc.*` ＋ `memory_trans_norm.*` ＋ `level_embed` ＋ `query_embedding.*` | 6 |
| **whole** | 上記の和 | 4グループの union | 229 |

### 除外（重要）: `dn_query_generator.label_embedding.weight`（1 param）
- θ0 は (256,256)、各ドメインは (num_classes,256) で **shape 不一致 → θ0 置換不可**。
- denoising クエリは**学習時のみ使用・評価では forward に不登場**（メモ §2.5, §3）→ θ0/θ1 いずれでも評価結果は不変。
- よって**全ロールバックグループから除外**し θ1_d^U のまま据え置く（exp_012 の position_ids と同じ無害な扱い）。
- このため whole=229（下流230のうち label_embedding 1 を除く）。

## ハイブリッド定義（各ドメイン5本）
θ1_d^U の対象接頭辞を θ0 に置換。それ以外（backbone/neck/language_model/text_feat_map/encoder＝上流）は unfrozen のまま。
- **H^whole / H^decoder / H^cls / H^reg / H^qsel**。整合: whole = decoder ∪ cls ∪ reg ∪ qsel（229 = 174+7+42+6）。

### 生成手順（学習なし）
`experiments/exp_015/make_hybrid_downstream.py`（接頭辞グループ指定・サニティ assert・label_embedding 自動除外）。
出力: `experiments/exp_015/hybrids/{domain}_{whole|decoder|cls|reg|qsel}_theta0.pth`（30本）。

## 使用する重み（exp_012〜014 と同一）
- θ0: `grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`（ZCOCO=0.504）
- θ1_d^U: underwater=exp_010 unfrozen best / 他5=exp_011 unfrozen best。

## 評価
- ZCOCO=`eval_base_coco.py`、適応=自ドメインFT config。いずれも (800,1333) keep_ratio・4GPU。
- 駆動: `experiments/exp_015/run_eval.sh {zcoco|adapt|all}`（exp_014 と同形式・継続実行＋失敗追跡）。
- 出力: `{domain}_{group}_theta0_{coco|adapt}/`（60本）。

## 比較・解釈（符号ルール・合意済み）
- **基準**: θ0=0.504 / frozen / unfrozen / exp_012〜014 の各 H。
- **ZCOCO（主証拠＝低下4ドメイン micro/vg/doc/em）**: H が**回復**すれば、その下流モジュールの適応が忘却の一因（クリーン）。
  - 4分解の回復量で decoder / cls / reg / qsel の寄与を内訳。whole と Σ単独の差で相互作用（劣/超加法）を読む。
- **適応**: 同符号で曖昧（上限読み）。クリーンは frozen vs unfrozen。
  - **cls（ContrastiveEmbed）を θ0 に戻すと分類の照合基準（log_scale/bias）が変わり適応が大きく落ちうる**（産物・要注意）。
  - reg を θ0 に戻すと box 回帰が事前学習分布へ戻る＝適応 box 精度への寄与を直接測る（exp_004 の reg ドリフト仮説の検証）。
- **exp_014 との統合**: encoder（上流検出経路）と下流（decoder/head）で、忘却・適応の寄与配分を確定。

## 方法論的注意
- 下流を θ0 にしても encoder→(qsel)→decoder の界面に不整合が残る（encoder は unfrozen のまま）。ZCOCO の回復は依然クリーン（不整合は下げ方向）。
- cls/reg は decoder 各層に share される（`share_pred_layer`）。cls 単独ロールバックは全層の分類基準を同時に θ0 化する。

## 段階的方針
1. H^whole＋H^{decoder,cls,reg,qsel} の ZCOCO＋適応を測る（30 hybrid・60評価）。
2. 低下4ドメインで「回復」を主判定、4分解で内訳、exp_014 と統合。
3. exp_012〜015 を一枚に統合し、忘却・適応の主因モジュールを全モデルで確定。

## コスト
重み合成（CPU, 30本）＋ 評価60本（forwardのみ・新規学習ゼロ、4GPU・~6.5h見込み）。

## 承認ゲート
行動原理①②に従い、本 design.md の承認後に生成・評価を実行する。

## 関連
- 前段: [[../exp_012/design]] / [[../exp_013/design]] / [[../exp_014/results/feature_enhancer_rollback]]
- アーキ（Query Selection §2.5, Decoder §2.6, Head §2.7, ContrastiveEmbed §3）: [[../../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]]
- 忘却源候補（reg ドリフト・log_scale/bias）: [[../exp_004/design]]
