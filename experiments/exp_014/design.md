# exp_014: Feature Enhancer（encoder）のロールバック — 全体＋3経路分解

## 目的
exp_012 は backbone 単体（Swin/BERT）、exp_013 は特徴抽出部（投影アダプタ含む）を θ0 に戻した。
exp_014 は検出経路の入口にあたる **Feature Enhancer（= encoder, 6層）** を θ0 に戻し、
ZCOCO 変化・適応を **enhancer 全体／その3経路（cross-modal fusion・テキスト・画像）** に帰属する。

- 学習は一切行わない（重み合成のみ）。能力軸: A（ZCOCO保持）と C（適応）。
- アーキ上 **Encoder = Feature Enhancer**（[[../../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]] §2.4）。

## モジュール定義（接頭辞・キー数は実体確認済み・全6ドメインで集合/shape一致）
- **enc（Feature Enhancer 全体）** = `encoder.*` = **276 params**
- **fusion** = `encoder.fusion_layers.*`（`SingleScaleBiAttentionBlock`, 画像↔テキスト双方向 cross-attn）= **108**
- **text** = `encoder.text_layers.*`（テキスト self-attn＋FFN）= **72**
- **image** = `encoder.layers.*`（画像 deformable self-attn, 4レベル）= **96**
- 整合: **enc = fusion ∪ text ∪ image**（276 = 108 + 72 + 96）。定数バッファ無し（θ0のみキー = 0）。

## ハイブリッド定義（各ドメイン4本）
θ1_d^U の対象接頭辞を θ0 に置換。それ以外（backbone/neck/language_model/text_feat_map/decoder/bbox_head）は **unfrozen のまま**。
- **H^enc_d**    = `encoder.*` を θ0 に置換。
- **H^fusion_d** = `encoder.fusion_layers.*` を θ0 に置換。
- **H^text_d**   = `encoder.text_layers.*` を θ0 に置換。
- **H^image_d**  = `encoder.layers.*` を θ0 に置換。

### 生成手順（学習なし）
`experiments/exp_014/make_hybrid_encoder.py`（接頭辞グループ指定で合成、サニティ assert 込み）。
出力: `experiments/exp_014/hybrids/{domain}_{enc|fusion|text|image}_theta0.pth`（24本）。

## 使用する重み（exp_012/013 と同一）
- θ0: `grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`（ZCOCO=0.504）
- θ1_d^U: underwater=exp_010 unfrozen best / 他5=exp_011 unfrozen best。

## 評価
- ZCOCO=`configs/mm_grounding_dino/eval_base_coco.py`、適応=自ドメインFT config。いずれも (800,1333) keep_ratio・4GPU。
- 駆動: `experiments/exp_014/run_eval.sh {zcoco|adapt|all}`（exp_013 と同形式・継続実行＋失敗追跡）。
- 出力: `{domain}_{group}_theta0_coco/`, `{domain}_{group}_theta0_adapt/`（48本）。

## 比較・解釈（符号ルール・合意済み）
- **基準**: θ0=0.504 / frozen / unfrozen / exp_012(H^Swin,H^BERT) / exp_013(H^imgfeat,H^txtfeat)。
- **ZCOCO（主証拠＝低下4ドメイン micro/vg/doc/em）**:
  - H が **回復**（unfrozen より上）すれば、その経路の適応が忘却の一因（**クリーン**。共適応の不整合は下げ方向のため、乗り越えての回復は確証）。
  - 低下/不変は帰属不能（共適応の不整合と分離不可）。
- **3経路の限界寄与**: H^enc の回復を、H^fusion / H^text / H^image の回復で内訳。
  3経路の回復量の和 ≈ enc 全体の回復量（経路間相互作用ぶんはズレとして観測）。
- **近ドメイン（underwater/aerial, ZCOCO上昇側）は補助的に見るのみ**（有益な適応の除去と不整合が同符号で曖昧）。
- **適応**: 同符号で曖昧（不整合と分離不可）。クリーンな適応寄与は frozen vs unfrozen 側で読む。
  - 特に **text/fusion 経路**は contrastive 結合（encoder 出力 query·text）への影響が大きく、適応が壊滅的に落ちうる（産物・上限として読む）。

## 方法論的注意（共適応は完全には消えない）
- encoder を θ0 にしても、**encoder→pre_decoder→decoder の界面に不整合が移る**（decoder/bbox_head は unfrozen のまま）。
  ただし「Feature Enhancer ストリームを θ0 にする」という解釈上クリーンな切り口であり、ZCOCO の**回復**は依然クリーン（不整合は下げ方向）。
- fusion は画像・テキスト両 memory を相互更新するため、fusion 単独ロールバックは両ストリームの下流に影響する（cross-modal 結合の寄与として解釈）。

## 段階的方針
1. H^enc（6）＋ H^{fusion,text,image}（18）の ZCOCO＋適応を測る。
2. 低下4ドメインで「回復」が出るか主判定。enc 全体と3経路内訳を突き合わせ。
3. exp_012/013 と統合し、忘却の主因モジュール（backbone vs 特徴抽出 vs enhancer の各経路）を一枚に整理。

## コスト
重み合成（CPU, 24本）＋ 評価48本（forwardのみ・新規学習ゼロ、4GPU・~5h見込み）。

## 承認ゲート
行動原理①②に従い、本 design.md の承認後に生成・評価を実行する。

## 関連
- 前段: [[../exp_012/design]]（backbone単体）/ [[../exp_013/design]]（特徴抽出部）
- アーキ（Encoder=Feature Enhancer §2.4, fusion/text/image 層）: [[../../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]]
- 現象の出所: [[../exp_011/results/frozen_vs_unfrozen_6domains]]
