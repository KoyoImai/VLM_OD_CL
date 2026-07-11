# exp_018: 部分学習（A: Feature Enhancer だけ凍結 / B: BERT＋text_feat_map のみ学習）— 3ドメイン

## 目的
exp_017（射影アダプタ neck＋text_feat_map のみ学習）に続き、**「特定モジュールの学習可否を切り替える」構成的検証**を進める。
ロールバック分析（exp_012〜016）で得た各モジュールの寄与を、実学習で確かめる。

- **条件A: Feature Enhancer だけ凍結**（`encoder.` 276 のみ凍結、それ以外は全て学習）。
  ＝**unfrozen（全学習）から encoder だけを外した構成**。rollback enc:whole（全学習後に encoder を θ0 に戻す）の**逆操作**にあたり、
  「encoder を最初から動かさないと、unfrozen 比で適応がどれだけ落ち／ZCOCO がどれだけ保たれるか」を測る。
- **条件B: Text Backbone(BERT)＋text_feat_map のみ学習**（`language_model.` 197 ＋ `text_feat_map.` 2 ＝199 params）。他は全凍結。
  ロールバックで txtfeat（BERT＋text_feat_map）が「忘却・適応ともほぼ無関与＝inert」と出た**同じ2モジュールを実学習で動かし**、本当に何も起きないか（適応も忘却も小さいか）を確認する。

いずれも重み合成でなく**実学習**（exp_017 と同種の FT）。能力軸: A（ZCOCO保持）と C（適応）。

## 対象ドメイン（3）
**underwater / videogames / electromagnetic** のみ。残りドメイン（aerial / microscopic / documents）は次実験で実施。
- 選定意図: 近い/易しい（underwater）、遠い・多クラス・exp_017 で適応毀損最大（videogames 0.782→0.113）、遠い（electromagnetic）の3点でスペクトルをカバー。

## 学習設定（凍結機構）
標準 finetune config は backbone/language_model を `lr_mult=0.0` で凍結（BERT はモデルレベルの凍結フラグ無し＝lr_mult のみで凍結、`frozen_stages=-1`）。
これに custom_keys を上書きして各条件の学習対象を設定する。**lr スキームは unfrozen ベースライン（exp_010/011）に一致**させる
（backbone/language_model=lr_mult=0.1＝実効1e-5、他は既定1.0＝1e-4）ことで、比較先 unfrozen と直接比較可能にする。

### 条件A: Feature Enhancer だけ凍結（`freeze_enc_{domain}.py`）
＝ unfrozen 構成で encoder だけを凍結。
- **凍結（lr_mult=0.0）**: `encoder.`（276）のみ。
- **学習**: backbone（lr_mult=0.1）, language_model（lr_mult=0.1）, neck / text_feat_map / decoder / bbox_head /
  memory_trans_fc / memory_trans_norm / query_embedding / level_embed / dn_query_generator（既定1.0）。
- `absolute_pos_embed` は decay_mult=0.0（unfrozen と同一）。

### 条件B: BERT＋text_feat_map のみ（`bert_tfm_{domain}.py`）
- **学習**: `language_model.`（197, lr_mult=0.1＝実効1e-5, unfrozen準拠）＋ `text_feat_map.`（2, 既定1.0＝1e-4）。
- **凍結（lr_mult=0.0）**: backbone, neck, encoder, decoder, bbox_head,
  memory_trans_fc, memory_trans_norm, query_embedding, level_embed, dn_query_generator。

### 共通
- **seed=0 を明示**（`randomness=dict(deterministic=False, seed=0)`。[[always-fix-seed-0]]・unfrozen と同一）。
- スケジュール: 20 epoch・base lr=1e-4・MultiStepLR(milestone=15)（frozen/unfrozen/exp_017 と同一＝比較可能）。
- 実行前に **optimizer param-group を検査**し、意図した凍結/学習配分になっていることを確認（exp_017 と同じ検証）。

## 評価
- **適応**: 学習中の domain valid `best_coco_bbox_mAP` を採用。
- **ZCOCO**: best ckpt を `eval_base_coco.py`（(800,1333)・4GPU）で評価。
- 出力: `experiments/exp_018/{domain}_{freezeEnc|bertTfm}_work_dir/`、`..._zcoco/`。

## 比較（実 mAP）
[[../module_rollback_summary]] の θ0 / frozen / unfrozen / neckTfm-only と同表で対比：
- **条件A（freeze_enc）** の予想: 比較先は **unfrozen**。encoder を動かさないぶん適応は unfrozen より落ちる（Feature Enhancer は適応の要のはず）。
  一方 ZCOCO は unfrozen より保たれる（忘却源 encoder を固定するため）。→ rollback enc:whole と対で「encoder が適応/忘却に効く量」を実学習で確認。
- **条件B（bert_tfm）** の予想: 適応・ZCOCO とも neckTfm-only 近傍かそれ以下で、frozen にすら届かない可能性。
  → txtfeat inert の再確認（テキスト側だけでは検出は適応しない）。

## コスト
実学習 6本（3ドメイン×2条件・各20ep・4GPU）＋ ZCOCO 評価6本。electromagnetic が最長（~14h/本）。逐次で概ね **1.5〜2日**。
※ backward は凍結層も通るため wall-clock は unfrozen とほぼ同等。

## 段階的方針
1. まず1本（underwater 条件A）で optimizer 検査＋loss 低下を確認。
2. 問題なければ残り5本を逐次学習。
3. 全結果を [[../module_rollback_summary]] に条件A/B 列として統合。

## 承認ゲート
行動原理①②に従い、本 design.md の承認後に学習・評価を実行する（実学習・高コスト）。

## 関連
- 対をなす射影のみ学習: [[../exp_017/design]]
- txtfeat inert の根拠: [[../exp_013/results/feature_rollback_imgfeat_vs_txtfeat]]
- Feature Enhancer＝忘却源/適応の要: [[../exp_014/results/feature_enhancer_rollback]]
- 統合表: [[../module_rollback_summary]]
- 凍結は選択の問題: [[encoder-freezing-is-a-choice]]
