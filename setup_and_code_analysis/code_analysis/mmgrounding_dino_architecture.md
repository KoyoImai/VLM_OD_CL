# MM-Grounding DINO アーキテクチャ参照資料（論文＋コード精読）

作成日: 2026-06-26。出典: 論文 arXiv:2401.02361（MM-Grounding DINO）／元論文 Grounding DINO arXiv:2303.05499、
および mmdet 実装コード。**本資料はコードに即した事実の記録**であり、推測は明示する。継続学習・忘却分析の設計の土台にする。

参照したコード（行番号は読んだ時点）:
- 検出器: `mmdet/models/detectors/grounding_dino.py`（親 `dino.py` → `deformable_detr.py`）
- 対照ヘッド: `mmdet/models/dense_heads/grounding_dino_head.py`
- Transformer層: `mmdet/models/layers/transformer/grounding_dino_layers.py`
- 言語モデル: `mmdet/models/language_models/bert.py`
- config: `configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py`、`*_finetune_8xb4_20e_aerial.py`

---

## 0. 一言で

**テキスト（クラス名）で条件付けるオープン語彙検出器**。画像とテキストを**融合(fusion)**してから、
900個のクエリがデコーダで物体に結びつき、各クエリは**box（回帰）**と**「どのテキストtrkンと近いか」（対照アライメント）**を出す。
**分類は固定の分類器重みではなく、クエリ埋め込みとテキスト埋め込みの内積**で行う。ここが通常の検出器と決定的に違う。

---

## 1. 主要モジュールとデータの流れ

```
画像 ─► [Image Backbone: Swin-T] ─► 多スケール特徴(4 levels) ─┐
                                                              ├─► [Encoder=Feature Enhancer ×6] ─► memory(画像) , memory_text(テキスト)
テキスト ─► [Text Backbone: BERT] ─► [text_feat_map] ─────────┘                │
("cat . dog . ...")                                                            │
                                                                               ▼
                                              [Language-guided Query Selection] (pre_decoder, topk=900)
                                                                               │
                                                                               ▼
                                              [Decoder ×6] (query が画像とテキストに交互に注意)
                                                                               │
                                                                               ▼
                                              [Head]  reg_branch→box(4) , ContrastiveEmbed→ query·text 内積
```

embed_dims=256（Transformer内部）。num_queries=900。encoder/decoder ともに **6層**。画像特徴は **4レベル**。

---

## 2. 各モジュール（コードに即して）

### 2.1 Image Backbone — Swin-T
- `backbone`（Swin, base embed_dims=96, 4ステージ）→ neck で 4レベルの 256次元特徴に。
- **本プロジェクトのFTでは凍結**（`aerial` config: `backbone: dict(lr_mult=0.0)`）。

### 2.2 Text Backbone — BERT-base-uncased
- `language_model`（`bert.py`）。クラス名を `. ` 区切りで連結したプロンプトをトークン化し、
  トークン埋め込み `embedded`、`text_token_mask`、`position_ids`、サブ文ごとの self-attention マスク `masks` を返す。
- **本プロジェクトのFTでは凍結**（`language_model: dict(lr_mult=0.0)`）。
- 注: トークン化は GLIP と違い max_length パディングしない（`longest` か `pad_to_max`）。
  `grounding_dino.py:161` 付近。プロンプト整形は `get_tokens_and_prompts`（同 134-183）。

### 2.3 text_feat_map（**学習対象**）
- `nn.Linear(BERT次元 → 256)`。BERT出力を検出器の埋め込み空間へ射影。`grounding_dino.py:86`。
- **BERTは凍結だが、この射影は学習される** → 「テキスト側の表現は完全には固定でない」。重要。

### 2.4 Encoder = Feature Enhancer（6層、`grounding_dino_layers.py:135-253`）
各層は次の順で**画像とテキストを融合**する（`forward` 228-252）:
1. **fusion_layer**（`SingleScaleBiAttentionBlock`）: 画像↔テキストの**双方向 cross-attention**。画像とテキストを相互に更新。
2. **text_layer**（`DetrTransformerEncoderLayer`）: テキストの self-attention（FFN含む）。
3. **image layer**（`DeformableDetrTransformerEncoderLayer`）: 画像の deformable self-attention（4レベル）。
- 出力: `memory`（画像トークン）と `memory_text`（テキストトークン）。
- **本プロジェクトでは凍結対象外＝学習される**（fusion / text_layer / image layer すべて）。

### 2.5 Language-guided Query Selection（`grounding_dino.py:348-417` pre_decoder）
- encoder出力 `output_memory` と `memory_text` から、**各画像トークンの「最大対照スコア」**を計算し
  （`enc_outputs_class = cls_branches[last](output_memory, memory_text, mask)`、その `.max(-1)`）、
  上位 `num_queries=900` のトークンを選んで初期参照点 `reference_points` とする（374-383）。
- → **クエリの初期化自体がテキストとのアライメントで決まる**（テキスト条件付き）。
- query 埋め込み本体は `self.query_embedding`（`nn.Embedding(900, 256)`、`grounding_dino.py:73`）。
- 学習時は denoising クエリ（dn_query_generator）も連結される（388-393）。

### 2.6 Decoder（6層、`grounding_dino_layers.py:25-132`）
各デコーダ層 `forward`（54-132）の順序:
1. **self_attn**（クエリ間、`MultiheadAttention` 8heads）→ norm
2. **cross_attn_text**（クエリ→テキスト、`MultiheadAttention` 8heads）→ norm … **テキストへの注意**
3. **cross_attn**（クエリ→画像、`MultiScaleDeformableAttention` 8heads）→ norm … **画像への注意**
4. **ffn** → norm
- `ref_point_head`（MLP）で参照点を埋め込みに変換（`grounding_dino_layers.py:268`）。
- 出力 `hidden_states`: 各層のクエリ埋め込み (num_layers, bs, num_queries, 256)。
- **本プロジェクトでは学習される**。

### 2.7 Head（`grounding_dino_head.py`）
2つの枝。デコーダ各層に対し適用（box refine）。`share_pred_layer` 時は全層で同一重みを共有。
- **回帰 reg_branch**（`_init_layers` 110-115）: `Linear(256,256)-ReLU-Linear(256,256)-ReLU-Linear(256,4)`。box(cx,cy,w,h)。**学習対象**。
- **分類 ContrastiveEmbed**（23-89）: **学習可能なクラス重みを持たない**。後述。

---

## 3. 分類の仕組み（最重要 — 通常検出器との決定的な違い）

`ContrastiveEmbed.forward`（`grounding_dino_head.py:62-89`）:
```python
res = visual_feat @ text_feat.transpose(-1, -2)   # クエリ埋め込み · テキストトークン埋め込み（内積）
res = res / sqrt(d)          # log_scale='auto' のとき
res = res + self.bias        # bias=True（初期値 -4.6 のスカラー、全体で1個）
res.masked_fill_(~text_token_mask, -inf)
# max_text_len(=256) に -inf パディング
```
- **クラスごとの重み行列が存在しない。** 学習可能なのは **log_scale（1スカラー）と bias（1スカラー）だけ**。
- つまり「あるクラスを検出できる/できない」は、**クエリ埋め込みとそのクラス名トークン埋め込みの内積の大小**で決まる。
- 推論時（`_predict_by_feat_single` 384-449）:
  - token単位スコア→`convert_grounding_to_cls_scores` でクラス単位スコアに変換（positive map で「どのトークンがどのクラスか」を集約）。
  - `cls_score.view(-1).topk(max_per_img=300)` で上位300の (クエリ×クラス) を選抜。`det_labels = idx % num_classes`、`box = idx // num_classes`。
- → **可視化や評価で出る box は、この「内積スコア上位300」**。スコアが低い検出は topk/閾値で落ちる（前回の議論の根拠）。

### num_classes の役割（誤解しやすい点）
- FT config の `num_classes`（`aerial` の `bbox_head=dict(num_classes=...)`）は、**denoising クエリ生成（dn_query_generator.label_embedding）**等に効くが、
  **ContrastiveEmbed にはクラス固有の重みは無い**。分類能力はあくまでテキスト埋め込みとの内積で決まる。
- 評価時に num_classes 不一致の警告が出ても検出に影響しないのはこのため（exp_003/005 の所見と整合）。

---

## 4. 学習と推論のデータフロー

### 学習 `loss`（`grounding_dino.py:419-502`）
1. 各GTラベルを、プロンプト内の対応トークン位置（positive_map）に変換。
2. テキストを `language_model`→`text_feat_map` で埋め込む。
3. 画像特徴抽出→`forward_transformer`（encoder→pre_decoder→decoder）。
4. `bbox_head.loss`: Hungarian マッチングで各GTを1クエリに割当て、
   - **分類損失**：cls_score（query·text）に対し、正しいトークン位置を1とする 0/1 ターゲットで Focal 系損失（`loss_by_feat_single` 502-599、text マスクでパディング部除外）。
   - **回帰損失**：L1 + GIoU。

### 推論 `predict`（`grounding_dino.py:504-621`）
- プロンプト整形→埋め込み→`forward_transformer`→`bbox_head.predict`→topk300→ラベル名付与。
- chunked（多クラスでトークン超過時）は分割推論（`chunked_size`）。

---

## 5. 本プロジェクトの凍結設定（「学習されるもの/されないもの」）

`aerial` finetune config（`paramwise_cfg`）:
- `backbone`（Swin）: `lr_mult=0.0` → **凍結**
- `language_model`（BERT）: `lr_mult=0.0` → **凍結**
- それ以外（**学習される**）: `text_feat_map`、encoder の fusion/text/image 層、`memory_trans_fc`、
  decoder 全体、`query_embedding`、head の reg_branch、ContrastiveEmbed の log_scale/bias。
- lr=1e-4、20ep。

> 重要な含意：**BERT本体は凍結だが、テキスト埋め込みは完全には固定されない**。
> text_feat_map・encoder の text_layer・fusion_layer が学習されるため、最終的にデコーダが照合する
> `memory_text`（融合後のテキスト表現）は FT で動く。「テキストアンカーは不変」と単純化してはいけない。

> **さらに重要：backbone(Swin)・BERT の凍結は「これまでの実験での選択」であって、モデルの制約ではない。**
> `lr_mult=0.0` は config の設定にすぎず、**場合によっては Image Encoder / Text Encoder も学習対象にできる**。
> 分析や手法を設計する際は、「凍結が絶対」と決めつけず、これらを学習させる選択肢も視野に入れる。

---

## 6. 継続学習・忘却分析にとっての構造的事実（推測でなく演繹）

以下は**アーキテクチャから論理的に導かれる事実**であり、分析手法の提案そのものではない（提案は別途、文献調査の上で行う）。

1. **「クラス重みの忘却」は定義上あり得ない。** 分類にクラス固有パラメータが無いため、COCOクラスの忘却は
   次のいずれかに帰着する：(a) デコーダのクエリ埋め込み（視覚側）のドリフト、(b) 融合後テキスト表現 `memory_text`
   （text_feat_map＋fusion＋text_layer）のドリフト、(c) reg_branch（box）のドリフト、(d) log_scale/bias の変化。
2. **分類スコア＝内積**なので、忘却は「内積が下がった」と必ず書ける。内積の低下は、上記(a)クエリ側か(b)テキスト側の
   **どちらのベクトルが動いたか**に分解できる（両者は学習対象、BERT入力は不変）。
3. **検出器はテキスト条件付き**：クエリ選択（2.5）もデコーダ（2.6 cross_attn_text）もテキストに依存する。
   よって「同じ画像でもプロンプト次第で出力が変わる」——closed-set検出器に無い自由度。
4. **box と分類は別経路だが、出力選抜（topk300）は分類スコア依存**。よって「box は出せるがスコアで落ちる」状態が起こり得る
   （局在と分類の切り分けには、スコア選抜前の生出力を見る必要がある＝前回の議論の技術的根拠）。
5. **凍結境界**：忘却が起きうるのは「学習される範囲」（§5）に限定される。backbone(Swin)・BERTは不変なので、
   画像/テキストの**低次特徴は保持**され、ドリフトは fusion 以降に局在するはず（要検証の仮説）。

---

## 7. 主要ハイパーパラメータ（config確定値）

| 項目 | 値 | 出典 |
|---|---|---|
| num_queries | 900 | pretrain config:11 |
| embed_dims（Transformer） | 256 | config |
| encoder 層数 / decoder 層数 | 6 / 6 | config:58,79 |
| 画像特徴レベル数 | 4 | self_attn num_levels=4 |
| decoder 各注意の heads | self 8 / text 8 / image(deform) 8 | config:83-87 |
| max_text_len | 256（多クラス時 512 に拡張） | contrastive_cfg、CLAUDE.md |
| log_scale / bias | 'auto'(÷√d) / True(初期-4.6) | contrastive_cfg:97 |
| max_per_img（topk） | 300 | test_cfg |
| FT 凍結 | backbone, language_model（lr_mult=0） | aerial config:89-90 |

---

## 関連
- 既存の論文要約: [[../../papers/MM-GroundingDINO_summary]]
- 継続学習での先行: [[../../papers/ZiRaGroundingDINO_summary]]
- 分析手法カタログ（既存・要再調査）: `experiments/exp_006.5/analysis_methods_catalog.md`
