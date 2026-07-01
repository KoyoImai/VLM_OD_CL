# exp_016: Cross-Modality Decoder の内部処理ごとのロールバック（4分解）

## 目的
exp_015 で下流のうち **Cross-Modality Decoder** が唯一の（中程度の）忘却源と分かった。
exp_014 が Feature Enhancer を image/text/fusion に分解したのと同様、exp_016 は
**デコーダ層の内部処理を4つに分解**し、忘却・適応がどの処理に起因するかを帰属する。

- 学習は一切行わない（重み合成のみ）。能力軸: A（ZCOCO保持）と C（適応）。
- **主仮説**: exp_014/013 で「忘却源は画像の空間的注意に局在」と判明。デコーダでも
  **dec_crossattn_img（クエリ→画像 deformable cross-attn、`sampling_offsets`/`attention_weights`）** が
  忘却を駆動し、text/self/ffn は inert、と予想。本実験はこの追試。

## モジュール定義（各処理＋直後の post-norm を同梱、実体確認済み）
デコーダ層の forward 順は self_attn → norms[0] → cross_attn_text → norms[1] → cross_attn(画像) → norms[2] → ffn → norms[3]。
各処理に**直後の LayerNorm を同梱**（機能単位）。全6層合計：

| グループ | 論文/実装対応 | 含むキー | params |
|---|---|---|---|
| **dec_selfattn** | クエリ↔クエリ self-attn（`MultiheadAttention`） | `*.self_attn.*` ＋ `*.norms.0.*` | 36 |
| **dec_crossattn_text** | クエリ→テキスト cross-attn（`MultiheadAttention`） | `*.cross_attn_text.*` ＋ `*.norms.1.*` | 36 |
| **dec_crossattn_img** | クエリ→画像 deformable cross-attn（`MultiScaleDeformableAttention`） | `*.cross_attn.*`(text除く) ＋ `*.norms.2.*` | 60 |
| **dec_ffn** | FFN 256→2048→256 | `*.ffn.*` ＋ `*.norms.3.*` | 36 |

- 整合: 4グループ = `decoder.layers.*` 全体（168 = 36+36+60+36）。重複・漏れなし。
- **4グループ外（unfrozen 据え置き）**: `decoder.ref_point_head`(4) ＋ `decoder.norm`(最終LayerNorm, 2) = 6 params。
  空間注意を持たない補助のため単独ロールバックしない。num_classes 依存キーは decoder に無く除外処理不要。

## ハイブリッド定義（各ドメイン4本）
θ1_d^U の対象グループを θ0 に置換。それ以外（上流 backbone/neck/lm/tfm/encoder、下流 head/qsel、
および ref_point_head/final_norm）は unfrozen のまま。
- **H^dec_selfattn / H^dec_crossattn_text / H^dec_crossattn_img / H^dec_ffn**。

### 生成手順（学習なし）
`experiments/exp_016/make_hybrid_decoder.py`（マッチャ指定・サニティ assert 込み）。
出力: `experiments/exp_016/hybrids/{domain}_{group}_theta0.pth`（24本）。

## 使用する重み（exp_012〜015 と同一）
- θ0: `grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`（ZCOCO=0.504）
- θ1_d^U: underwater=exp_010 unfrozen best / 他5=exp_011 unfrozen best。

## 評価
- ZCOCO=`eval_base_coco.py`、適応=自ドメインFT config。いずれも (800,1333) keep_ratio・4GPU。
- 駆動: `experiments/exp_016/run_eval.sh {zcoco|adapt|all}`（exp_014/015 と同形式・継続実行＋失敗追跡）。
- 出力: `{domain}_{group}_theta0_{coco|adapt}/`（48本）。

## 比較・解釈（符号ルール・合意済み）
- **基準**: θ0=0.504 / frozen / unfrozen / **exp_015 の decoder（=whole decoder 174）** / exp_014 の encoder 各経路。
- **ZCOCO（主証拠＝低下4ドメイン micro/vg/doc/em）**: H が**回復**すれば、その処理の適応が忘却の一因（クリーン）。
  - 4処理の回復量を比較し、decoder 内の忘却源を特定。**Σ(4処理) ≒ exp_015 decoder(whole) の回復** かで加法性を確認
    （残差は ref_point_head/final_norm＋処理間相互作用）。
  - **主仮説の判定**: dec_crossattn_img の回復が他3処理を大きく上回れば「画像空間注意が忘却源」を追試確認。
- **適応**: 同符号で曖昧（上限読み）。クリーンは frozen vs unfrozen。
  - dec_crossattn_text を θ0 に戻すとテキスト照合が変わり適応が落ちうる（産物の可能性、要注意）。

## 方法論的注意
- デコーダを θ0 にしても encoder→pre_decoder→decoder→head の界面に不整合が残る（他は unfrozen）。ZCOCO の回復は依然クリーン。
- 各処理に post-norm を同梱するため、効果は「注意/FFN の重み＋その正規化」の合算（norm は 2/層と小さく、主因は注意/FFN）。

## 段階的方針
1. H^{dec_selfattn,dec_crossattn_text,dec_crossattn_img,dec_ffn} の ZCOCO＋適応を測る（24 hybrid・48評価）。
2. exp_015 decoder(whole) と突き合わせ、decoder 内の忘却源を特定。exp_014 encoder image と対比。
3. exp_012〜016 を統合し「忘却源＝画像の空間的注意（encoder self-attn＋decoder cross-attn）」仮説の最終確認。

## コスト
重み合成（CPU, 24本）＋ 評価48本（forwardのみ・新規学習ゼロ、4GPU・~5h見込み）。

## 承認ゲート
行動原理①②に従い、本 design.md の承認後に生成・評価を実行する。

## 関連
- 前段: [[../exp_014/results/feature_enhancer_rollback]]（encoder分解）/ [[../exp_015/results/downstream_rollback]]（decoder=whole）
- アーキ（Decoder §2.6, 3注意＋FFN の forward 順）: [[../../setup_and_code_analysis/code_analysis/mmgrounding_dino_architecture]]
