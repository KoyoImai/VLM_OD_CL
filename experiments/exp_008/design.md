# exp_008: 単純 LoRA の追加による学習・評価（LoRA組込みの確立＋ベースライン）

## 位置づけ
提案手法（マージ方式 LoRA）の実装に入る前段。**MM-Grounding DINO に LoRA を組み込む実装を確立**し、
**素の LoRA（base 凍結＋低ランクアダプタ、マージ／直交化なし）が適応と COCO 忘却にどう効くか**を、
full FT（exp_003）/ frozen FT（exp_004,005）と比較して把握する。
- 逐次・マージ・干渉制御は**まだ入れない**（単独ドメインの単純 LoRA のみ）。
- 開発ドメイン: **underwater / electromagnetic / documents**（各単独）。まず underwater で実装検証 → 3ドメイン。

## 目的
1. LoRA 注入の実装が mmdet の学習・評価で正しく動くこと（base 凍結・LoRA のみ学習）を確認。
2. 素の LoRA の「適応 mAP」「COCO 忘却」を測り、full FT / frozen FT と比較。
   - 仮説: LoRA は base を保持し低ランク差分のみ → frozen 全FT より忘却が小さい可能性（rank で調整可）。

## 実装したプログラム（`projects/lora_cl/`・既存コード不変）
mmdet 本体は触らず、`custom_imports=['projects.lora_cl']` で読み込む独立モジュールとして実装（オプトイン）。

| ファイル | 役割 |
|---|---|
| `lora_layers.py` | **`LoRALinear`**：既存 `nn.Linear` を `W h + (α/r)·B(A h)` に置換。**元の weight/bias を同じキー名で保持**（事前学習 ckpt をそのままロード可）。`A` 乱数初期化・`B` ゼロ初期化（初期は元と同一出力）。`merge()`（base へ統合）を実装。 |
| `grounding_dino_lora.py` | **`GroundingDINOLoRA(GroundingDINO)`**：構築時に対象 `nn.Linear` を `LoRALinear` に置換し、**LoRA 以外を `requires_grad=False`**。付与層・スキップ層・trainable 数をログ出力。 |
| `optim.py` | **`TrainableParamsConstructor`**：optimizer に **trainable(=LoRA) のみ**を登録（mmengine 既定は凍結 param も登録するため明示的に絞る）。 |
| `__init__.py` | 上記を import して登録。 |

## LoRA を「導入できる」箇所（実装上の対象範囲）
`lora` 設定で **`nn.Linear` のみ**を対象に選べる（①構成要素 `include` ②種別 `include_types` ③正規表現 `include_patterns` の AND、`exclude_patterns` で除外）。凡例: ◎可=include 指定で導入可 / ×skip=nn.Linear だが MHA で無効→自動除外 / ×外=非Linear or 凍結。

| 構成要素 / サブ層 | コード名（例） | 実装 | LoRA導入 | 理由 |
|---|---|---|---|---|
| Image Backbone | `backbone.*`（Swin qkv 等） | nn.Linear | ×外 | 凍結方針（既定除外） |
| Text Backbone | `language_model.*`（BERT q/k/v 等） | nn.Linear | ×外 | 凍結方針（既定除外） |
| neck | `neck.convs.*` | 1×1 Conv2d | ×外 | conv（Linearでない）。初回除外 |
| テキスト射影 | `text_feat_map` | nn.Linear | ◎可 | include に追加すれば対象 |
| クエリ選択 | `memory_trans_fc` | nn.Linear | ◎可 | 同上 |
| encoder 画像self-attn | `encoder.layers.x.self_attn.{value_proj,output_proj,sampling_offsets,attention_weights}` | nn.Linear(deformable) | ◎可 | |
| encoder FFN | `encoder.layers.x.ffn` / `encoder.text_layers.x.ffn` | nn.Linear | ◎可 | |
| encoder テキストself-attn | `encoder.text_layers.x.self_attn.attn.{in_proj,out_proj}` | MHA | ×skip(out_proj)/×外(in_proj) | in_proj=生Param, out_proj=functionalで無効 |
| encoder fusion | `encoder.fusion_layers.x.attn.*` | nn.Linear(BiAttn) | ◎可 | |
| decoder 自己注意 | `decoder.layers.x.self_attn.attn.*` | MHA | ×skip/×外 | 同上 |
| decoder→画像 cross-attn | `decoder.layers.x.cross_attn.*` | nn.Linear(deformable) | ◎可 | |
| decoder→テキスト cross-attn | `decoder.layers.x.cross_attn_text.attn.*` | MHA | ×skip/×外 | 同上 |
| decoder FFN | `decoder.layers.x.ffn` | nn.Linear | ◎可 | |
| decoder 参照点MLP | `decoder.ref_point_head.layers.*` | nn.Linear | ◎可 | |
| bbox_head 回帰 | `bbox_head.reg_branches.*` | nn.Linear | ◎可 | Localization |
| bbox_head 分類 | `bbox_head.cls_branches.*` | 対照(bias のみ) | ×外 | nn.Linearでない |
| クエリ初期値 | `query_embedding` | Embedding | ×外 | nn.Linearでない |

## 学習で「実際に導入した」箇所（本実験の設定）
config: `lora=dict(r=16, alpha=16, include=['encoder','decoder','bbox_head'])`

| 対象（構成要素） | 付与した LoRA 層 | 層数 |
|---|---|---|
| encoder 画像層（deformable射影4＋FFN2）×6層 | value/output/sampling/attention_weights + ffn×2 | 36 |
| encoder text_layers（FFN2）×6層 | ffn×2 | 12 |
| encoder fusion_layers（attn射影6）×6層 | v/l/values_v/values_l/out_v/out_l_proj | 36 |
| decoder（cross_attn deformable射影4＋FFN2）×6層 | value/output/sampling/attention_weights + ffn×2 | 36 |
| decoder 参照点MLP | ref_point_head.layers.0/1 | 2 |
| bbox_head 回帰（reg_branches 7×3） | reg_branches.k.{0,2,4} | 21 |
| **付与 合計** | | **143** |
| 自動スキップ（MHA out_proj） | encoder text 6 ＋ decoder self 6 ＋ cross_attn_text 6 | 18 |
| 対象外（凍結 / 非Linear） | backbone・BERT・MHA in_proj・cls対照・query_embedding | — |
| **学習パラメータ** | **LoRA(lora_A/B) のみ** | **2,400,704 / 175,320,029 (1.37%)** |

※ `text_feat_map` / `memory_trans_fc` は top-level 名のため `include=['encoder','decoder','bbox_head']` には**含まれない**（付ける場合は include に追加）。

## ハイパーパラメータ
- rank `r=16`、スケール `α=16`（exp_007 B：適応は中程度の低ランク、上位16で56〜89%）。
- optimizer: AdamW, **lr=1e-4（通常学習 exp_003〜006 と同一）**, wd=1e-4。
  **`constructor='TrainableParamsConstructor'`** で LoRA のみ optimizer 登録。
- 実効バッチ64（per-GPU8×4×累積2）、20 epochs(milestone[15])、seed=0, deterministic=False、4GPU 分散。

## 実装の検証（実施済み）
1. 構築＋事前学習ロード: strict=False で missing は lora_* のみ（base は正しくロード）。
2. **実学習4イテレーションで base 重みが完全不変**（LoRALinear base / MHA in_proj / **backbone(Swin)** / **BERT** すべて max|Δ|=0）、**LoRA(lora_A/B) のみ変化**、loss 計算 OK。
3. **optimizer の登録パラメータは LoRA のみ**（numel=2.40M）。
→ 「学習されるのは LoRA だけ・事前学習済みモデルは一切更新されない」を実証。

## 実行
```bash
# 各ドメイン d ∈ {underwater, electromagnetic, documents}
bash tools/dist_train.sh experiments/exp_008/configs/<d>_lora.py 4 --work-dir experiments/exp_008/<d>_work_dir
BEST=$(ls experiments/exp_008/<d>_work_dir/best_coco_bbox_mAP_epoch_*.pth | tail -1)
# LoRA を base へマージして plain ckpt 化（merge_lora.py、後述）
python projects/lora_cl/merge_lora.py "$BEST" experiments/exp_008/<d>_merged.pth
# 適応評価（自ドメイン）: マージ済み plain ckpt を既存 plain config で評価
bash tools/dist_test.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_<d>.py experiments/exp_008/<d>_merged.pth 4 --work-dir .../<d>_eval
# 忘却評価（COCO）
bash tools/dist_test.sh configs/mm_grounding_dino/eval_base_coco.py experiments/exp_008/<d>_merged.pth 4 --work-dir .../<d>_coco
```
※ 評価は **LoRA を base にマージ**（`W += (α/r)·BA`）して plain な state_dict にしてから、既存の plain config
（ドメイン用 / `eval_base_coco`）でロード・推論する。マージ後は通常の GroundingDINO と同一構造のため LoRA 用 eval config は不要。
LoRALinear の `merge()` と各層の `lora_scaling`(buffer) を使って `merge_lora.py` で実施。

## 比較の基準値と結果（適応 / COCO）
| ドメイン | full FT(exp_003) | frozen FT(exp_004/005) | 単純LoRA(r16,lr1e-4) |
|---|---|---|---|
| underwater | 0.340 / 0.438 | 0.316 / 0.414 | **0.218 / 0.417** |
| electromagnetic | — | 0.454 / 0.331 | 取得予定 |
| documents | — | 0.478 / 0.275 | 取得予定 |
（COCO zero-shot 上限 0.504）

### underwater 単純LoRA(r16,lr1e-4) の所見
- 適応 0.218（< frozen 0.316 < full 0.340）＝**適応が最も低い**。
- COCO 0.417（≈ frozen 0.414）＝**忘却は frozen 全FT とほぼ同じで減っていない**。
- → 「単純に LoRA を足すだけでは忘却は減らず、適応だけ下がる」。原因切り分けのため下記 sweep を追加。

## 期待される成果物
- 実装: `projects/lora_cl/`（LoRALinear / GroundingDINOLoRA / TrainableParamsConstructor / merge_lora.py）、
  `experiments/exp_008/configs/<d>_lora.py`
- `experiments/exp_008/<d>_work_dir/`（学習・マージ前ckpt）, `<d>_merged.pth`（マージ後）, `<d>_eval/`, `<d>_coco/`
- `experiments/exp_008/results/lora_summary.md`（適応／COCO忘却 を full/frozen と比較）
- `experiments/exp_008/outputs/notes.md`（LoRA の挙動・所見・次への示唆）

## 想定される懸念点
1. **LoRA 注入**（検証済）: base 凍結・LoRA のみ学習を実学習で確認済（base Δ=0、optimizer は LoRA のみ）。
   MHA(in_proj 生Param / out_proj は functional) は対象外・自動スキップ。
2. **lr/epoch の適合**: 通常学習と同一の lr=1e-4 を採用。LoRA は学習パラメータが少ないため収束が遅い可能性 →
   underwater の loss 推移・val mAP で確認し、必要なら別途調整。
3. **評価**: マージ済み plain ckpt を既存 plain config でロード。COCO 評価は num_classes 不一致警告（既知・影響なし）。
4. OOM は per-GPU8 で実績内（LoRA は学習パラメータ減でむしろ軽い）。

## 成功基準
- LoRA 学習が回り（trainable=LoRA のみを確認）、3ドメインの適応 mAP と COCO 忘却を取得。
- 素の LoRA を full FT / frozen FT と比較し、**「LoRA は忘却を減らせるか／適応は保てるか」**の傾向を得る。
- 結果を手法フェーズ（マージ方式 LoRA, exp_009以降）の設計（rank/lr/対象層）に引き継ぐ。

---

# 追加実験 (exp_008b): underwater lr×rank sweep

## 動機
underwater 単純LoRA(r16,lr1e-4) が「適応 0.218（低い）・COCO 0.417（忘却が減らない）」という反直感的結果。
原因が **(a) lr 不足（LoRA は高 lr 定石）** か **(b) rank=容量不足** かを切り分けるため、underwater で lr と rank を振る。

## 設定（固定条件・公平化）
- underwater 単独・frozen base・実効64・20ep・milestone[15]・seed0・対象=encoder+decoder+bbox_head。
- **alpha=rank に統一（scaling=1）** → rank は「容量」だけを変える（スケール非依存で切り分け）。
- 各本: 学習 → `merge_lora.py` でマージ → underwater 適応評価 ＋ COCO 忘却評価。

## グリッド（新規6本、既存 (1e-4,r16) と合わせ比較）※適応/COCO
| lr \ rank | 16 | 32 | 64 |
|---|---|---|---|
| 1e-4 | (済) 0.218/0.417 | (済) **0.241/0.412** | (済) **0.256/0.393** |
| **1e-3** | (済) **0.315/0.315** | (済) **0.335/0.323** | (済) **0.336/0.250** |
| **5e-3** | (済) **0.322/0.136** | (済) **0.257/0.097** | **発散・除外** |

### 完了分の所見（lr1e-3 系・基準 frozen FT 0.316/0.414）
- **lr=1e-4 行（全rank完了）は適応低迷で頭打ち**：r16/r32/r64 = 0.218/0.241/0.256（COCO 0.417/0.412/0.393）。rank↑でも適応は frozen FT(0.316) に届かず、**律速は rank でなく lr**であることを確定（rank はわずかな適応↑とCOCO↓のみ）。
- **lr を 1e-4→1e-3 に上げると適応が 0.218→0.315 と frozen FT(0.316) 水準まで回復**。lr 不足が単純LoRA(1e-4)の適応低迷の主因だった。
- ただし**回復と引き換えに COCO 忘却が増大**（r16 で 0.417→0.315）。**rank↑でも適応はわずかに伸びる(0.315→0.336)が COCO はさらに悪化**（r64 で 0.250）。
- → 「**適応とゼロショット保持のトレードオフ**」が単純LoRAでも明確に再現。exp_007 の「適応と忘却が同一部分空間を奪い合う」主張と整合。提案手法（マージ方式）では rank/スケール抑制＋soft保護でこのトレードオフ点を改善するのが狙い。

### lr5e-3 系の所見（lr が高すぎる：忘却激化・発散）
- **適応は伸びず COCO だけ壊滅**：r16 で 0.322/**0.136**、r32 で **0.257**/**0.097**（適応はむしろ低下、COCO は zero-shot 0.504 の 1/4 以下）。lr1e-3 系より全面的に劣化。
- **r64 は発散**：epoch 11 で `grad_norm=inf`、loss が上昇（57→64）し、Hungarian matching が NaN/Inf を踏んで `matrix contains invalid numeric entries` でクラッシュ → design.md の方針通り**除外**。
- **結論**：単純 LoRA の適切な lr は **1e-3**（適応 frozen FT 水準＋COCO の劣化が相対的に最小）。5e-3 は高すぎて不安定・大忘却。**採用値＝lr=1e-3、rank は適応/忘却トレードオフで r16〜r32**（r64 は COCO 劣化が大）。
- sweep 全6本（+既存 1e-4,r16）完了。全結果は `results/sweep_underwater.txt`。

config: `experiments/exp_008/configs/underwater_lora_lr{1e-3,5e-3}_r{16,32,64}.py`
（`_base_='./underwater_lora.py'` を継承し lora.r/alpha と optimizer.lr のみ上書き）

## 見たいこと
- **lr↑で適応が frozen FT(0.316) に近づくか**、その時 **COCO 忘却は増えるか**（適応とゼロショットのトレードオフ）。
- **rank↑で適応が伸びるか**（容量律速か）。
- → 単純 LoRA の適応・忘却の性質を把握し、提案手法（マージ方式）の rank/lr 設定に反映。

## 成果物
- `experiments/exp_008/uw_lr*_r*_work_dir/`（学習）, `uw_*_merged.pth`, `uw_*_eval/`, `uw_*_coco/`
- `experiments/exp_008/results/sweep_underwater.txt`（各 (lr,rank) の 適応/COCO 一覧）

## コスト・懸念
- 6本×約6h ≈ **36h**（4GPU 逐次）。OOM は per-GPU8 で実績内（rank↑でも LoRA は微増のみ）。
- lr=5e-3 は高めで発散の可能性 → loss を監視、発散時は当該 (lr,rank) を除外。
