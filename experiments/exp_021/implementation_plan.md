# exp_021 実装計画書: DitHub の MM-Grounding DINO 上での再現

作成 2026-07-12。実装前のユーザー承認ゲート。手法の詳細は
[[../../papers/DitHub_implementation_notes]] を正本とし、本書は「mmdetection の
上でどう作るか」だけを定める。前例: [[../exp_020/implementation_plan]]（ZiRa）。

## 1. 方針（ユーザー指定、exp_020 と同一）

公式実装（https://github.com/chiara-cap/DitHub）に準拠し、mmdetection に合わせて
調整する。学習ハイパーパラメータは exp_xxx 系列（lr 1e-4、20 epoch、seed=0、
4 GPU）に合わせる。実装コードは mmdet の同種プログラム群と同じディレクトリに
新規ファイルとして追加し、既存ファイルは `__init__.py` を含めて一切変更しない。
登録は config の custom_imports。忠実に実装できない点が出たら止めて相談する。

## 2. 何を作るか（新規ファイル台帳）

mmdet 側:

| パス | 役割 |
|---|---|
| `mmdet/models/layers/dithub_layers.py` | DitHubLinear（クラス別 A プール + 共有 B の LoRA 層）、DitHubTextSelfAttention（q/k/v/o 分解型 attention）、DitHubState（フェーズ・クラス選択・ライブラリの singleton） |
| `mmdet/models/detectors/dithub_grounding_dino.py` | DitHubGroundingDINO（対象層の走査・差し替え、凍結、loss/predict でのクラス設定） |
| `mmdet/engine/hooks/dithub_phase_hook.py` | warmup → specialization 切替フック（epoch 境界で enable_per_class を発火） |

experiments 側:

| パス | 役割 |
|---|---|
| `experiments/exp_021/implementation_plan.md` | 本書（実装計画・台帳） |
| `experiments/exp_021/design.md` | 学習実験の設計書（実装検証後に作成・承認） |
| `experiments/exp_021/configs/dithub_{domain}.py` | ドメイン毎の学習 config（標準版） |
| `experiments/exp_021/configs/dithub_official_{domain}.py` | ドメイン毎の学習 config（公式準拠版） |
| `experiments/exp_021/configs/dithub_eval_coco.py` | ZCOCO 評価用 config |
| `experiments/exp_021/check_dithub_setup.py` | 学習前検証スクリプト |
| `experiments/exp_021/run_train.sh` | 標準版の学習＋ZCOCO評価の一括実行スクリプト |
| `experiments/exp_021/run_official.sh` | 公式準拠版の学習＋ZCOCO評価の一括実行スクリプト |
| `experiments/exp_021/merge_dithub.py` | タスク境界処理スクリプト（式4の B 融合 + ライブラリ継承。将来の逐次実験用、exp_021 では単体テストまで） |

## 3. 設計の要点

### 3.1 対象層の選定と差し替え

公式の規則をそのまま移植する: **out_features ≥ 128 の全 nn.Linear** を対象とし、
`backbone / language_model / text_feat_map / neck / decoder / bbox_head /
query_embedding / dn_query_generator / memory_trans_norm` で始まる名前を除外する。
**memory_trans_fc は含める**（論文は「encoder のみ」と書くがコードは
enc_output = memory_trans_fc にも挿しており、コード準拠とする。相談事項①）。
実対象は encoder の画像側 deformable attention（sampling_offsets /
attention_weights / value_proj / output_proj）、画像側 FFN、テキスト側
self-attention（q/k/v/o）、テキスト側 FFN、融合層 BiAttention の 6 射影、
memory_trans_fc。

DitHubLinear は ZiRaLinear と同じ「凍結済み元重みを同居させる nn.Linear
サブクラス」方式で、checkpoint のキー名を保つ。テキスト側 self-attention のみ、
mmcv の MultiheadAttention が持つ融合 in_proj を q/k/v の独立 Linear に分解した
DitHubTextSelfAttention へ差し替える（公式の apply_custom_attention に対応）。
計算は `F.multi_head_attention_forward(use_separate_proj_weight=True)` を使い
数学的等価を保証し、checkpoint の `in_proj_weight / in_proj_bias / out_proj.*`
キーは `_load_from_state_dict` の上書きで分解して読み込む（既存 ckpt を無変換で
ロード可能）。decoder の attention は LoRA 対象外なので分解しない（公式は decoder
も分解するが恒等変換であり機能差はない）。

### 3.2 層の構造と初期化（公式 LinearPool と同一の数式）

各対象層: 凍結 weight/bias + `shared_lora_b`（out×r、0 初期化）+
`warmup_lora_a`（r×in、kaiming）+ `per_class_lora_A`（ParameterDict、
ドメインのクラス数分、kaiming）。B=0 のため初期状態の出力は θ0 と**厳密に一致**。
scaling = lora_alpha / r = 8/16 = 0.5。

### 3.3 フェーズ制御とクラス選択

DitHubState（singleton）が warmup フラグと「現在のサンプル別クラス」を保持する。
学習 forward は、warmup 中は全サンプルに warmup_lora_a、specialization 中は
検出器の loss() が画像ごとに GT クラスから一様ランダムに 1 つ選んだクラスの A を
サンプル別に適用する（バッチ次元でスタックして bmm、公式と同一）。切替は
custom hook が epoch 境界で `enable_per_class` を呼び、公式の式3の通り、
クラスごとに「学習済みなら λ_A·A_wu + (1−λ_A)·A_old、未学習なら A_wu のコピー」
を実行する。20 epoch を公式の等分規則に従い **warmup 10 + specialization 10**
に割る（相談事項③）。フェーズは各層の persistent buffer に記録し、checkpoint
から評価する際にも正しいフェーズの forward が再現されるようにする。

### 3.4 評価時の forward

公式と同一: プロンプト中のクラスのうちモジュールを持つものの `mean(B·A_c)` を
1 回加算。モジュールを持たないクラスだけなら θ0 と厳密一致。ZCOCO では COCO 80
クラス名とドメインクラス名の一致分のみが適応の影響を受ける（ドメインによっては
一致ゼロ = ZCOCO が構造的に 0.504 になる。これは手法の性質であり バグではない
ことを事前に明記しておく）。公式にはタスク途中の評価が存在しないため、
warmup フェーズ中の val は「全プロンプトクラスに warmup_lora_a を適用」とする
（学習時 forward の意味論と一致させる。相談事項④）。

### 3.5 逐次学習の機構（本実験では不活性）

単発ドメイン学習では、fetch/merge（式3の学習済み分岐）と B の融合（式4）は
発火しない（式4は第 2 タスク以降のみ）。exp_021 で検証されるのは
warmup→specialization の分業、共有 B、確率的クラス特化、クラス条件付き推論
であり、VCS 的なライブラリ管理は逐次実験で初めて動く。この限界を design.md に
明記する。

### 3.6 DDP の注意

specialization 中は選ばれたクラスの A のみが勾配を持ち、他クラスの A は
その iteration で不使用になるため、config で `find_unused_parameters=True` を
設定する（若干のオーバーヘッドを許容）。

### 3.7 activation checkpointing の掛け直し（2026-07-13 追記）

上流 `grounding_dino_layers.py` の `num_cp` は fairscale の
`checkpoint_wrapper`（再入版）を encoder の layers と fusion_layers に適用する。
再入版は backward を二度走らせるため DDP がパラメータの ready を二重にマークし、
`Expected to mark a variable ready only once` で落ちる。可変な使用パラメータ集合
（クラス別 A）とは本質的に両立しない。

対処として config では上流の `encoder=dict(num_cp=0)` で再入版を無効化し、
`DitHubGroundingDINO._enable_checkpointing()` が PyTorch の非再入版
（`torch.utils.checkpoint(..., use_reentrant=False)`、DDP 併用時の推奨実装）を
encoder の先頭 `encoder_cp` 層（既定 6）の layers / fusion_layers / text_layers に
掛け直す。`nn.Module` を入れ子にせず `forward` 属性のみを差し替えるため、
state_dict のキーは変わらず既存 ckpt と互換である。

数値検証（2026-07-13）: 同一初期値・同一入力で `encoder_cp` の 0 と 6 を比較し、
loss は完全一致（差 0）、LoRA 勾配の最大絶対差は 1.1e-6（float32 の再計算に伴う
丸めの範囲）、ピークメモリは 4375 MiB → 2595 MiB。したがって checkpointing の
有無で学習結果は変わらない。

**必要になった経緯**: 標準版の videogames のみ 87 クラスで `max_text_len=512` の
ため、融合層の注意行列（画像トークン約 13000 × テキストトークン 512）が倍増し、
checkpointing 無し（num_cp=0）では 40 GB を超えて CUDA OOM で落ちた。掛け直し後は
8〜18 GB で収まる。他 5 ドメインは 256 トークンで影響を受けない。underwater と
aerial は checkpointing 無しで学習済みだが、上記の数値同一性より再学習は不要。

## 4. ハイパーパラメータの決定表（ユーザー裁定 2026-07-12: 2 体制を両方実行）

共通（両体制）: seed=0 明示、dn 有効（裁定②）、LoRA r=16 / alpha=8（scaling
0.5）/ dropout 0、対象層規則 = out ≥ 128 の Linear + 除外リスト +
memory_trans_fc 含む（裁定①）、λ_A=0.3 / λ_B=0.7（単発では不活性）、
追加損失なし、clip_grad max_norm 0.1、学習対象は LoRA パラメータのみ。

| 項目 | 標準版（exp_xxx 共通） | 公式準拠版 |
|---|---|---|
| スケジュール | 20 epoch, milestones [15] | 3000 iter 固定, iter 1200 で lr ×0.1 |
| フェーズ切替 | epoch 10 終了時（等分） | iter 1500（等分） |
| lr / wd | AdamW 1e-4 / 1e-4 | AdamW 1e-3 / 1e-2 |
| batch / GPU | 4 GPU 分散（batch 4×4） | batch 2 / 1 GPU |
| lr減衰とフェーズの位置 | 減衰(15ep)が切替(10ep)の後 | 減衰(40%)が切替(50%)の前 |

既知の構造差: 標準版では specialization の前半 5 epoch が減衰前の lr で走る
（公式は specialization 全体が減衰後）。標準版は exp_xxx との比較可能性を、
公式準拠版は実装忠実性の検証と論文水準の再現を担う（exp_020 と同じ二本立て）。
標準版の lr 1e-4 / wd 1e-4 では LoRA の成長が遅い可能性があり、適応 mAP の
立ち上がりを監視して、立ち上がらない場合は中断して相談する。

## 5. 公式実装からの意図的な逸脱

1. dn を無効化しない（全基準線・exp_020 と条件を揃える）。
2. スケジュールを 3000 iter 固定でなく 20 epoch にする（ユーザー方針）。
3. warmup フェーズ中の val に warmup_lora_a を適用する（公式に途中評価は無い。
   学習時 forward と同じ意味論の自然な拡張）。
4. per_class_lora_A の確保を全タスク一括でなく当該ドメイン分のみとする
   （公式の全タスク分確保は B=0 のため出力に影響しない実装上の癖）。
5. decoder の attention 分解を行わない（LoRA 対象外のため恒等変換）。

## 6. 実装後・学習前の検証（check_dithub_setup.py）

1. **パラメータ監査**: 学習可能 = lora_* のみ。対象層の列挙と総数の報告
   （除外リスト漏れの検出。bbox_head や decoder に LoRA が入っていないこと）。
2. **attention 分解の等価性**: DitHubTextSelfAttention と元の
   nn.MultiheadAttention の出力一致（誤差 ~1e-6）。
3. **θ0 等価性**: B=0 の初期状態で検出出力が素の MM-GDINO と**厳密一致**
   （ZiRa と違い初期値 1e-8 の誤差すら無いはず）。ckpt ロードの
   in_proj 分解読み込みが正しいことも兼ねる。
4. **勾配疎通**: warmup 中は warmup_lora_a と shared_lora_b のみに勾配、
   per_class_lora_A には勾配なし。切替後はサンプル選択されたクラスの A のみに勾配。
5. **enable_per_class の単体テスト**: 未学習クラス = warmup コピー、
   学習済みクラス = 式3の重み付き融合、の数値確認。
6. **評価合成の検証**: 評価 forward の出力が「クラス別 ΔW の平均を手計算で
   重みに加算したモデル」と一致（誤差 ~1e-5）。
7. **ZCOCO 名前重複の事前調査**: 各ドメインのクラス名と COCO 80 クラス名の
   一致リストを出力（ZCOCO が θ0 と一致するはずのドメインを事前に特定）。

## 7. 実行範囲の提案（design.md で確定）

全 6 ドメインの単発ドメイン学習 + ZCOCO 評価（exp_019/020 と同一プロトコル）。
これにより ZiRa（exp_020）と DitHub の 2 手法が同じ表に並ぶ。逐次学習
（fetch / merge / 式4 が動く設定）は別実験として設計する。

## 8. 相談事項の裁定記録（2026-07-12）

- ① memory_trans_fc: **含める**（コード準拠。ユーザー裁定）。
- ② dn: **有効のまま**（ユーザー裁定）。
- ③ フェーズ配分・体制: **標準版（20ep, 10+10, lr 1e-4）と公式準拠版
  （3000 iter, 1500+1500, lr 1e-3, wd 1e-2）の両方を実行**（ユーザー裁定）。
- ④ warmup 中の val: **warmup_lora_a を全プロンプトクラスに適用**。報告用の
  best checkpoint は **specialization 期（epoch 11〜20）から選ぶ**。warmup 期の
  val が上回った場合は「クラス特化が効いていない」観察として別途記録
  （ユーザー裁定）。
- ⑤ 実行範囲: **単発 6 ドメイン**（標準版・公式準拠版の両体制。ユーザー裁定）。
- 追加裁定: B のタスク間融合（式4）は merge_dithub.py によるタスク境界の
  オフライン処理として実装（B_prev = load_from の checkpoint 内の B）。
  exp_021 では実行経路に乗らず、単体テストまで。
