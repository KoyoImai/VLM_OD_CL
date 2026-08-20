# exp_045 設計書: EWC と InfLoRA の Roboflow100 6 ドメイン逐次（バッファ不使用）

作成 2026-08-16。**実行前のユーザー承認ゲート**（行動原理1・3）。
関連: [[../../experiment_notes/note15]]（本実験のノート）、
[[../exp_042/design]]（EWC / InfLoRA の実装と ODinW-13 実測）、
[[../exp_024/design]]・[[../exp_040/design]]（リプレイフリー＝同一枠の対照）、
[[../exp_039/design]]・[[../exp_041/design]]（ZiRa / DitHub の RF100 系列）。

## 0. 位置付け

exp_042 で実装・検証済みの EWC（正則化系）と InfLoRA（部分空間直交化系）を、
RF100 の 6 ドメイン逐次に載せる。**バッファ不使用**なので、対照の中心は同じ枠の
リプレイフリー（exp_024＋exp_040 接続）であり、バッファを使う系列
（リプレイ / 蒸留 / ZiRa・DitHub リプレイ有り）とは「バッファなしで正則化・
構造制約だけでどこまで守れるか」という軸で比較する。

## 1. 目的

EWC と InfLoRA の 2 条件を、underwater → electromagnetic → videogames → aerial →
microscopic → documents の 6 ドメインで学習・評価する（2026-08-16 確定）。
**仮説と判定基準は note15 でユーザーが確定する。**

## 2. 条件

### 2.1 共通の学習枠（リプレイフリー系列と同一。note15「蒸留やリプレイに揃える」）

| 項目 | 値 |
|---|---|
| epoch / スケジュール | 20 epoch、MultiStepLR milestones=[15]、gamma=0.1 |
| optimizer | AdamW lr 1e-4 / weight_decay 1e-4、grad clip 0.1（L2） |
| バッチ | **バッファ不使用**: 現在ドメインのみ、batch 4/GPU・DefaultSampler（4 GPU 合計 16。exp_024/exp_040 replayfree と同一） |
| dn / num_classes / seed | **有効** / 256 / 0 |
| データ | ODVGDataset ＋ RandomSamplingNegPos（既存系列と同一） |
| 実行 | MPRG クラスタ、4 GPU × 2 ジョブ並行（2026-08-16 確定） |

学習 config はデータ・スケジュールとも exp_024 の `fullft_replayfree_<domain>.py`（前半3）
と exp_040 の `replayfree_<domain>.py`（後半3）を継承し、モデルだけを差し替える。

**dn の注記**: exp_042（ODinW-13 枠）は dn 無効だったが、RF100 枠は dn 有効。
枠に従い両手法とも dn 有効で走らせる（2026-08-16 確定。exp_042 との読み比べ時の但し書き）。

### 2.2 EWC 条件（実装は projects/ewc_cl。exp_042 で検証済み）

| 項目 | 値 | 経緯 |
|---|---|---|
| 学習対象 = EWC 対象 | **全モジュール**（フル FT ＋ 全体ペナルティ） | exp_042 と同一（2026-08-16 確定） |
| λ | **10³**（1 水準） | 2 ジョブ構成のため 1 水準。exp_042 の 3 水準比較でタスク保持の最終 Avg が最良だった値（0.3089。10²=0.2665 / 10⁴=0.2478）を採用（2026-08-16 確定） |
| Fisher | 対角・経験 Fisher、batch 1、**上限 1,000 枚**（seed 0 無作為抽出）、各タスク終了時に 2 バッファへ蓄積 | exp_042 と同一 |
| 事前学習への制約 | **無し**（保護対象は学習済みタスクのみ。t=1 は無制約フル FT） | EWC 原論文の標準形。exp_042 §2.2 と同一 |
| encoder checkpointing | **num_cp=0**（2026-08-19 修正） | 既定の fairscale 再入型 cp は「EWC ペナルティ（θ の forward 外使用）× DDP」と衝突し、t=2 でペナルティ活性化と同時に "marked as ready twice" で停止（2026-08-19 クラスタで発生・最小再現で機構を実証）。§6 の但し書き参照 |

### 2.3 InfLoRA 条件（実装は projects/lora_cl。exp_042 で検証済み）

| 項目 | 値 | 経緯 |
|---|---|---|
| 挿入箇所 / r | **232 層（Swin・BERT・feature enhancer・text_feat_map）/ r=16**、alpha=16（scaling=1） | exp_042 と同一（2026-08-16 確定） |
| A の設計 / B のみ学習 / タスク後マージ | 公式実装準拠（入力共分散の DualGPM 射影 → SVD） | exp_042 と同一 |
| DualGPM 閾値 | lamb=0.95 → lame=1.0 線形、**総タスク数 T=6** | 公式 DomainNet 設定。T のみ本実験に合わせる |
| 共分散の推定 | **上限 1,000 枚**（seed 0）、学習前後の 2 パス | exp_042 と同一 |
| ランク不足の対処 | 実効ランク（S > 1e-3·S[0]）超の A 行はゼロ化 | 実装済み。RF100 は多クラスプロンプトのため ODinW の 1 クラスタスクより起きにくい見込み（§4.2 で実測） |
| アダプタの学習率 | **RF100 枠の paramwise（backbone / language_model に lr_mult=0.1）をアダプタにも適用**: Swin 内 51 層・BERT 内 72 層の lora_B は lr 1e-5、encoder 108 層・text_feat_map は 1e-4（実測確認 2026-08-18） | 継承元の paramwise が名前の部分一致で掛かる。exp_042（ODinW 枠）は全 232 層一様 1e-4 だったため手法内対照の但し書き。**このまま進めると決定（2026-08-18、ユーザー判断）** |

### 2.4 条件一覧（2 ジョブ）

| # | 条件 | 学習パラメータ数 |
|---|---|---:|
| 1 | EWC（λ=10³） | 172.98M（全モジュール） |
| 2 | InfLoRA | 2.96M（lora_B のみ 232 層） |

## 3. 判定材料

**最終的な判定はユーザーが行う（note15）。**

1. 各 t（1〜6）での学習済み全ドメイン mAP と ZCOCO。
2. リプレイフリー（exp_024＋exp_040 接続）との対照＝同一枠・バッファ無しの中心対照。
3. バッファ使用系列（リプレイ exp_027/040、蒸留E exp_035/040、ZiRa / DitHub exp_039/041）
   との横並び（6 ドメイン表への追加）。
4. exp_042（ODinW-13）との手法内対照（枠が違うことの但し書き付き: dn・スケジュール・
   データ形式・逐次長）。

## 4. 実装

### 4.1 作るもの（実装本体は exp_042 で完成済み。実験側のみ）

| 成果物 | 内容 |
|---|---|
| `gen_configs.py` | 学習 config 12 本（`{ewc,inflora}_{6ドメイン}.py`）。exp_024 / exp_040 の replayfree config を継承しモデルだけ差し替え |
| `run_ewc.sh` / `run_inflora.sh` | 逐次ドライバ。EWC は「学習 → Fisher 推定・状態蓄積 → 評価」、InfLoRA は「設計 → 学習 → メモリ更新 → マージ → 評価」（各タスク） |
| `sbatch_ewc.sh` / `sbatch_inflora.sh` | クラスタ投入 2 本 |
| `check_exp045_setup.py` | 実行前検証（§4.2） |
| `README.md` | クラスタ手順（専用クローン） |

### 4.2 実行前に確認する項目（本環境）

1. 12 本の config が build でき、学習設定・データ経路が replayfree 系列と一致
   （機械的 diff: model 以外は exp_024 / exp_040 と同一。dn 有効・batch 4・DefaultSampler）
2. EWC: 対象 all・λ/state の --cfg-options 到達性（負テスト込み）。ODVG 経路
   （RandomSamplingNegPos）での実データ 1 step が有限、Fisher 推定が走ること
3. InfLoRA: dn 有効・ODVG 経路で 232 層挿入・B のみ学習・実データ 1 step が有限。
   共分散収集 → 設計 → 学習 1 step → メモリ更新（T=6 の閾値）→ マージの一連が通ること
4. RF100 プロンプトでのテキスト側共分散の実効ランクの実測（ランク不足の有無を記録）
5. VRAM 実測（batch 4/GPU）
6. クラスタ側パス（/local_cache ステージング）で config が解決すること

## 5. 実行環境とコスト

**MPRG クラスタ、2 ジョブ並行**（各 4 GPU、a6000_ada、`--exclude=node03`）。
作業ディレクトリは**専用クローン `/home/kouyou/VLM_OD_CL_exp045`**（これまでの運用に
合わせた前提。変更があれば指示いただきたい）。exp_043（2）・exp_044（1）と合わせて
5 本になるため、同時実行 4 本の制限により 1 本はキュー待ちで自動開始される
（投入は 8 本まで可。2026-08-16 確認済みの運用）。

| ジョブ | 内容 | 見積もり |
|---|---|---:|
| `exp045_ewc` | EWC 6 ドメイン＋Fisher＋評価 | 約 40〜45 h |
| `exp045_inflora` | InfLoRA 6 ドメイン（設計・メモリ更新・マージ込み）＋評価 | 約 30〜40 h |

`--time=96:00:00`。転送が必要な `.pth` は無い（θ0 から開始）。
ディスクは ckpt 約 24 GB（EWC 6 本 ＋ InfLoRA work_dir 6 本・merged 6 本）
＋ EWC 状態（各タスク後 約 1.4 GB × 6）＋ InfLoRA 設計・メモリ（小）。

## 6. リスク・懸念

- **EWC はゼロショット（事前学習）を保護しない**（§2.2）。exp_042 では t=1 のフル FT で
  ZCOCO が崩壊した。RF100 枠（dn 有効・RandomSamplingNegPos・lr 1e-4・20 epoch）では
  崩れ方が異なり得るが、これは測る対象そのもの。
- **InfLoRA の共分散収集は dn 有効の loss 経路で行う**（exp_042 は dn 無効で検証）。
  収集は forward の入力の記録であり dn の有無に依存しない設計だが、§4.2 で一連を実測する。
- EWC の状態ファイル（約 1.4 GB/タスク）を 6 回書くため、クローンのディスク消費に注意。
- **EWC のみ encoder checkpointing 無効（num_cp=0）**（2026-08-19 修正）。対照の
  replayfree / InfLoRA は num_cp=6 のままで、checkpointing の有無は勾配を浮動小数点
  レベルで変える（exp_039 §7 の実測で相対 〜1e-1。損失の定義・データ・スケジュールは
  同一）。学習の数学的定義は変わらないが、数値条件の差として記録する。
  static_graph=True で cp を保つ代替案は検証コスト（exp_046 完了待ち）とのトレードで
  見送り（2026-08-19 ユーザー判断）。なお t=1（underwater）は修正前に num_cp=6 で
  学習・Fisher 推定・評価まで完了しており有効なため取り直さず、**t=1 は num_cp=6・
  t≥2 は num_cp=0 の混在**とする（t=1 はペナルティ不活性＝素のフル FT と同条件で、
  num_cp=6 は replayfree と揃っている。2026-08-19 判断）。
- **DDP は `static_graph=True`（EWC のみ）**（2026-08-20 修正）。上の num_cp=0 は encoder の
  fairscale 再入型 cp を切るだけで、**Swin backbone の `with_cp=True`**（事前学習 config、
  `mmdet/models/backbones/swin.py:375` が `use_reentrant` 未指定＝再入型）は残っていたため、
  同じ機序が t=2 で再発した（2026-08-20 クラスタ。
  `backbone.stages.3.blocks.1.ffn.layers.1.bias has been marked as ready twice`）。
  本環境 2 GPU（A100 40GB、batch 4/GPU、λ=10³、実 Fisher 状態）で同一パラメータ・
  同一 index 169 として再現したうえで 2 案を実測した:

  | 案 | 結果 | allocated ピーク | iteration 時間 |
  |---|---|---:|---:|
  | `backbone.with_cp=False` | **iter 13 で OOM**（2 回再現。確保総量 39.42 GiB） | 28,119 MiB | 1.60 s |
  | `static_graph=True`（cp 維持） | **303 iteration 完走**（`loss_ewc` 全 iteration に出現、0.122→1.51） | 31,226 MiB | 1.49 s |

  採用は `static_graph=True`（2026-08-20 ユーザー承認）。再入型 checkpointing を DDP 下で
  使うための PyTorch 公式の構成であり、学習の計算そのもの（cp あり）は t=1・対照
  （replayfree / InfLoRA / LoRA）と同一のまま、変わるのは DDP が勾配集約の順序を初回
  イテレーションで確定して再利用する点だけである。`encoder=dict(num_cp=0)` は残すため、
  上記の num_cp 混在の記述は変わらない。**メモリの但し書き**: ペナルティ活性時の実測は
  31.2 GB（batch 4/GPU）で、§4.2 で記録した 5.1 GB（ペナルティ不活性の 1 step 検証）とは
  桁が違う。A6000 Ada 48GB には収まるが余裕は約 1.5 倍である。

## 7. 承認をお願いする範囲

§2 の条件で、EWC（λ=10³）と InfLoRA の 2 条件を RF100 の 6 ドメインで逐次学習・
評価すること。**実行は MPRG クラスタ 2 ジョブ**（§5）。
本環境で行うのは config 作成と実行前検証（§4.2）まで。

## 8. 確定済みの判断（2026-08-16 の議論）

- 対象は EWC と InfLoRA の 2 条件、バッファ不使用、基本ハイパラは既存 RF100 系列と同一
- ドメインは 6 ドメイン通し
- クラスタ 2 ジョブ並行（1 本キュー待ち許容）
- EWC: 全モジュール・λ=10³・Fisher 上限 1,000
- InfLoRA: 232 層・r=16・DualGPM 0.95→1.0（T=6）・共分散上限 1,000
- dn は RF100 枠に従い有効
