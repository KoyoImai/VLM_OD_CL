# exp_042 設計書: EWC と InfLoRA の実装と ODinW-13 逐次学習・評価

作成 2026-08-15。**実行前のユーザー承認ゲート**（行動原理1・3）。
関連: [[../exp_034/design]]（ZiRa 再現・ODinW-13 共通評価系の確立）、
[[../exp_037/design]]（DitHub 再現）、[[../exp_038/design]]（素の LoRA）、
[[../../experiment_notes/note12]]。

## 0. 位置付け

ODinW-13 の共通評価系（同じ θ0・seed 42 のタスク順・同じデータ版・plain 評価 config）には
ZiRa（exp_034）・DitHub（exp_037）・素の LoRA（exp_038）が載っている。本実験は従来の
継続学習手法から**正則化系の EWC** と**部分空間直交化系の InfLoRA** を実装し、同じ土俵に載せる。

note12 の方針は「論文と公式実装に従って実装する」。調査の結果（§2.2・§2.3 の根拠欄）:

- **EWC に公式実装は無い**（原論文 PDF にコード公開の記載なし）。論文本文と、スター数上位の
  公開実装 4 件（Avalanche 2,083★ / GMvandeVen 1,881★ / Mammoth 833★ / GEM 408★、
  2026-08-15 調査）の実装慣行に従う。
- **InfLoRA は公式実装がある**（liangyanshuo/InfLoRA、CVPR2024、113★）。
  `methods/inflora.py` と `models/vit_inflora.py` の手続きに従う。

## 1. 目的

EWC と InfLoRA を MM-Grounding DINO 上に実装し、実装検証を通した上で、ODinW-13 の
13 タスク逐次学習・評価を行う。EWC の λ は本走 3 水準（10²/10³/10⁴）の並行比較で見る（§2.4）。

**仮説と判定基準は実験ノート（note12）でユーザーが確定する。** 本書は測るものと条件を定める。

## 2. 条件

### 2.1 共通の学習枠（exp_034/037/038 の ODinW-13 枠。lr / wd のみ変更）

| 項目 | 値 | 経緯 |
|---|---|---|
| iteration | 3000 / タスク、batch 2（1 GPU） | exp_037/038 と同一 |
| optimizer | AdamW **lr 1e-4 / weight_decay 1e-4** | **2026-08-15 指示**。exp_038 で lr 1e-3 は崩壊、lr 1e-4（条件C）で解消した実測に基づく。wd は RF100 のフル FT 系と同じ値 |
| lr schedule | MultiStepLR milestones=[1200], gamma=0.1 | exp_037/038 と同一 |
| grad clip | max_norm 0.1（L2） | 同上 |
| dn | 無効（`use_dn=False` ＋ `NoDNGroundingDINOHead`） | 同上 |
| num_classes | 256 / seed 0 / AMP・EMA なし | 同上 |
| データ・逐次順 | exp_034 の 13 タスク定義・seed 42 順（pistols → PascalVOC → … → ShellfishOpenImages） | 同上 |
| 逐次方式 | 各タスク終了時に θ へ反映（EWC はそのまま、InfLoRA はマージ）→ plain config で評価 | exp_038 §2.3 と同型 |
| 評価 | 学習済みタスク 1..t ＋ ZCOCO。t=0（θ0）も測って exp_034 実測と照合 | exp_038 §3 と同一 |

### 2.2 EWC の仕様（2026-08-15 の議論で確定）

| 項目 | 決定 | 根拠 |
|---|---|---|
| 学習対象 | **config で構成要素単位に選択**。EWC 対象＝学習対象で、対象外は凍結（requires_grad=False） | ユーザー指示 |
| ペナルティ | 原論文のタスク別二次項を **2 バッファ（A=ΣF_k、B=ΣF_k·θ*_k）で等価実装**。`L = L_t + (λ/2)·Σ_i [A_i·θ_i² − 2·B_i·θ_i + const]` | 論文本文 §2「the sum of two quadratic penalties is itself a quadratic penalty」。Avalanche 既定 `separate` と同じペナルティ |
| Fisher | **対角・経験 Fisher**：タスク学習終了時 θ*_t で、学習と同じ検出損失（dn 無効の全損失項）の勾配²を平均 | Avalanche・GEM と同方式。GMvandeVen では `true` オプション。分類用の厳密 Fisher（全ラベル列挙）は検出に適用不能 |
| Fisher のバッチ | **batch=1**（1 枚ずつ勾配²） | GMvandeVen 既定・Mammoth と同じ。バッチをまとめると勾配相殺で過小推定 |
| Fisher のサンプル数 | **config で上限枚数を指定**。タスクの訓練枚数が上限以下なら全数、超えるなら seed 0 の無作為抽出 | ユーザー指示（2026-08-15）。公開実装の既定は全数、GMvandeVen に上限機能（fisher_n） |
| λ | **config で指定**（ドライバが --cfg-options で明示）。本走は 3 水準 {10², 10³, 10⁴} を並行比較（§2.4。2026-08-15 変更） | ユーザー指示 |

### 2.3 InfLoRA の仕様（公式実装 liangyanshuo/InfLoRA に準拠）

公式実装の手続き（`methods/inflora.py` の `_train`・`update_DualGPM`、
`models/vit_inflora.py` の `Attention_LoRA`）:

1. **各タスクの学習前**に、訓練データを推論モードで流し、各挿入層の**入力の共分散
   （Σ xᵀx / n）**を蓄積する（勾配空間の GPM 流近似）。
2. t=1 はその SVD の上位 r 主成分、t≥2 は **DualGPM のメモリで射影してから** SVD の
   上位 r 主成分で、**次元削減行列（lora_cl の記法で A、r×d_in）を設計して固定**する
   （公式は 1/√3 を掛ける）。**学習するのは B（d_out×r、ゼロ初期化）のみ**。
3. 学習後、もう一度入力共分散を蓄積し、**DualGPM のメモリを更新**する。閾値は
   `lamb + (lame−lamb)·t/T` の線形増加（公式 DomainNet 設定: 0.95 → 1.0）。
   メモリは 'remove'/'retain' の双対表現で層次元の半分以下に保つ。
4. 分岐は forward で `Σ_t B_t·A_t` を W に加算しており、**タスク終了時に W へマージするのと
   等価**。本実装は lora_cl のマージ流（タスク後に θ へ焼き込み）で同じ関数を実現する。
   DualGPM のメモリ（feature_list / project_type）はタスク間でファイルとして持ち越す。

本プロジェクトでの設定:

| 項目 | 決定 | 根拠 |
|---|---|---|
| 挿入箇所 | **config で選択**。選択肢は `projects/lora_cl` と同一（Swin / BERT / feature enhancer（MHA 分解込み）/ text_feat_map など） | ユーザー指示（2026-08-15）。公式は ViT の k・v 射影のみだが、対象モデルが違うため挿入箇所は本プロジェクトの枠組みに従う |
| r・DualGPM 閾値（lamb→lame） | config で指定。本走の値は §9 | 公式 DomainNet は r=30・0.95→1.0 |
| スケーリング | 公式の `ΔW = B·A`（α/r 係数なし、A に 1/√3）に合わせる | 公式実装準拠 |
| 共分散推定のサンプル数 | Fisher と同じ方式（**上限枚数を config 指定**、以下なら全数） | 公式は全数 2 パス。EWC と整合させる |
| optimizer | 共通枠の AdamW lr 1e-4 / wd 1e-4（公式の SGD+cosine は使わない） | 共通枠を優先（§2.1）。逸脱として §7 に記録 |

### 2.4 EWC の λ: 本走で 3 水準を並行比較（2026-08-15 変更）

~~当初は 2 タスクのパイロットで λ を決める二段構えを採用したが~~、パイロットは行わず、
**本走（13 タスク）を λ ∈ {10², 10³, 10⁴} の 3 水準で 3 本走らせて比較する**
（2026-08-15 のユーザー指示で変更）。λ=0 の対照（ペナルティ無しフル FT の 13 タスク走）は
4 本目が必要になるため含めない。λ の適否は本走の適応・保持・ZCOCO の釣り合いで
ユーザーが判定する。λ=10⁴ の学習が発散しないか（IncDet の勾配爆発の指摘。§6）は
本走の序盤ログで確認する。

### 2.5 本走の構成（4 GPU 並行）

| GPU | 内容 |
|---|---|
| 0 | EWC 13 タスク（λ=10²） |
| 1 | EWC 13 タスク（λ=10³） |
| 2 | EWC 13 タスク（λ=10⁴） |
| 3 | InfLoRA 13 タスク（232 層・r=16） |

4 本は独立で、途中で止まったものだけ同じコマンドで再開できる。
EWC の work_dir・状態・評価出力は λ ごとに分離する（`ewc_lam{λ}_*`）。

## 3. 判定材料（exp_038 §3 と同形式）

0. θ0 の 13 タスク＋ZCOCO（exp_034 実測と照合し評価系の同一性を確認）
1. 適応: 13 タスクの最終 mAP 平均と θ0 からの変化（タスク別内訳も）
2. 忘却: 各タスクの「学習直後」対「t=13 時点」
3. ZCOCO の t=0〜13 推移
4. 既存値との相対: ZiRa（exp_034）・DitHub（exp_037）・素の LoRA（exp_038 条件A/B/C）
5. 13×13 推移表
6. EWC の λ 3 水準（10²/10³/10⁴、本走間）の適応・保持・ZCOCO の比較

## 4. 実装

### 4.1 作るもの

| 成果物 | 内容 |
|---|---|
| `projects/ewc_cl/` | EWC 実装（新規）。学習対象の選択と凍結、2 バッファのペナルティ（loss への加算）、タスク終了時の Fisher 推定（batch 1・上限枚数・seed 0 抽出）と 2 バッファ更新・保存 |
| `projects/lora_cl/` への追加 | InfLoRA モジュール。入力共分散の蓄積フック、A の SVD 設計（射影込み）、DualGPM メモリの更新・保存、B のみ学習の凍結制御。既存の挿入箇所選択・マージ・MHA 分解は再利用 |
| `experiments/exp_042/` | 本走（EWC 3 λ ＋ InfLoRA）の config 生成、逐次ドライバ、実行前検証 `check_exp042_setup.py`、README |
| `papers/EWC_implementation_notes.md` / `papers/InfLoRA_implementation_notes.md` | 論文・公開実装と本実装の対応（式・値・逸脱）を記録 |

mmdetection 本体（`mmdet/`）は変更しない（lora_cl と同じ方針）。

### 4.2 実行前に確認する項目（実装検証。本環境）

1. EWC: 選択した対象だけが requires_grad=True になり、対象外が 1 step で不動であること
2. EWC: ペナルティの勾配が解析値 `λ·(A·θ − B)` と一致すること（小モデルで数値照合）
3. EWC: Fisher 推定が batch 1・上限枚数・seed 0 抽出で走り、全要素が非負・有限であること。
   2 バッファ更新がタスク別保持（Avalanche `separate` 相当の素朴実装）と一致すること
4. InfLoRA: A が設計値のまま学習中不変で、B のみ更新されること
5. InfLoRA: t≥2 で設計した A の行空間が DualGPM メモリと直交すること（'remove' 型で
   ‖M^T A^T‖ が数値誤差の範囲）
6. InfLoRA: マージ後のモデルが分岐付き forward と一致すること（既存 merge_lora の検証流用）
7. 両手法: 実データ 1 step で loss が有限、θ0 からの 1 タスク目が exp_038 と同じデータ・
   評価経路に載ること（t=0 評価が exp_034 実測と一致）

## 5. 実行環境とコスト（本環境。クラスタは使わない）

exp_038 条件C の実測（3000 iter ≈ 28 分、評価込み 1 タスク周期 ≈ 47 分、A100 1 枚）を基準。
EWC はフル重みの backward で 1 タスク 40〜60 分、InfLoRA は LoRA 並み＋共分散 2 パスと見積もる。

| 内容 | 概算 |
|---|---:|
| 本走 EWC（13 タスク × 3 λ、3 GPU 並行） | 実時間 13〜16 h ＋ Fisher・評価 |
| 本走 InfLoRA（13 タスク） | 12〜15 h |

4 GPU をすべて使用（§2.5 の割り当て。exp_041 はクラスタなので競合しない）。
並行実行なので実時間は最長の 1 本と同じ 14〜18 h 程度。

## 6. リスク・懸念

- **公式 InfLoRA との逸脱**: optimizer（公式 SGD+cosine → 本実験 AdamW+MultiStep）、
  挿入箇所（公式 k・v のみ → 本実験は config 選択）、対象が分類 ViT → 検出器。
  「公式実装に従う」のは A の設計・DualGPM・B のみ学習という**手法の核**であり、
  最適化まわりは共通枠を優先する。読み取りにはこの逸脱を前提とする。
- **EWC の λ スケール**: 検出損失と Fisher 正規化に依存し、既知の値（原論文 Atari の 400 等）は
  移らない。本走を 3 水準（桁刻み）で走らせる設計にした理由そのもの。
- **二次ペナルティの勾配爆発**: IncDet（TNNLS 2020）は検出で EWC の二次損失が勾配爆発を
  起こしやすいと報告している（`papers/CL-detection/EWC-in-detection_survey.md`）。
  共通枠の grad clip 0.1（L2）が緩衝になるが、λ=10⁴ の本走序盤で学習が発散しないかを
  確認する。なお VLM 検出器（Grounding DINO 系）への EWC 適用の前例は調査で見つからず、
  ドメイン増分検出ベンチマーク TiROD では EWC 系（IncDet）はリプレイ系に大差で劣る実測がある。
- **Fisher 推定時の dropout**: 推定は損失経路（model.loss）で行うが dropout は無効化する
  （Avalanche・GMvandeVen が model.eval() で推定するのに対応）。実装時に検証項目へ含める。
- **InfLoRA の共分散行列**: 層ごとに d_in×d_in（最大 1024²）を持つ。232 層でも合計約 1 GB
  以内で、SVD も層あたり秒オーダーの見込み（実装時に実測して README に記録する）。

## 7. 承認をお願いする範囲

§2 の条件で、(1) EWC・InfLoRA の実装と実装検証（§4.2）、(2) 本走（§2.5、
EWC 13 タスク × 3 λ ＋ InfLoRA 13 タスク、4 GPU 並行）を本環境で行うこと。
(1) は 2026-08-15 完了（5/5・8/8 ＋ 挙動検証 ＋ 公開品質監査）。

## 8. 確定済みの判断（2026-08-15 の議論）

- EWC の学習対象＝EWC 対象を config で選択、対象外は凍結。「全モジュール」も選択可能にする
- **本走の EWC 学習対象は全モジュール**（フル FT＋全体ペナルティ。パラメータ
  178,649,629 個、2 バッファ約 1.43 GB。dn 無効で損失に関与しないパラメータは Fisher=0 に
  なるが学習でも動かないため実害なし）
- InfLoRA の挿入箇所も config で選択、選択肢は lora_cl と同一
- lr 1e-4 / weight_decay 1e-4、他は ODinW-13 共通枠
- EWC は 2 バッファ実装、対角・経験 Fisher、batch 1、上限枚数 config
- λ は当初パイロットで決める二段構え → **本走 3 水準（10²/10³/10⁴）の並行比較に変更**
  （2026-08-15。パイロットは行わない。λ=0 対照は含めない）
- **4 GPU 並行**: EWC 3 λ ＋ InfLoRA（§2.5。2026-08-15 指示）
- **本走の InfLoRA は exp_038 条件A と同じ 232 層（Swin・BERT・feature enhancer・
  text_feat_map）・r=16**（2026-08-15 確定。素の LoRA と挿入箇所・ランクを揃え、
  A の部分空間設計の有無だけの比較にする）
- **Fisher・入力共分散の推定サンプル数は上限 1,000 枚**（2026-08-15 確定。タスクの訓練
  枚数が 1,000 以下なら全数、超えるなら seed 0 の無作為抽出。上限に掛かるのは
  PascalVOC / EgoHands / pistols / CottontailRabbits の 4 タスク）

## 9. 承認前の判断点（すべて確定済み）

1. パイロットと本走の EWC 学習対象 → **全モジュール**（2026-08-15。§8）。
2. 本走の InfLoRA 挿入箇所と r → **exp_038 条件A と同じ 232 層・r=16**（2026-08-15。§8）。
3. Fisher・共分散の上限枚数 → **1,000 枚**（2026-08-15。§8）。
