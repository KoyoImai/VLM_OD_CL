# exp_047 設計書: 素の LoRA の Roboflow100 6 ドメイン逐次（バッファ不使用）

作成 2026-08-18。**実行前のユーザー承認ゲート**（行動原理1・3）。
関連: [[../../experiment_notes/note15]]（本実験のノート。2026-08-18 更新で LoRA を追加）、
[[../exp_045/design]]（EWC / InfLoRA の RF100。同一枠・直接の対照）、
[[../exp_038/design]]（ODinW-13 の素の LoRA。プロトコルの出典）、
[[../exp_024/design]]・[[../exp_040/design]]（リプレイフリー＝同一枠のフル FT 対照）。

## 0. 位置付け

note15 の RF100 バッファ不使用系列（exp_045: EWC / InfLoRA）に**素の LoRA**を加える。
InfLoRA（exp_045）とは挿入箇所・ランク・スケーリングを完全に揃えるため、差分は
「**A をランダム初期化して学習する（素の LoRA）か、入力部分空間から設計して固定する
（InfLoRA）か**」だけになる。フル FT（リプレイフリー exp_024＋exp_040）との対照で
「低ランク制約そのものの効果」も読める。

## 1. 目的

素の LoRA の 1 条件を、underwater → electromagnetic → videogames → aerial →
microscopic → documents の 6 ドメインで学習・評価する。
**仮説と判定基準は note15 でユーザーが確定する。**

## 2. 条件

### 2.1 共通の学習枠（exp_045 と同一）

20 epoch / MultiStepLR [15] / AdamW lr 1e-4・wd 1e-4 / grad clip 0.1 /
**バッファ不使用**（現在ドメインのみ・batch 4/GPU・DefaultSampler・4 GPU 合計 16）/
dn 有効 / num_classes 256 / seed 0 / ODVG＋RandomSamplingNegPos。
学習 config は exp_045 と同じく exp_024 / exp_040 の replayfree config を継承し、
モデルだけを差し替える。

### 2.2 LoRA の仕様（2026-08-18 確定）

| 項目 | 値 | 経緯 |
|---|---|---|
| 検出器 | `GroundingDINOLoRA`（既存。exp_038 で検証済み） | — |
| 挿入箇所 / r / alpha | **232 層（Swin・BERT・feature enhancer・text_feat_map）/ r=16 / alpha=16（scaling=1）** | InfLoRA（exp_045）と完全に同一（2026-08-18 確定）。exp_038 の alpha=8 とはスケーリングが異なる点は読み比べの但し書き |
| 初期化 | A kaiming・B=0（各タスク開始時に引き直し。開始時点は θ_{t-1} と厳密に等価） | exp_038 §2.3 のプロトコル |
| 逐次方式 | 各タスク終了時に `merge_lora.py` で base へマージ（MHA 再融合込み）。θ_t は plain 構造で、次タスクの load_from と評価に使う | exp_038 と同一 |
| dn | RF100 枠に従い**有効**（use_dn を渡さない） | exp_045 と同一 |
| アダプタの学習率 | **RF100 枠の paramwise を適用**: Swin 内 51 層・BERT 内 72 層の lora_A/B は lr 1e-5（lr_mult=0.1）、encoder 108 層・text_feat_map は 1e-4 | exp_045 InfLoRA と同一の扱い（2026-08-18 決定）。exp_042/038（ODinW 枠、一様 lr）との読み比べの但し書き |

学習パラメータ数 5.67M（A＋B。InfLoRA の 2.96M は B のみのため、学習対象の
パラメータ数は素の LoRA の方が約 2 倍になる。これは手法差の一部として記録）。

### 2.3 対照の整理

| 対照 | 差分 |
|---|---|
| InfLoRA（exp_045） | A の扱いのみ（ランダム学習 vs 部分空間設計・固定）※学習パラメータ数も 5.67M vs 2.96M |
| リプレイフリー（exp_024＋exp_040） | 低ランク制約の有無（フル FT 172.98M vs LoRA 5.67M） |
| EWC（exp_045） | 正則化 vs 構造制約（どちらもバッファ無し） |
| exp_038（ODinW の素の LoRA） | 枠が異なる（dn・スケジュール・データ形式・alpha=8） |

## 3. 判定材料

**最終的な判定はユーザーが行う（note15）。**

1. 各 t（1〜6）での学習済み全ドメイン mAP と ZCOCO。
2. §2.3 の対照（中心は InfLoRA との対比較と、リプレイフリーとの対照）。

## 4. 実装

### 4.1 作るもの（実装本体は完成済み。実験側のみ）

| 成果物 | 内容 |
|---|---|
| `gen_configs.py` | 学習 config 6 本（`lora_{domain}.py`）。exp_045 と同じ replayfree config を継承しモデルだけ差し替え |
| `run_lora.sh` | 逐次ドライバ: 学習 → `merge_lora.py` で θ_t → 学習済み 1..t ＋ ZCOCO 評価（plain 評価 config）。exp_038 のドライバを RF100 用に写す |
| `sbatch_lora.sh` | クラスタ投入 1 本 |
| `check_exp047_setup.py` | 実行前検証（§4.2） |
| `README.md` | クラスタ手順（専用クローン） |

### 4.2 実行前に確認する項目（本環境）

1. 6 本の config が build でき、学習設定・データ経路が exp_045（replayfree 継承）と一致。
   機械的 diff: model の差分が exp_045 inflora に対して「type と inflora キーの有無」だけ
   （lora 設定は同一）
2. dn 有効・ODVG 経路で 232 層挿入・A/B のみ学習（5.67M）・実データ 1 step が有限
3. マージ（merge_lora）→ plain モデルで読めること（scaling=1 の確認込み）
4. VRAM 実測（batch 4/GPU）

## 5. 実行環境とコスト

**MPRG クラスタ、1 ジョブ（4 GPU、a6000_ada、`--exclude=node03`）**。
作業ディレクトリは**専用クローン `/home/kouyou/VLM_OD_CL_exp047`**（これまでの運用に
合わせた前提。変更があれば指示いただきたい）。クラスタの他ジョブ（exp_043/044/045）と
合わせて同時実行 4 本を超える分はキュー待ちで自動開始される。

コスト: LoRA はフル FT より backward が軽く、リプレイフリー 6 ドメイン（約 37〜40 h）
と同程度かやや軽い **約 30〜38 h**（評価・マージ込み）。`--time=96:00:00`。
転送が必要な `.pth` は無い（θ0 から開始）。ディスクは work ckpt 6×約 2 GB ＋
merged θ 6×約 2 GB ≒ 24 GB。

## 6. リスク・懸念

- **ゼロショット保護の機構は無い**（base が毎タスク更新される。exp_038 §2.3 と同じ
  設計上の帰結）。ZCOCO の劣化はフル FT より小さくなる可能性が高い（更新が低ランク）
  が、InfLoRA より大きく出ることがあり得る。これは測る対象そのもの。
- alpha=16 は exp_038（alpha=8）の 2 倍のスケーリング。lr 1e-4 では exp_042 の
  InfLoRA（alpha=16）が安定に学習しており、発散の懸念は小さい。

## 7. 承認をお願いする範囲

§2 の条件で、素の LoRA の 1 条件を RF100 の 6 ドメインで逐次学習・評価すること。
**実行は MPRG クラスタ 1 ジョブ**（§5）、約 30〜38 時間。
本環境で行うのは config 作成と実行前検証（§4.2）まで。

## 8. 確定済みの判断（2026-08-18）

- 素の LoRA を exp_047 として RF100 で実行（note15 更新）。クラスタ実行（ユーザー指示）
- 挿入箇所・r・alpha は InfLoRA と完全に同一（232 層・r=16・alpha=16 = scaling 1）
- 枠は exp_045 と同一（バッファ不使用・6 ドメイン・dn 有効・既存 RF100 ハイパラ）
- 逐次方式は exp_038 のプロトコル（毎タスク B=0 から・タスク後マージ）
- アダプタの学習率は RF100 枠の paramwise のまま（Swin/BERT 内は lr 0.1×。2026-08-18 決定）
