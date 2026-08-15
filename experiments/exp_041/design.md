# exp_041 設計書: ZiRa / DitHub × リプレイ有無の後半3ドメイン学習（note11 の残り）

作成 2026-08-15。**実行前のユーザー承認ゲート**（行動原理1・3）。
関連: [[../exp_039/design]]（同条件での前半3ドメイン。本実験はその続き）、
[[../exp_040/design]]（リプレイフリー・リプレイ・蒸留Eの後半3ドメイン。データ構成を共有）、
[[../../experiment_notes/note11]]。

## 0. 位置付け

note11 は「前半3ドメイン学習済みのパラメータを初期値として後半3ドメイン
（aerial → microscopic → documents）を学習する」実験である。リプレイフリー・リプレイ・
蒸留Eの3条件は exp_040 で実施する。本実験は残りの **ZiRa / DitHub × リプレイ有無の4条件**を、
exp_039 の 3 タスク目終了時の融合済みパラメータから続けて学習・評価する。

これにより 6 条件（リプレイフリー / リプレイ / 蒸留E / ZiRa±リプレイ / DitHub±リプレイ）が
6 ドメイン通しで同じ土俵（20 epoch / lr 1e-4 / wd 1e-4 / 同一データ構成）に載る。

## 1. 目的

exp_039 の4条件を t=4〜6（aerial → microscopic → documents）へ延長し、
学習済み全ドメインと ZCOCO の推移を測る。

**仮説と判定基準は実験ノート（note11）でユーザーが確定する。** 本書は測るものと条件を定める。

## 2. 条件

### 2.1 初期パラメータ（note11 の実験条件）

いずれも exp_039 の 3 タスク目（videogames）終了時の**融合済み** ckpt。
本環境と、クラスタの `/home/kouyou/VLM_OD_CL`（exp_039 を実行したクローン）の両方に既にある。
実行用クローン（§5）へはクラスタ内コピーで持ち込む（git 管理外のため clone には入らない）。

| 条件 | 初期パラメータ |
|---|---|
| ZiRa リプレイ無し | `experiments/exp_039/zira_replayfree_merged/merged_after_t3_videogames.pth` |
| ZiRa リプレイ有り | `experiments/exp_039/zira_replay_merged/merged_after_t3_videogames.pth` |
| DitHub リプレイ無し | `experiments/exp_039/dithub_replayfree_merged/merged_after_t3_videogames.pth` |
| DitHub リプレイ有り | `experiments/exp_039/dithub_replay_merged/merged_after_t3_videogames.pth` |

note11 の 16 行目のラベルは「Dithub+リプレイ」だがパスは `zira_replay_merged` であり、
ZiRa+リプレイの誤記と解釈した（パスの側を正とする）。

融合済み ckpt の中身（継続の意味）:

- **ZiRa**: Rep+ 適用済み（`W_llrb ← W_llrb + s·W_hlrb`、HLRB 1e-8・s 0.1 に再初期化）。
  t=4 は HLRB をゼロ近傍から学習する。exp_039 の t=2→3 と同じつなぎ方。
- **DitHub**: 式4 適用済みの共有 B ＋ 前半3ドメイン全クラス（152 クラス）の A ライブラリ。
  各タスクの学習 config は現ドメインのクラスのみスロットを持ち、ライブラリの他クラス A は
  タスク終了時の `merge_dithub.py`（A 和集合）で保全される。exp_039 と同じつなぎ方。

### 2.2 揃える設定（exp_039 / exp_040 と同一）

20 epoch / MultiStepLR milestones=[15] / AdamW lr 1e-4 wd 1e-4 / grad clip 0.1(L2) /
batch はリプレイ有り 6・無し 4 / dn 有効 / num_classes 256 / seed 0 / in-training val 無し。
学習 config は **exp_040 のドメイン config を継承**し、手法固有の設定だけを上書きする
（exp_039 が exp_023/exp_024 のドメイン config を継承したのと同じ方式）。
これによりデータ経路（現在ドメイン・参照バッファ・過去プール）が exp_040 と構造的に一致する。

### 2.3 手法固有の設定（exp_039 と同一）

- **ZiRa**: ZiL λ=0.1、LLRB lr×0.2、s 初期値 0.1、タスク終了ごとに Rep+
  （`experiments/exp_020/merge_hlrb.py`）。
- **DitHub**: r=16 / alpha=8、クラス別 A ＋ 共有 B（encoder ＋ `memory_trans_fc` の 109 層）、
  warmup→specialization 切替 epoch 10、式3 λ_A=0.3、タスク終了ごとに式4 λ_B=0.7 ＋ A 和集合
  （`experiments/exp_021/merge_dithub.py`）。`encoder=dict(num_cp=0)`・`encoder_cp=6`、
  リプレイ有りは `DitHubReplayGroundingDINO`・無しは `DitHubODVGGroundingDINO`
  （2026-08-13 の exp_039 修正をそのまま使う）。

DitHub の切替点と式3 の対象。式3 は現ドメインのクラスのうち過去ドメインで学習済みの
ものに発火する。クラスの同一性は DitHub 実装の **canonical キー**
（小文字化・非英数字→`_`。`dithub_layers.py` の `canonical_key`）で判定されるため、
突き合わせも canonical キーで行った（2026-08-15 実測。当初の生文字列比較は
aerial の `Fish` を取りこぼしていたため訂正）:

| ドメイン | クラス数 | iters/epoch（概算） | 切替点（epoch 10、概算） | 式3 の対象 |
|---|---:|---:|---:|---|
| aerial (t=4) | 22 | 約 415 | 約 4,150 iter | `['Fish']`（underwater の `fish` と同一キー） |
| microscopic (t=5) | 28 | 約 599 | 約 5,990 iter | なし |
| documents (t=6) | 59 | 約 1,117 iter | 約 11,170 iter | `['object']`（videogames で学習済み） |

iters/epoch は 4 GPU × 現在ドメイン 16 枚/step（exp_039 §2.2 と同じ算出）。
画像数は aerial 6,643 / microscopic 9,576 / documents 17,866。
なお aerial は `orange-sphero` と `orange_sphero` が同一 canonical キーに衝突するため、
22 クラスに対しクラス別モジュールは 21 個になる（DitHub 実装の仕様どおり。
初期 ckpt 読込の missing/unexpected keys もこの数で照合する。check §6.3 項目5）。

### 2.4 リプレイの構成（exp_040 §2.4 と同一）

3 ソース `[現在 4, 参照 1, 過去プール 1]`、batch 6、
`CurrentEpochMultiSourceSampler(source_ratio=[4,1,1])`。
参照バッファは Objects365v1 1,000 枚固定。過去プールは学習済みドメイン各 500 枚の累積
（t=4: 前半3ドメイン、t=5: +aerial、t=6: +microscopic）。バッファ定義は
`experiments/exp_023/buffer/` に 6 ドメイン分すべて存在することを確認済み（2026-08-15）。

リプレイ有り DitHub の specialization では、参照・過去プール画像が現ドメインのライブラリに
無いクラスを持つが、`DitHubReplayLinear` が warmup_lora_a ＋共有 B を通す（exp_039 で検証済み）。

## 3. 判定材料

**最終的な判定はユーザーが行う（note11）。** 数値の閾値は設けない。

1. 各 t での学習済み全ドメイン mAP と ZCOCO（t=4, 5, 6）。
2. 前半3ドメイン（exp_039 の t=1〜3）と接続した 6 ドメイン通しの推移。
3. exp_040 の 3 条件（リプレイフリー / リプレイ / 蒸留E）との横並び比較。

## 4. 手順

exp_039 の逐次ドライバの構成を後半3ドメインに移す。各ドメイン t（4→5→6）で:

1. 学習（20 epoch）。load_from は前タスクの融合済み ckpt（t=4 は §2.1 の初期パラメータ）。
2. タスク境界処理: ZiRa は Rep+、DitHub は式4 ＋ A 和集合（初回スキップは無し。
   t=4 から常に前ライブラリと融合する）。
3. 評価: 学習済み 1..t ドメインと ZCOCO。

評価 config:

- **ZiRa**: 前半3ドメインと ZCOCO は exp_023 の `zira_eval_*.py` をそのまま使う。
  後半3ドメイン用に `zira_eval_{aerial,microscopic,documents}.py` を新設する
  （exp_026 の `eval_*.py` を継承し ZiRa の model 上書きだけ載せる。exp_023 と同じ型）。
- **DitHub**: 既存の `dithub_eval_*.py` のクラス和集合（152）は**前半3ドメインの和集合**であり、
  後半のクラスを含まない（ラベルマップとの照合で確認済み。6 ドメイン和集合は 260 クラス）。
  t≥4 のライブラリを漏れなくロードするため、**260 クラス和集合版**の
  `dithub_eval_{6ドメイン,zcoco}.py` を exp_041 に新設し、全評価をこれで行う
  （データ部は exp_023 / exp_026 の eval config を継承。評価時はプロンプト中のクラスにだけ
  A が当たるため、和集合の拡大は前半ドメインの評価値を変えない）。

## 5. 実行環境とコスト

**MPRG クラスタで実行する**（exp_039 / exp_040 と同じ運用）。クラスタ側の前提
（パーティション `a6000_ada`・`--exclude=node03`・Singularity・`/local_cache` ステージング・
Objects365 / MSCOCO の bind）は exp_039 §5.1 と同一。

作業ディレクトリは**新規クローン `/home/kouyou/VLM_OD_CL_exp041`**（2026-08-15 指示）。
sbatch の `REPO` 既定値もこれにする。clone で不足するのは git 管理外の初期パラメータ 4 本
（計約 5.2 GB）のみで、既存クローン `/home/kouyou/VLM_OD_CL` の exp_039 出力からクラスタ内で
コピーする。継承元の exp_040 config・exp_023 のバッファ定義・評価 config は git 管理下にあり
clone に入る（exp_041 の成果物と合わせて push 済みであることが前提）。

所要時間は exp_040 の見積もり（後半3ドメイン: リプレイ無し約 19 h・有り約 30 h、評価込み）を
基準に、ZiRa / DitHub も full FT と同程度とみなす。評価は t ごとに学習済み全ドメイン＋ZCOCO
（計 15 ドメイン評価＋3 ZCOCO / 条件）なので exp_039 より重い（＋数時間/条件）。

**ジョブは4本構成**（条件ごとに1ジョブ・並列。2026-08-15 指示）。

| ジョブ | 内容 | 見積もり |
|---|---|---:|
| `exp041_zira_replayfree` | ZiRa リプレイ無し | 約 25 h |
| `exp041_zira_replay` | ZiRa リプレイ有り | 約 35 h |
| `exp041_dithub_replayfree` | DitHub リプレイ無し | 約 25 h |
| `exp041_dithub_replay` | DitHub リプレイ有り | 約 35 h |

`--time=72:00:00`。4 ジョブ並列で実時間約 35 時間。ディスクは work_dir の ckpt ＋融合済み
ckpt で約 60 GB（DitHub ライブラリは A 蓄積で t=6 に約 2.5 GB へ肥大する見込み）。
`/local_cache` へのステージングはジョブごとに発生する。

同時実行は 4 ジョブまでなので、4本同時に走るのは他の実験（exp_040 の 3 ジョブなど）が
走っていない間に限られる。投入順・タイミングはユーザーが決める。

## 6. 実装

### 6.1 既にあるもの

初期パラメータ 4 本（本環境・クラスタ両方）、exp_040 のドメイン config（継承元）、
`merge_hlrb.py` / `merge_dithub.py`、検出器・フック・サンプラ一式、バッファ定義 6 ドメイン分、
exp_023 の ZiRa / DitHub 評価 config（前半3ドメイン＋ZCOCO）、exp_026 の評価 config（後半3ドメイン）。
**mmdet 本体の変更は不要。**

### 6.2 作るもの

| 成果物 | 内容 |
|---|---|
| `gen_configs.py` | 学習 config 12 本（`{zira,dithub}_{replay,replayfree}_{aerial,microscopic,documents}.py`）を生成。exp_040 のドメイン config を継承し、exp_039 と同じ手法ボディ（dithub_classes はラベルマップから）を載せる |
| `configs/zira_eval_{aerial,microscopic,documents}.py` | exp_026 eval を継承した ZiRa 評価 config |
| `configs/dithub_eval_*.py`（6 ドメイン＋zcoco） | 260 クラス和集合版の DitHub 評価 config |
| `run_sequential.sh` | 後半3ドメイン版の逐次ドライバ（INIT＝§2.1、LEARNED に前半3ドメインを予め積む、DitHub は t=4 から式4 を実行） |
| `sbatch_{zira,dithub}_{replayfree,replay}.sh` | クラスタ投入スクリプト4本（§5 の4ジョブ構成。exp_039 の sbatch を条件単位に分けて写す） |
| `check_exp041_setup.py` | 実行前検証（§6.3。本環境で実施） |
| `README.md` | クラスタでの投入から結果転送までの手順 |

### 6.3 実行前に確認する項目（本環境）

1. 4 構成 × 3 ドメインすべてで model が build でき、学習対象が手法どおり
   （ZiRa は RDB、DitHub は LoRA 109 層）。
2. 学習設定が §2.2 と一致（20 epoch / milestones[15] / lr 1e-4 / wd 1e-4 / batch / seed 0 / dn）。
3. 手法固有値が §2.3 と一致（λ=0.1・η=0.2・s=0.1 / r=16・alpha=8・warmup_epochs=10・
   trained_classes）。DitHub は `encoder num_cp=0`・`encoder_cp=6`・リプレイ有りの検出器クラスを含む。
4. model 辞書の機械的 diff: 学習 config が exp_039 の同条件 config と手法部分で一致し、
   データ部分が exp_040 の同ドメイン config と一致すること（exp_039 item7 / exp_040 item7 の方式）。
5. §2.1 の初期パラメータを load_from で読み、実データ 1 step で loss が計算できること
   （4 条件。DitHub は 152 クラスライブラリ読み込み時の想定内 missing/unexpected keys を確認）。
6. リプレイのバッチ構成が [現在 4, 参照 1, 過去プール 1] で、過去プールが累積構成であること。
7. DitHub 評価 config（260 クラス版）で t=3 のライブラリ ckpt を読み、既存の 152 クラス版と
   評価値が一致すること（和集合拡大が評価を変えないことの実測確認。1 ドメインで可）。
8. クラスタ側パス（`/local_cache/${SLURM_JOB_ID}/datasets`）で config が解決すること。

## 7. リスク・懸念

- **lr 1e-4 は両手法の論文値の 1/10**（exp_039 §7 と同じ。揃えることの代償として記録する）。
  前半3ドメインの exp_039 の実測で適応がどうだったかが参考になる。
- **DitHub の warmup が documents で約 11,170 iter** と、論文の 1,500 iter の 7 倍超になる
  （epoch 基準化の帰結。exp_039 §7 と同種）。
- **DitHub の activation checkpointing（encoder_cp=6）が勾配を変える**（exp_039 §7 の実測、
  ‖Δg‖/‖g‖ ≈ 1e-1。原因未特定）。exp_039 と条件を揃えるため encoder_cp=6 を維持する。
- documents は 59 クラスで、評価は exp_026 の eval config（チャンク処理を含む既存の型）で
  賄える。max_text_len の制約（256）には収まる。

## 8. 承認をお願いする範囲

§2 の条件で、ZiRa / DitHub × リプレイ有無の 4 条件を、exp_039 の t=3 融合済みパラメータから
aerial → microscopic → documents へ逐次学習・評価すること。**実行は MPRG クラスタ**（§5）、
4 ジョブ並列で実時間約 35 時間。本環境で行うのは config 作成と実行前検証（§6.3）まで。

## 9. 確定済みの判断

- ジョブは4本構成（条件ごとに1ジョブ・並列）。2026-08-15 指示。投入順・タイミングはユーザーが決める。
- 実行はクラスタの新規クローン `/home/kouyou/VLM_OD_CL_exp041` で行う。2026-08-15 指示。
