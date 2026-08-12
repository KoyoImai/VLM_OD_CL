# exp_039 設計書: ZiRa と DitHub を Roboflow100 で、リプレイ・蒸留と同じ設定で学習・評価

作成 2026-08-12。**実行前のユーザー承認ゲート**（行動原理1・3）。
関連: [[../exp_023/design]]（RF100 逐次の初回。ZiRa/DitHub は論文準拠設定）、
[[../exp_027/design]]（リプレイ）、[[../exp_028/design]]（蒸留）、
[[../exp_034/design]] / [[../exp_037/design]]（ODinW-13 での再現）。

## 0. 位置付け

RF100 の 3 ドメイン逐次では、リプレイ（exp_027）と蒸留（exp_028）が
**20 epoch / lr 1e-4 / batch 6** という共通の土台で測られている。一方 ZiRa と DitHub は
exp_023 で **論文準拠の設定**（ZiRa: 2000 iter / lr 1e-3、DitHub: 3000 iter / lr 1e-3 / wd 1e-2）
で走っており、リプレイ・蒸留と横並びに置けない。

本実験は ZiRa と DitHub を**リプレイ・蒸留と同じハイパーパラメータ**で走らせ、
4 手法を同一の土俵に乗せる。手法固有の設定（ZiRa の ZiL・Rep+、DitHub のクラス別 LoRA・
warmup/specialization・式3/式4）はそれぞれの手法どおりに保つ。

## 1. 目的

RF100 の 3 ドメイン逐次において、ZiRa と DitHub を、リプレイ・蒸留と共通の学習設定で
測り直す。リプレイの有無の両方を取る。

**仮説と判定基準は実験ノートでユーザーが確定する。** 本書は測るものと条件を定める。

## 2. 条件

### 2.1 揃える設定（リプレイ・蒸留と同一）

`experiments/exp_023/configs/fullft_replay_base.py` の値。exp_027 / exp_028 はこれを継承している。

| 項目 | 値 |
|---|---|
| 継承元 | `grounding_dino_swin-t_pretrain_obj365.py`（事前学習 config） |
| データセット | `ODVGDataset`（`*_train_od.json` ＋ `*_label_map.json`） |
| プロンプト生成 | `RandomSamplingNegPos` |
| num_classes | 256 |
| epoch | 20、`MultiStepLR` milestones=[15]、gamma=0.1 |
| optimizer | AdamW **lr 1e-4 / weight_decay 1e-4** |
| grad clip | max_norm 0.1（L2） |
| batch size | リプレイ有り **6**、リプレイ無し **4** |
| dn | **有効**（事前学習 config のまま） |
| seed | 0、`deterministic=False` |
| in-training val | 行わない（評価は逐次ドライバ） |
| 逐次順 | underwater → electromagnetic → videogames |

**exp_023 からの変更点**は次の 3 つ。いずれもご指示の「ハイパーパラメータはリプレイ・蒸留と同じ」に従ったもの。

1. **エポック基準に変更**: ZiRa 2000 iter / DitHub 3000 iter → 両者とも 20 epoch。
2. **lr**: 1e-3 → **1e-4**。
3. **weight_decay**: DitHub のみ 1e-2 → **1e-4**（ZiRa は元から 1e-4）。

`paramwise_cfg` の `backbone` / `language_model` の `lr_mult=0.1` は継承するが、
ZiRa も DitHub も Swin / BERT を凍結するため実効的には無影響。

### 2.2 手法固有の設定（各手法どおり）

**ZiRa**（`papers/ZiRa_implementation_notes.md`）

| 項目 | 値 |
|---|---|
| Zero-interference Loss | `L_total = L_cls + L_loc + λ·(ZiL_vision + ZiL_language)`、**λ = 0.1** |
| LLRB の学習率差別化 | 名前に `freeze` を含む param のみ **lr × η、η = 0.2**（`paramwise_cfg` の `llrb`） |
| スケール s | RDB ごとに 1 個の学習可能スカラー、**初期値 0.1** |
| Rep+（タスク終了ごと） | `W_llrb ← W_llrb + s·W_hlrb`、HLRB を 1e-8・s を 0.1 に再初期化 |

**DitHub**（`papers/DitHub_implementation_notes.md`）

| 項目 | 値 |
|---|---|
| LoRA | r=16 / alpha=8（scaling 0.5）/ dropout 0、encoder ＋ `memory_trans_fc` の 109 層 |
| A | クラス別（`per_class_lora_A`）、B は全クラス共有 |
| warmup → specialization | 学習量の**半分**で切替 ＝ **epoch 10**（`DitHubSeqPhaseHook(warmup_epochs=10)`） |
| 式3（切替時） | `A_c ← λ_A·A_wu + (1−λ_A)·A_c_old`、**λ_A = 0.3** |
| 式4（タスク終了時） | `B ← (1−λ_B)·B_prev + λ_B·B_opt`、**λ_B = 0.7** |
| ライブラリ更新 | `experiments/exp_021/merge_dithub.py`（式4 ＋ 過去クラス A の和集合） |

exp_023 / exp_037 では iter 基準で `warmup_iters=1500`（3000 の半分）だった。本実験は epoch
基準になるため `warmup_epochs=10` を使う（`DitHubSeqPhaseHook` は両方の指定に対応する）。

フェーズの中身は次のとおり。**前半（warmup）**は全画像が `warmup_lora_a`（クラス非依存の 1 本）
を使い、クラス別 A は学習されない。**後半（specialization）**は画像ごとに GT クラスから一様
ランダムに 1 つ選び、そのクラスの A を適用する。共有 B は両フェーズを通じて学習される。
切替時に、過去タスクで学習済みのクラスには式3 が発火し、未学習のクラスには `A_c ← A_wu` の
コピーが入る。クラス数 1 のタスクは warmup 自体を行わないが、RF100 の 3 ドメインはいずれも
複数クラスなので**全タスクで warmup が走る**。

切替点をドメイン別に示す。iters/epoch は 4 GPU × batch 6（うち現在ドメイン 4）＝ 1 step あたり
現在ドメイン 16 枚から算出。exp_028 のログ（underwater `[20][750/790]`、electromagnetic
`[20][1550/1588]`、videogames `[20][500/515]`）と一致する。リプレイ無し（batch 4 × 4 GPU）でも
epoch 長は現在ドメインで決まるため同じ。

| ドメイン | クラス数 | iters/epoch | 20 epoch | 切替点（epoch 10） | 式3 の対象 |
|---|---:|---:|---:|---:|---|
| underwater | 28 | 790 | 15,800 | 7,900 iter | なし |
| electromagnetic | 39 | 1,588 | 31,760 | 15,880 iter | なし |
| videogames | 87 | 515 | 10,300 | 5,150 iter | `['person', 'car']` |

exp_037（ODinW-13）の warmup が 1,500 iter だったのに対し、本実験は underwater で 7,900 iter と
5 倍以上長い。DitHub の ablation は warmup の追加を +2.3 Avg の寄与としているが、この長さでの
検証は論文の範囲外である。

### 2.3 リプレイの有無

4 本を走らせる。

| # | 手法 | リプレイ | batch | 既存 config |
|---:|---|---|---:|---|
| 1 | ZiRa | 無し | 4 | `exp_023/configs/zira_replayfree_*.py` |
| 2 | ZiRa | 有り | 6 | `exp_023/configs/zira_replay_*.py` |
| 3 | DitHub | 無し | 4 | `exp_023/configs/dithub_replayfree_*.py` |
| 4 | DitHub | 有り | 6 | `exp_023/configs/dithub_replay_*.py` |

リプレイの構成は exp_023 / exp_027 と同一。t=1 は［現在, 参照(Objects365v1 1,000)］で
`source_ratio=[2,1]`、t≥2 は［現在, 参照, 過去プール］で `source_ratio=[4,1,1]`
（現在:バッファ = 2:1）。

## 3. 判定材料

**最終的な判定はユーザーが行う。** 数値の閾値は設けない。

1. 各ドメインの適応 mAP（学習直後）と最終値、忘却量。
2. ZCOCO の推移（t=0〜3）。
3. リプレイ有無の差（同一手法内）。
4. 4 手法（リプレイ / 蒸留 / ZiRa / DitHub）× リプレイ有無での対照。
   リプレイ・蒸留の値は exp_027 / exp_028 の実測を用いる。
5. exp_023 の論文準拠設定との差（同一手法内で、設定変更が何をもたらしたか）。

## 4. 手順

既存の逐次ドライバの構成を踏襲する。各ドメイン t で「学習 → 手法固有の後処理 → 評価」。

- ZiRa: 学習 → Rep+（タスク終了時にモデル内で実行）→ 学習済みドメインと ZCOCO を評価。
- DitHub: 学習 → `merge_dithub.py` でライブラリ更新（式4）→ 同上。

評価 config は exp_023 の `zira_eval_*.py` / `dithub_eval_*.py` / `*_eval_zcoco.py` を使う。

## 5. 実行環境とコスト

**本実験は MPRG クラスタで実行する**（本環境では実行しない）。exp_027 / exp_028 と同じ運用。

### 5.1 クラスタ側の前提

| 項目 | 内容 |
|---|---|
| パーティション | `a6000_ada`（A6000 Ada ×4、VRAM 48GB/GPU）。`--gres=gpu:4` |
| 除外ノード | `node03`（`/local_cache` が用意されず mkdir で落ちる。exp_028 と同じ） |
| 実行形態 | Singularity コンテナ（`--nv` ＋ `--bind`）、`/home/kouyou/sif/docker-image-of-mmdetection4singularity.sif` |
| リポジトリ | `/home/kouyou/VLM_OD_CL` |
| データ | `/home/kouyou/datasets/rf100_domain` を `/local_cache/${SLURM_JOB_ID}/datasets` へ**ステージングして使う**（NFS ホーム直読みはしない） |
| リプレイ用 Objects365 | 同じく bind が必要（exp_027 / exp_028 と同じ） |
| ZCOCO | `/dataset01` の MSCOCO を bind |
| 制限 | 同時実行 4 ジョブ / 同時投入 8 ジョブ、バッチ最大 168 h |
| ログ | `/home/kouyou/logs/result_%x_%j.txt`、`error_%x_%j.txt` |

### 5.2 所要時間

exp_027（リプレイ有り full FT、batch 6、20 epoch）と exp_024（リプレイ無し、batch 4）の
本環境（A100 40GB）での実測から。

| | underwater | electromagnetic | videogames | 計 |
|---|---:|---:|---:|---:|
| リプレイ有り（実測、full FT） | 8.9 h | 16.6 h | 5.7 h | **31.2 h** |
| リプレイ無し（実測、full FT） | 4.7 h | 10.1 h | 約 3 h | **約 18 h** |

ZiRa / DitHub は学習パラメータが少ないぶん backward は軽いが、ZiRa は二分岐・DitHub は
LoRA 経路が増えるため forward は同等以上。**full FT と同程度**と見積もる。

| 本数 | 学習 | 評価 | 小計 |
|---|---:|---:|---:|
| ZiRa 無し / DitHub 無し（各 18 h） | 36 h | 約 5 h | 41 h |
| ZiRa 有り / DitHub 有り（各 31 h） | 62 h | 約 5 h | 67 h |
| **合計** | | | **約 108 GPU 時間** |

### 5.3 ジョブ分割

同時実行 4 ジョブの制限内に収める。バッチ最大 168 h に対し最長のジョブでも 36 h 程度なので、
`--time` は余裕を見て 72:00:00 とする。

| ジョブ | 内容 | 見積もり |
|---|---|---:|
| `exp039_zira` | ZiRa リプレイ無し → 有り（直列） | 約 54 h |
| `exp039_dithub` | DitHub リプレイ無し → 有り（直列） | 約 54 h |

2 ジョブを並列投入すれば実時間は**約 54 時間**。1 ジョブ内をさらに分けて 4 並列にすれば
約 36 時間になるが、同一ジョブ内で `/local_cache` のステージングを共有できる利点を取り、
2 ジョブ構成を既定とする。ディスクは 4 本 × 3 ドメインの ckpt で約 100 GB
（ホーム約 3TB のうち。ckpt 1 個約 1.95 GB）。

### 5.4 本環境で行うこと

config の作成と実行前検証（§6.3）までを本環境で済ませ、クラスタへは push して持ち込む。
学習・評価はクラスタでのみ行う。

## 6. 実装

### 6.1 既にあるもの

exp_023 の ZiRa / DitHub 用 config（replay / replayfree × 3 ドメイン、計 12 本）と
評価 config、`merge_dithub.py`、`ZiRaGroundingDINO`、`DitHubODVGGroundingDINO`、
`DitHubSeqPhaseHook`。データ・リプレイバッファも exp_023 のものをそのまま使う。

### 6.2 作るもの

| 成果物 | 内容 |
|---|---|
| `configs/zira_base.py` / `configs/dithub_base.py` | exp_023 の base を継承し、§2.1 の設定（20 epoch / lr 1e-4 / wd 1e-4 / epoch 基準スケジュール）に差し替える |
| `configs/{zira,dithub}_{replay,replayfree}_{domain}.py` | exp_023 のドメイン config を継承し base だけ差し替える（データは複製しない） |
| `run_sequential.sh` | 4 本を回す逐次ドライバ。手法とリプレイ有無を引数で切り替え |
| `sbatch_zira.sh` / `sbatch_dithub.sh` | クラスタ投入スクリプト（§5.1・§5.3）。exp_028 の `sbatch_a.sh` を雛形に、`/local_cache` へのステージング、Objects365 と `/dataset01` の bind、`--exclude=node03` を踏襲 |
| `check_exp039_setup.py` | 実行前検証（本環境で実施） |

### 6.3 実行前に確認する項目

1. 4 構成すべてで model が build でき、学習対象が手法どおり（ZiRa は RDB、DitHub は LoRA 109 層）
2. 学習設定が §2.1 と一致（20 epoch / milestones[15] / lr 1e-4 / wd 1e-4 / batch / seed 0 / dn 有効）
3. 手法固有値が §2.2 と一致（ZiRa: λ=0.1・η=0.2・s=0.1 / DitHub: r=16・alpha=8・λ_A=0.3・λ_B=0.7）
4. DitHub のフェーズ切替が epoch 10 で起きること（`warmup_epochs=10` がログに出る）
5. 実データ 1 step で loss が計算でき、想定外のパラメータが更新されないこと
6. リプレイのバッチ構成が [現在4, 参照1, 過去1]（t≥2）になっていること
7. クラスタ側のパス（`/local_cache/${SLURM_JOB_ID}/datasets` 配下）で config が解決すること。
   本環境のパスをハードコードしないこと（exp_027 / exp_028 と同じ方式）

## 7. リスク・懸念

- **lr 1e-4 は ZiRa / DitHub の論文値（1e-3）の 1/10** である。exp_008 の素の LoRA では
  lr=1e-4 で適応が frozen FT 水準に届かなかった。両手法とも適応が伸びない可能性がある。
  これは「揃えることの代償」であり、条件の帰結として記録する。
- **DitHub の weight_decay を 1e-2 → 1e-4 に変える**のは論文設定からの逸脱である。
  ご指示（ハイパーパラメータはリプレイ・蒸留と同じ）に従ったが、wd を手法固有として
  1e-2 のまま残す選択もあり得る。**§9 で確認いただきたい。**
- **dn は有効**にする（リプレイ・蒸留と同じ）。ODinW-13 の exp_034 / exp_037 では公式実装に
  合わせて dn を無効化したので、そちらとは条件が異なる。RF100 側の横並びを優先した。
- 20 epoch は exp_023 の 2000/3000 iter より学習量が大きい（underwater で 790 iter × 20 =
  15,800 iter）。DitHub の warmup が 10 epoch = 約 7,900 iter と長くなる。

## 8. 承認をお願いする範囲

§2 の条件で、ZiRa / DitHub × リプレイ有無の 4 本について、RF100 の 3 ドメインを
逐次学習・評価すること。**実行は MPRG クラスタ**（§5）。約 108 GPU 時間、
2 ジョブ並列で実時間約 54 時間。

本環境で行うのは config の作成と実行前検証までで、学習・評価は行わない。

## 9. 承認前に判断いただきたい点

1. **DitHub の weight_decay** を 1e-4（リプレイ・蒸留に合わせる）とするか、1e-2（論文値）で
   残すか。本書は 1e-4 で書いている。
2. **ジョブ分割**。§5.3 の 2 ジョブ構成（手法ごとに直列、実時間 約 54 h）とするか、
   4 ジョブに分けて並列にするか（実時間 約 36 h、ただし `/local_cache` へのステージングが
   ジョブごとに発生する）。
