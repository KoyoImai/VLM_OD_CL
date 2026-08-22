# exp_050 設計書: DGS の Roboflow100 6 ドメイン逐次（バッファ不使用）

作成 2026-08-22。**実行前のユーザー承認ゲート**（行動原理1・3）。
関連: [[../exp_049/design]]・[[../exp_049/results/summary]]（ODinW-13 実装検証・再現）、
[[../exp_049/implementation_plan]]（移植の正本）、
[[../exp_045/design]]・[[../exp_047/design]]（同一枠のバッファ不使用系列 = 直接の対照）、
[[../exp_039/design]]・[[../exp_041/design]]（ZiRa/DitHub の RF100 = 「手法固有部のみ維持し
共通枠に載せる」前例）。

## 0. 位置付け

ODinW-13 で実装検証・再現済みの DGS（exp_049、案A: トポロジー蒸留有効）を、
RF100 の 6 ドメイン逐次に載せる。バッファ不使用なので、対照の中心は同じ枠の
リプレイフリー（exp_024＋exp_040）と EWC / InfLoRA / 素の LoRA（exp_045 / 047）、
および ZiRa / DitHub（exp_039 / 041）である。

## 1. 目的

DGS を underwater → electromagnetic → videogames → aerial → microscopic → documents で
逐次学習し、各 t の学習済み全ドメイン mAP と ZCOCO を測る。
**仮説と判定基準はユーザーがノートで確定する。**

## 2. 条件

### 2.1 共通の学習枠（RF100 系列と同一。ZiRa/DitHub/EWC/InfLoRA/LoRA の前例に従う）

| 項目 | 値 |
|---|---|
| epoch / スケジュール | 20 epoch、MultiStepLR milestones=[15]、gamma=0.1 |
| optimizer | AdamW **lr 1e-4** / wd 1e-4、grad clip 0.1（stage1/2 とも。公式の 8e-4/5e-4 からの逸脱として記録。ZiRa（公式 1e-3 → 枠 1e-4）と同じ扱い） |
| バッチ | 現在ドメインのみ、batch 4/GPU × 4 GPU = 16（DefaultSampler） |
| dn / num_classes / seed | stage1 有効・**stage2 のみ無効（手法固有）** / 256 / 0 |
| データ | ODVGDataset ＋ RandomSamplingNegPos（既存系列と同一。config は exp_024 / exp_040 の replayfree を継承しモデルだけ差し替え） |
| AMP | 不使用（exp_049 と同一） |
| 逐次方式 | 各ドメインの epoch_20 を次の load_from に。in-training val なし |

### 2.2 DGS 固有（exp_049 と同一 = 公式値・案A）

r=16 / alpha=32、enhancer 6 層の両側 FFN、frozen_cfg 全凍結（lora_ のみ学習
＋stage1 は dn label_embedding）、DTG: expand_th=150・全量統計・min_eig 1e-3、
ルーティング ood_th=500（ZCOCO 評価は 200）、EMA マージ λ=0.2、
stage2: 擬似ラベル σ=0.4/IoU 0.7 ＋ トポロジー蒸留 'inter-class'（γ=3/5）＋
Group Init ＋ クエリ初期化。

### 2.3 段階的スコープ（stage2 の要否は DTG の実測で決まる）

RF100 の 6 ドメインは ODinW-13 より分布が離れており（exp_049 では 13 タスクが
12 グループ）、**全ドメインが別グループになる公算が大きい**。その場合 stage2 は
一度も走らず、実装は stage1 のみで完結する。手順:

1. **特徴抽出**（θ0・全量 約 80,300 枚・1 回）→ **DTG ドライラン**（統計と割当のみ。
   学習なし・数分）で グループ割当を確定し、ユーザーに報告する。
2. 併合が無ければ stage1 のみで本走。
3. **併合が出た場合は本走前に停止して報告する**（stage2 のデータ側機構は
   ODinW-13 の METAINFO / COCO 形式に依存しており、ODVG・RF100 用の
   クラス空間処理の追加実装が必要になる。その実装は別途承認を取る）。

### 2.4 評価

exp_049 と同じ方式で RF100 用の評価 config を生成する:
各 t で学習済みドメイン 1..t の per-domain 評価（データ定義は exp_023 / exp_026 の
評価 config と同一の valid・(800,1333)・batch 1・num_classes 256、videogames は
chunked_size=40。**DGS の画像単位ルーティング付き**）＋ ZCOCO（eval_base_coco の
データ定義・ood_th=200）。DGS の chunked 推論経路（fast_chunked_predict）は
実行前検証で確認する。

## 3. 判定材料

**最終的な判定はユーザーが行う。**

1. 各 t の学習済み全ドメイン mAP と ZCOCO。
2. 対照: リプレイフリー / EWC / InfLoRA / 素の LoRA / ZiRa / DitHub（note15 の表に追記可能な形）。
3. DTG のグループ割当と、ZCOCO 評価でのルーティング OOD 率
   （RF100 ドメインは COCO から遠いため、COCO が全て OOD → ZCOCO ≈ θ0 と
   なるかが読みどころ。exp_049 §4 の対照）。

## 4. 実装（design 承認後に着手）

| 成果物 | 内容 |
|---|---|
| `extract_feats_rf100.py` | 6 ドメインの DTG 用特徴抽出（θ0・全量） |
| `gen_configs.py` | 学習 config（stage1 × 6。stage2 は §2.3-3 の場合のみ）＋ドメイン別焼き込み |
| `gen_eval_configs.py` | 評価 config（eval_after_t{1..6} ＋ ZCOCO） |
| `run_dgs_rf100.sh` | 逐次ドライバ（DTG → stage 選択 → 学習 → 評価。exp_049 の写し） |
| `check_exp050_setup.py` | 実行前検証: config build・凍結/容量・ODVG 実データ 1 step・videogames の chunked predict・DTG ドライラン・ZCOCO ルーティング |

## 5. 実行環境とコスト

**MPRG クラスタ 1 ジョブ（a6000_ada・4 GPU）で実行**（2026-08-22 ユーザー決定）。
専用クローン方式（exp_045/048 と同じ）。DTG 用特徴はジョブ内で抽出する
（θ0・決定的なので環境間で同一。ゲート判定用のドライランは本環境で先行実施し、
グループ割当を報告してから投入する）。

| 項目 | 見積もり |
|---|---:|
| 特徴抽出（1 回） | 約 1〜1.5 h（GPU 1 枚） |
| 学習 6 ドメイン | 約 5,025 iters/epoch 合計 × 20 ≒ 100k iters、LoRA のみ学習で約 28〜40 h |
| 評価（21 本＋ZCOCO 6 本） | 約 4〜6 h |
| ckpt | **last（epoch_20）のみ保存・optimizer 状態なし**（2026-08-22 ユーザー決定。約 0.7 GB × 6） |

## 6. リスク・懸念

- **ODVG 経路での DGS 学習は未検証**（exp_049 は COCO 形式）。検出器の loss は
  上流 GroundingDINO と同一の tokens_positive 分岐を持つため動く見込みだが、
  実行前検証の実データ 1 step で必ず確認する（§4）。
- videogames の chunked 推論（fast_chunked_predict）はルーティングと併用した
  実績が無い（検証項目）。
- lr を枠の 1e-4 に落とすため、公式（8e-4）より適応が弱く出る可能性がある。
  これは ZiRa/DitHub と同じ「枠優先」の設計判断であり、但し書きとして記録する。
- 忘却はグループ構造上ほぼゼロになる見込み（他グループの学習は自グループに
  影響しない）。比較の主戦場は適応と ZCOCO（ルーティングの OOD 挙動）になる。

## 7. 承認をお願いする範囲

§4 の実装＋実行前検証＋特徴抽出＋DTG ドライラン（グループ割当の報告まで）。
**本走（6 ドメイン逐次）は割当の報告後に改めて確認を取る**（§2.3。併合が出た
場合は stage2 の追加実装の承認も併せて確認する）。
