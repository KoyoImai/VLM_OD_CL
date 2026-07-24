# exp_023 実装計画書: full finetuning ＋ リプレイ（＝素朴リプレイ）

- 作成日: 2026-07-22
- 状態: **実装前の計画（レビュー用）**。承認・レビュー後に実装へ着手する。実験実行は design.md 承認が前提。
- 対象: **full finetuning ＋ リプレイ（＝素朴リプレイ）のみ**。他手法（ZiRa/DitHub/条件A）は別途。
- 基づく設計: [[design]]（承認待ち）／[[../replay_experiment_plan]]。

---

## 0. この手法の定義

- 全モジュール学習（unfrozen：backbone/language_model の凍結を外す）＋ 素朴なバッファ混合。
- 逐次順: underwater → electromagnetic → videogames。

## 1. 全体構成（逐次学習ドライバ）

単一の学習 run ではなく、ドメインごとに config を切り替える**逐次ドライバ**で回す（バッファが各ドメイン後に変わるため）。

各ドメイン t について:
1. config を生成（現在ドメイン t の train ＋ 現時点のバッファ）。
2. 前ドメインの **last ckpt** から `load_from` して学習（t=1 は θ0 から）。
3. 評価: 現ドメイン t の valid、全過去ドメイン valid、ZCOCO。
4. ドメイン t の train から **500 をランダムサンプリング**してバッファに追加。

## 2. データ準備

- **各ドメインの OD-ODVG 変換**: underwater/electromagnetic/videogames の COCO 形式アノテーションを `coco2odvg.py` で OD-ODVG に変換。**ドメイン別 label_map** を生成（id_map の COCO 専用ハードコードは各ドメイン用に用意する必要あり）。
- **参照データ Objects365v1**（データセット保存先は `/workspace/kouyou/datasets/`）:
  - **全量をダウンロード**して `/workspace/kouyou/datasets/objects365v1/` に配置。**学習には一部（バッファ用 1,000 サンプル）のみ使用**する。
  - `coco2odvg.py -d o365v1` で OD-ODVG（`o365v1_train_od.json` ＋ `o365v1_label_map.json`）に変換。
  - そこから **1,000 サンプルの参照バッファ subset** を独立スクリプトで抽出（事前選択・固定ファイル化）。
  - 抽出した 1,000 に対し **COCO2017-val とのハッシュ照合**を実施し、重複が無いことを確認（ZCOCO 妥当性）。
- **バッファ構成**: 参照 1,000（固定・全期間利用）＋ 各ドメイン 500。t=1（underwater 学習時）は過去ドメインが無いのでバッファ＝参照 1,000 のみ。
- **バッファは全比較手法で共通**（同一の固定サンプル）。選択はモデル非依存（train セットからのランダム抽出）なので、**学習前に一括で事前選択**し固定ファイル化する（下記6節）。これにより手法間の差が「バッファ内容の違い」と交絡しない。

## 3. データローダ・混合

- `ConcatDataset([現在ドメイン ODVG, バッファ ODVG])` を構成。
- `MultiSourceSampler(batch_size, source_ratio=[1, 0.5])` でバッチ内の 現在:バッファ ＝ 1:0.5 を実現。
  - 例: 現在4 ＋ バッファ2 にするなら batch_size=6, source_ratio=[1,0.5]（`num_per_source[0]=batch_size−Σrest`）。端数割り当ての挙動は実装検証で確認。
- 全ソース（現在・過去・参照）は **OD-ODVG ＋ `RandomSamplingNegPos`**（同一ドメイン負例、num_sample_negative=85 / full_sampling_prob=0.5 / max_tokens=256）。ドメイン別 label_map。

## 4. モデル・学習設定（full finetuning）

- **全モジュール学習**: finetune config の `paramwise_cfg` から backbone/language_model の `lr_mult=0` を外す（unfrozen 化）。
- epoch=20、milestones=[15]、gamma=0.1、lr=1e-4、seed=0。
- 各ドメインで LR スケジュール・optimizer 状態をリセット、重みのみ前ドメイン last を引き継ぐ（`load_from`）。
- 適応 mAP は **last** を採用。

## 5. 評価

- 各ドメイン学習後: 現ドメイン valid（適応 mAP）、全過去ドメイン valid（各ドメインの label_map で個別に）、ZCOCO（`eval_base_coco.py` 経由）。
- 忘却はドメイン別（平均なし）。

## 6. バッファの事前選択（スタンドアロン・全手法共通）

- バッファ選択は**学習とは別プロセスの独立スクリプト**で行い、**学習前に一括で**確定する。
  - 参照 Objects365v1 から 1,000、各ドメイン（underwater/electromagnetic/videogames）の train から 500 ずつを、**独自 seed で決定的にランダム選択**し、固定ファイル（odvg）として保存。
- 各学習 run はこの**固定バッファファイルを読み込むだけ**。学習 run は開始時に自分の seed=0 で RNG を初期化するため、**バッファ選択スクリプトの乱数消費は学習の RNG に影響しない**（学習データの順番はバッファ事前選択に左右されない）。
- 全比較手法が**同一の固定バッファ**＋同一 seed=0 を使うため、手法間でバッファ内容・学習データ順が揃い、交絡しない。
- 逐次では、ドメイン t の学習時に「参照 ＋ t より前のドメインのバッファ」を固定ファイルから読み込んで使う（選択自体は事前に済んでいる）。

## 7. 実装検証（実行前・行動原理の役割1）

- (a) ドメイン別 label_map で **負例が同一ドメインに閉じる**ことを確認。
- (b) 混合バッチが loss() の `tokens_positive` 経路で正しく処理されること、`MultiSourceSampler` の比率が意図通り（現在:バッファ＝1:0.5）であること。
- (c) **単一ドメイン学習（underwater、リプレイ無し）の数値が現行（全クラス連結 CocoDataset）とどれだけ変わるか**（負例サンプリングへの切替の影響量。design.md 8節の「過去実験と直接比較不可」の根拠づけ）。
- (d) 全モジュールに勾配が流れること（unfrozen 化の確認）。

## 8. 主な実装課題・caveat

- **Objects365v1 は全量ダウンロード**して `/workspace/kouyou/datasets/objects365v1/` に配置（数十〜100GB 超、空き 238G。学習には 1,000 サンプルのみ使用）。ダウンロード完了が前提タスク。
- `coco2odvg.py` の id_map は COCO 80 クラス専用ハードコード → 各ドメインの label_map を正しく作る必要（誤るとラベルとクラス名がずれる）。
- `MultiSourceSampler` の `num_per_source` は batch_size と source_ratio から整数割り当て。1:0.5 を意図通りにする batch_size 選定と端数挙動を検証。
- 逐次でバッファが変わるため、単一 config でなく**ドライバ＋動的 config 生成**が必要。
- テキスト方式を負例サンプリングに変えるため、**単一ドメインでも現行と数値が変わり得る**（検証(c)で定量）。

## 9. 成果物

- 逐次ドライバスクリプト、ドメイン別 config、変換済み odvg・label_map、参照 subset、バッファファイル、`experiments/exp_023/full_ft_replay_{ドメイン}_work_dir`、ZCOCO 出力。

## 10. 未確定・依存

- design.md の承認。
- 参照データ取得可否・コスト。
- 混合比率・バッファサイズの探索範囲（初期値のみ確定）。
