# exp_021 設計書: DitHub 再現（単発ドメイン学習・全 6 ドメイン × 2 体制）

作成 2026-07-12。実行前のユーザー承認ゲート。
実装の詳細と裁定の記録は [[implementation_plan]]、手法の正本は
[[../../papers/DitHub_implementation_notes]]。

## 1. 目的

既存手法 DitHub（NeurIPS 2025、クラス別 LoRA ライブラリで IVLOD を解く
ZiRa の直接競合）を MM-Grounding DINO 上で忠実に再現し、本プロジェクトの
統一プロトコル（単発ドメイン学習 + ZCOCO 評価、seed=0）で測定する。
これにより ZiRa（exp_020）と DitHub が同じ表に並び、既存手法 2 種の適応能力と
忘却特性を実測で比較できる。DitHub の適応位置は encoder + memory_trans_fc で、
exp_011 のロールバック分析が忘却・適応の主座と特定した場所に一致する点も
比較の観点になる。

## 2. 条件

2 体制 × 6 ドメイン（underwater / aerial / videogames / microscopic /
documents / electromagnetic。real_world 除外）= 12 run。

- 標準版: 20 epoch（warmup 10 + specialization 10）、lr 1e-4 / wd 1e-4、
  milestones [15]、4 GPU、`configs/dithub_{domain}.py`、`run_train.sh`
- 公式準拠版: 3000 iter（warmup 1500 + spec 1500）、batch 2、lr 1e-3 /
  wd 1e-2、iter 1200 で decay、1 GPU、`configs/dithub_official_{domain}.py`、
  `run_official.sh`
- 共通: seed=0、dn 有効、LoRA r=16 / alpha=8、対象 109 層
  （encoder 全 Linear + memory_trans_fc）、学習対象は LoRA のみ
  （19.6M 級 / 総パラメータの約 9%）、追加損失なし

## 3. 測定と判定（事前宣言）

各 run で適応 mAP と ZCOCO mAP。標準版の報告用 best は裁定④により
specialization 期（epoch 11〜20）から選ぶ。warmup 期の val が specialization 期を
上回った場合は「クラス特化が適応に寄与していない」観察として別途記録する。
公式準拠版は最終 iterate（iter 3000）を評価する（公式と同じ）。

比較基準線: module_rollback_summary.md の frozen / unfrozen / neckTfm と
exp_020 の ZiRa 行。解釈の観点:
- 適応: ZiRa（挿入位置が狭い neck+tfm）を超えるか。encoder 適応は frozen
  （neck+encoder+下流を学習）にどこまで迫るか。
- 忘却: DitHub はクラス条件付き適用のため、COCO と名前が重複するクラスを
  持たないドメイン（underwater / microscopic、検証7で確認済み）では
  **ZCOCO が構造的に θ0=0.504 と一致するはず**。重複を持つドメイン
  （electromagnetic 4 件、videogames 3 件、aerial / documents 各 1 件）だけが
  ZCOCO に影響し得る。この予測が外れたら実装を疑う。

## 4. 実装検証（実行済み、10/10 OK + 式4 の実モデル往復）

check_dithub_setup.py: ①対象 109 層・学習可能は lora_* のみ・除外漏れなし、
②attention 分解はビット一致、③θ0 等価性もビット一致（B=0）、④勾配疎通
（warmup では warmup_A と B のみ、specialization では選択クラスの A のみ）、
⑤式3（コピー / 融合）単体、⑥評価合成の手計算一致（9.5e-7）、⑦COCO 名前
重複の事前調査、⑧specialization のサンプル別 bmm 経路の数値一致、
⑨推論時モジュール選択（ドメインプロンプトで 28/28 クラス選択・出力変化、
未知クラスプロンプトでは変化 0.0 = ゼロショット保護構造の成立）、
⑩フック配線（epoch 9 で発火せず epoch 10 で全層切替）。加えて merge_dithub.py（タスク境界処理）は、合成テンソルでの
単体テストに加えて実モデルの state_dict での往復を確認済み: 式4 が全 109 層に
適用され、第 1 タスク（前段 θ0）ではスキップされ、マージ後の state_dict を
新モデルが missing / unexpected ゼロで読み込み、B が式4 の値と一致する。
未検証なのは完全な逐次動作（次タスクのモデルが過去クラスの A を確保して
式3 の fetch 分岐を発火させる部分）で、これは逐次実験の設計時に実装・検証する
（implementation_plan.md 3.5）。

## 5. 既知の注意点・限界

- 単発学習のため fetch / merge（式3の学習済み分岐）と式4（B 融合）は発火しない。
  検証されるのは warmup→specialization の分業・共有 B・確率的クラス特化・
  クラス条件付き推論まで（implementation_plan.md 3.5）。
- 標準版の lr 1e-4 / wd 1e-4 は公式（1e-3 / 1e-2）より一桁小さく、LoRA の
  立ち上がりが遅い可能性がある。適応 mAP が立ち上がらない場合は中断して相談。
- DDP は find_unused_parameters=True が必要（specialization 中、選択されなかった
  クラスの A が不使用のため）。学習開始直後の数 iter でエラーが出ないことを確認する。
- 標準版は ep10-20 の checkpoint を保持し（裁定④の best 選択のため）、
  ZCOCO 評価後に best と epoch_20 以外を削除してディスクを節約する。

## 6. コスト

標準版 6 run × 半日前後（exp_020 と同等）+ 公式準拠版 6 run × 約 40 分
+ ZCOCO 各 6 分 ≈ 標準版 3〜4 日 + 公式版 半日。

## 7. 成果物

- `experiments/exp_021/{domain}_dithub[_official]_work_dir` / `..._zcoco`
- `results/dithub_reproduction.md`: 12 run の実測値と基準線・ZiRa との比較表
  （学習完了後に作成）
- module_rollback_summary.md の横断表への DitHub 行追記

## 関連
- 実装計画・裁定記録: [[implementation_plan]]
- ZiRa との対比: [[../exp_020/design]], [[../../papers/DitHub_implementation_notes]]
