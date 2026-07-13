# DitHub 実装ノート（論文 + 公式コード読解、2026-07-12）

対象: Cappellino et al. "DitHub: A Modular Framework for Incremental
Open-Vocabulary Object Detection" (NeurIPS 2025, arXiv:2503.09271)。
論文 PDF（全22頁）と公式実装 https://github.com/chiara-cap/DitHub を両方読み、
忠実な再実装に必要なレベルまで手法を確定させた記録。概要は
[[OVD-CL/DitHub_summary]]、ZiRa との対比は [[ZiRa_implementation_notes]]。

## 1. 手法の全体像

IVLOD（ZiRa と同一のタスク設定）を、単一の適応重みではなく**クラス別モジュールの
ライブラリ**で解く。LoRA の A 行列をクラス別（クラス固有知識）、B 行列を全クラス・
全タスク共有（汎用知識）に分離し、Version Control System のように branch / fetch /
merge でライブラリを管理する。追加の損失項はなく、Grounding DINO の元の損失のみで
学習する。主結果（ODinW-13 full-shot, 3 seed 平均）: ZCOCO 47.01 / Avg 62.19
（ZiRa 46.26 / 57.98、ゼロショット GD 47.41 / 46.80）。

## 2. LoRA の挿入位置（公式コードの規則）

`get_lora_modules`: モデル内の **全 nn.Linear のうち out_features ≥ 128** のものに
挿入。ただし名前が `transformer.dec* / bert / backbone / feat_map` で始まる層は除外。
実質は encoder（Feature Enhancer）の全 Linear で、具体的には
(a) 画像側 deformable attention の sampling_offsets / attention_weights /
value_proj / output_proj、(b) 画像側 FFN、(c) テキスト側 self-attention の
q/k/v/o（nn.MultiheadAttention の融合 in_proj を分解する専用モジュールに
差し替えて適用）、(d) テキスト側 FFN、(e) 融合層 BiAttention の 6 つの射影、
に加えて (f) `transformer.enc_output`（mmdet で言う **memory_trans_fc**）も
除外規則に掛からず含まれる（論文は「encoder のみ」と書くが、コードでは
enc_output も適応される）。bbox_embed は decoder 経由の名前で先に列挙されるため
memo 機構で除外される。neck（input_proj、Conv2d）と text_feat_map は対象外。

LoRA の設定: r=16、lora_alpha=8（scaling = alpha/r = 0.5）、dropout 0。
A は kaiming 初期化、共有 B は 0 初期化（初期状態で ΔW = B·A = 0、つまり θ0 等価）。

## 3. 層ごとの構造（LinearPool）

各挿入層は次を持つ: 凍結された元の weight/bias、共有 `shared_lora_b`（out×r）、
warmup 用 `warmup_lora_a`（r×in）、クラス別 `per_class_lora_A`（dict、r×in ×
クラス数）。状態（現在のタスク・クラス・フェーズ・過去タスクのライブラリ）は
singleton の TaskMemory が保持し、全 LinearPool が参照する。

forward（学習時）: バッチの**画像ごと**に「その画像の GT に含まれるクラスから
一様ランダムに 1 つ」選び（stochastic training strategy）、そのクラスの A を
その画像の分だけ適用する（バッチ内でサンプルごとに異なる A をスタックして bmm）。
warmup フェーズ中は全サンプルが `warmup_lora_a` を使う。B は常に共有 1 本。

forward（評価時）: プロンプト中のクラスのうちライブラリに A が存在するものを
集め、`mean_c(B·A_c)` を 1 回だけ加算する（合成モジュールで single forward）。
**モジュールを持たないクラスには何も適用されない**。プロンプト中に該当クラスが
1 つも無ければ出力は素の θ0 と一致する。ZCOCO はこの機構により構造的に保護される
（COCO 80 クラスのうちライブラリと名前が一致するものだけが適応の影響を受ける）。

## 4. タスク内の学習手順

タスク t（クラス集合 C_t）の学習は max_iter の**前半 = warmup、後半 =
specialization** に等分される（`iter == max_iter/2` で切替）。

1. **Warmup**: 全画像で warmup_lora_a と shared B を学習（クラス非依存の
   タスク知識を獲得し、specialization の共通初期化を作る）。
2. **切替（enable_per_class）**: 各クラス c ∈ C_t について、過去に学習済み
   （ライブラリに実学習された A_c が存在）なら
   `A_c ← λ_A·A_wu + (1−λ_A)·A_c_old`（式3、λ_A=0.3、fetch+merge）。
   未学習なら `A_c ← copy(A_wu)`（branch）。
3. **Specialization**: 画像ごとにランダム選択した 1 クラスの A_c を学習。
   B も学習を続ける。
4. **タスク終了（end_task）**: 全 A_c をライブラリへ保存し、
   `B ← (1−λ_B)·B_prev + λ_B·B_opt`（式4、λ_B=0.7）。ただし B_prev は
   タスク 2 以降にのみ存在するため、**最初のタスクでは式4は実行されない**。

## 5. 学習設定（公式コード準拠）

| 項目 | 値 |
|---|---|
| ベースモデル | Grounding DINO Swin-T（O365+GoldG+Cap4M、ZCOCO 47.41） |
| optimizer | AdamW, lr 1e-3, weight_decay 1e-2（LoRA パラメータのみ学習） |
| スケジュール | タスクあたり 3000 iter × batch 2、iter 1200 で lr ×0.1 |
| フェーズ配分 | warmup 1500 iter + specialization 1500 iter（等分） |
| grad clip | max_norm 0.1, norm_type 2 |
| タスク | ODinW-13 をランダム順で逐次学習（3 seed 平均） |
| 追加損失 | なし（GD の contrastive cls + localization のみ） |
| ハード | RTX A5000 1 枚、1 run 約 8 時間、LoRA 総量 19.6M params / 75.65 MB |

## 6. アブレーションの要点

warmup 追加で +2.3 Avg、式4（B merge）で +0.5、式3（A merge）が最大の寄与で
計 +6.13（Base 比）。クラス特化を外した EnE（ランダムな 1 モジュールを学習）比で
+1.23 Avg。rank は r=2 でも Avg 60.25（ZiRa 同メモリで +2.28）、r=1 でも 57.04。
λ_A は低いほど良く（0.3 が ODinW-13 最適）、λ_B は高いほど良い（0.7 峰）。
encoder のみの適応が decoder のみ / 両方より ZCOCO・Avg とも優る（Table A）。
B をクラス別にすると Avg +1.03 だがメモリ倍増・ZCOCO 低下（Table B）。

## 7. 論文とコードの食い違い・コードで初めて分かること

1. 論文は「encoder のみ適応」だが、コードは `transformer.enc_output`
   （mmdet の memory_trans_fc）にも LoRA を挿す。
2. 対象層の実定義は「out_features ≥ 128 の Linear」という規則であり、
   attention_weights（out=128）が含まれる一方、bbox 回帰の最終層（out=4）等は
   自然に外れる。
3. 式3の適用条件はクラス単位のカウンタ（specialization 中に実際に選択された
   回数）で判定され、「ライブラリに存在するが一度も学習されていない」クラスは
   merge でなく warmup のコピーを受ける。
4. per_class_lora_A は全タスクの全クラス分が最初から kaiming 初期化で確保される
   （B=0 なので出力には影響しない。実体は enable_per_class で上書き）。
5. 評価はタスク内では行われず、全タスク終了後に一括評価する（epoch ごとの
   val は存在しない）。
6. 論文の「2 epochs」に相当する記述はなく、タスクあたり 3000 iter 固定。
7. dn（denoising）はモデル build で無効化されている（ZiRa 公式と同じ系譜の
   コードベース）。

## 8. ZiRa との構造対比（本プロジェクトの観点）

挿入位置: ZiRa = neck + text_feat_map（抽出→融合の入口）、DitHub = encoder 全体
+ enc_output（融合そのもの）。本プロジェクトの exp_011 ロールバック分析で
「忘却・適応の主座は encoder（特に画像 self-attn）」と特定した場所に、DitHub は
直接適応を挿している。忘却対策: ZiRa = 出力ノルム罰則（ZiL）+ タスク境界の融合、
DitHub = クラス条件付き適用（未知クラスには ΔW を掛けない）+ A/B の分業と融合。
DitHub は追加損失なしで、ゼロショット保護を構造（クラス選択性）で実現する。
