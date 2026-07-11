# ZiRa 実装ノート（論文 + 公式コード読解、2026-07-10）

対象: Deng et al. "Zero-shot Generalizable Incremental Learning for Vision-Language
Object Detection" (NeurIPS 2024)。`papers/ZiRaGroundingDINO.pdf` と公式実装
https://github.com/JarintotionDin/ZiRaGroundingDINO を両方読んで、忠実な再実装に
必要なレベルまで手法を確定させた記録。論文とコードが食い違う箇所は末尾に明記する。

## 1. 手法の全体像

ZiRa は事前学習済み VLODM（Grounding DINO）本体を全凍結し、2 箇所にだけ並列の
側枝 RDB（Reparameterizable Dual Branch）を挿入して逐次学習する。忘却対策は
(1) RDB 出力のノルムを罰する Zero-interference Loss (ZiL)、(2) タスク終了ごとに
高学習率枝を低学習率枝へ融合して初期化し直す再パラメータ化 (Rep+)、の二つで、
リプレイも蒸留用モデルコピーも使わない。

## 2. RDB の挿入位置（2 箇所のみ）

言語側は text_feat_map（BERT 出力 768 → 256 の Linear）に並列。入力は BERT の
last_hidden_state で、出力は本体の text_feat_map 出力に加算される。

視覚側は入力射影 conv（MM-GDINO で言えば neck の ChannelMapper）に並列。バックボーン
各レベルの 1×1 conv と、追加レベルの 3×3 stride-2 conv のそれぞれに RDB が付く。
重要な実装詳細として、RDB 出力は GroupNorm の**前**で本体 conv の出力に加算される
（`GN(conv(x) + rdb(x))`）。

これ以外（Swin、BERT、encoder、decoder、bbox_head 等）は全て凍結される
（BERT の pooler も明示的に凍結）。

## 3. RDB の構造と初期化

各挿入点の RDB は、元の層と同形の層 2 本（フルランク。LoRA ではない）から成る。

- HLRB（高学習率枝）: 重み・バイアスとも定数 1e-8 で初期化
- LLRB（低学習率枝、コード上の名前は `freeze_conv` / `freeze_linear`）: 0 で初期化
- スケール s: 学習可能スカラー（RDB ごとに 1 個）、初期値 0.1

順伝播（学習時）は `x_rdb = s · HLRB(x) + LLRB(x)`。評価時は LLRB のみ通す
（HLRB はタスク末尾で LLRB へ吸収されるため、推論では常に不要）。

## 4. Zero-interference Loss (ZiL)

各 RDB について、
`ZiL = SmoothL1(s·HLRB(x), 0) + SmoothL1(x_rdb, 0)`（reduction='mean'）。
第 2 項（RDB 全体の出力ノルム）が事前学習知識の保護、第 1 項（HLRB 出力ノルム）が
「前タスクまでの知識を保持する LLRB+本体の出力からの乖離」を罰して下流タスクの
忘却を防ぐ。視覚側は 4 レベル分を加算、言語側は 1 個。

総損失は `L_total = L_cls + L_loc + λ · (ZiL_vision + ZiL_language)`、λ = 0.1。

## 5. 再パラメータ化（Rep+、タスク終了ごと）

各タスクの学習完了直後に全 RDB で
`W_llrb ← W_llrb + s · W_hlrb`、`b_llrb ← b_llrb + s · b_hlrb`
を実行し、HLRB を 1e-8、s を 0.1 に再初期化する。これにより枝の本数はタスク数に
よらず一定で、次タスクでは「LLRB = 過去タスクの蓄積」「HLRB = 新タスクの獲得」
という分業が再現される。全タスク終了後は、線形性を使って本体の凍結層と RDB を
1 本に統合でき、推論コストは元モデルと同一になる（研究上重要な性質:
**RDB は元モジュールへ再パラメータ化可能** = 上限論法の適用対象）。

## 6. 学習設定（公式コード準拠）

| 項目 | 値 |
|---|---|
| ベースモデル | Grounding DINO Swin-T（O365+GoldG+Cap4M、ZCOCO 47.37） |
| optimizer | AdamW, lr 1e-3, weight_decay 1e-4 |
| 学習率差別化 | LLRB（名前に `freeze` を含む param）のみ lr×η、η=0.2 |
| スケジュール | タスクあたり 2000 iter × batch 2、iter 800 で lr ×0.1 |
| grad clip | max_norm 0.1, norm_type 2 |
| タスク | ODinW-13 をランダム順で逐次学習（seed 42、3 seed 平均） |
| dn (denoising) | ZiRa の build では dn_number=0（無効化） |
| 評価 | 全タスク学習後に各タスク AP の平均（Avg）と ZCOCO |

主結果（full-shot）: ZCOCO 46.06 / Avg 59.73（zero-shot 上限 47.37、素の逐次 FT
に相当する構成は ZCOCO 39.72）。

## 7. アブレーションの要点

Rep+（融合＋再初期化）を外すと下流と事前学習の両方で忘却が悪化する。ZiL を外すと
ZCOCO が 46.09 → 39.72 級に崩れる。L_hlrb 単独では最良 hAP に届かず、L_rdb と併用
して初めて両立する。片側（V のみ / L のみ）より両側挿入が良い。単枝 (SB) は ZCOCO
最高だが Avg が大幅に低く、二枝+Rep+ (RDB) が hAP 最良。λ は 0.05〜0.1、η は
0.1〜0.2 が最適域。ノルムは L1 / SmoothL1 がほぼ同等（L2 は ZCOCO 寄り）。

## 8. 論文とコードの食い違い（再実装時の裁定が必要な点）

1. ノルム種: 論文本文は L1 だが、公開コードの既定は SmoothL1（reduction='mean'）。
2. s の初期値: 論文 Table 8 の既定行は言語 1.0 / 視覚 0.1 だが、コードは両方 0.1。
3. スケジュール: 論文は「2 epoch、1 epoch 後に decay」だが、コードはタスクあたり
   2000 iter 固定・iter 800 で decay（データセットサイズ非依存）。
4. HLRB の再初期化: 論文は「ゼロにリセット」、コードは 1e-8。
5. 論文は denoising に言及しないが、コードは ZiRa 構成で dn を無効化している。

## 9. MM-GDINO（本リポジトリ）への対応関係

| ZiRa（公式コード） | MM-GDINO（mmdet） |
|---|---|
| `feat_map` | `text_feat_map`（Linear 768→256） |
| `input_proj[l][0]`（conv） | `neck.convs[l].conv`（1×1 ×3） |
| `input_proj[3][0]`（3×3 s2） | `neck.extra_convs[0].conv` |
| GN 前に RDB 加算 | `neck.convs[l].gn` の前に加算が必要 |
| freeze_all + unfreeze adapter | RDB のみ学習可、他は requires_grad=False |

注意: MM-GDINO の学習 config は dn を使う・BERT pooler の扱い・ContrastiveEmbed
の bias 等が原論文 Grounding DINO と異なるため、再実装では「ZiRa の挿入と損失」
だけを移植し、本体は MM-GDINO のまま凍結するのが最小差分。
