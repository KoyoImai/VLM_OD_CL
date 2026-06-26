# exp_008 結果まとめ：単純 LoRA の適応／COCO 忘却（full / frozen FT との比較）

作成日: 2026-06-26。本書は exp_008（単純 LoRA の組込み確立＋ sweep）の**数値集約版**。
生ログ: [`sweep_underwater.txt`](./sweep_underwater.txt)、設計: [`../design.md`](../design.md)、所見: [`../outputs/notes.md`](../outputs/notes.md)。

すべての数値は各 work_dir 配下の eval/COCO ログ（`coco/bbox_mAP`）から直接抽出・検証済み（2026-06-26）。
評価は **LoRA を base へマージ**（`W += (α/r)·BA`）した plain ckpt を既存 plain config でロードして測定。

---

## 1. 基準値（参照）

| 設定 | underwater 適応 mAP | COCO mAP（汎用保持） |
|---|---|---|
| zero-shot（事前学習のまま） | 0.051 | 0.504 |
| full FT（exp_003） | 0.340 | 0.438 |
| frozen 全FT（exp_004/005） | 0.316 | 0.414 |

- **適応の到達目標** ≒ frozen FT 0.316 / full FT 0.340。
- **COCO 保持の上限** = zero-shot 0.504（マージ方式では base 経路を残せないため、これを完全保持はできない）。

---

## 2. underwater LoRA sweep（単独・alpha=rank・実効64・20ep・milestone[15]・seed0）

対象層 = `encoder + decoder + bbox_head`（付与 143 層、学習パラメータ = LoRA のみ 2.40M / 175.3M = 1.37%）。
alpha=rank に統一（scaling=1）し、rank を「容量」だけを変える軸として切り分け。

**適応 mAP / COCO mAP**

| lr ＼ rank | r16 | r32 | r64 |
|---|---|---|---|
| **1e-4** | 0.218 / 0.417 | 0.241 / 0.412 | 0.256 / 0.393 |
| **1e-3** | 0.315 / 0.315 | 0.335 / 0.323 | 0.336 / 0.250 |
| **5e-3** | 0.322 / 0.136 | 0.257 / 0.097 | 発散・除外（TRAIN_FAILED） |

- **採用値 = lr=1e-3、rank=r16〜r32**（適応が frozen FT 水準＋COCO 劣化が相対的に最小）。
- lr=5e-3 r64 は epoch 11 で `grad_norm=inf`、Hungarian matching が NaN/Inf を踏みクラッシュ → 方針通り除外。

---

## 3. 読み取れること

### (a) 適応の律速は rank ではなく lr
- lr=1e-4 行は rank を 16→64 に上げても適応は 0.218→0.256 で頭打ち、frozen FT(0.316) に届かない。
- lr を 1e-4→1e-3 に上げた瞬間に適応が 0.218→0.315 と frozen FT 水準まで回復。
- → 単純 LoRA(1e-4) の適応低迷の主因は **容量不足ではなく学習率不足**。

### (b) 適応とゼロショット保持は明確にトレードオフ
- 適応を取り戻す（lr↑）と COCO が崩れる（r16 で 0.417→0.315）。
- rank↑でも適応はわずかに伸びる（0.315→0.336）が COCO はさらに悪化（r64 で 0.250）。
- → **素の LoRA はマージしても「奪い合い」を解かない**。full/frozen FT と同じトレードオフ曲線をなぞるだけ。

### (c) exp_007 の機序分析と整合
- exp_007：忘却＝共有重みの COCO/過去部分空間成分。だが適応も同一部分空間に依存（奪い合い）。
- 本 sweep の (b) はこの主張をハイパーパラメータ平面上で再現した形。

---

## 4. 手法フェーズ（exp_009 以降）への引き継ぎ

- **rank/lr の既定**: lr=1e-3、rank は中程度（r16〜r32）。
- **新規性の所在**: 「LoRA を使うこと」ではなく、**LoRA 上の干渉制御（soft 保護／部分的直交化）でトレードオフ曲線を外側へ押し出せるか（Pareto 改善）**。
- 本 sweep が引いた「素の LoRA のトレードオフ曲線」が、提案手法の**比較下地（このカーブを上回れば貢献）**となる。

---

## 5. 未取得（exp_008 のスコープ外・次の予定）

- 第1層：electromagnetic / documents の単純 LoRA（lr=1e-3, r16）→ 3 ドメインで単独適応/COCO を揃える。
- 第2層：3 ドメイン逐次 素朴FT（下限）＋ joint（上限）。※design.md 承認ゲートを通す。

## 関連
- 設計: [[../design.md]] / 進め方の議論: [[../../exp_008.5/research_plan_minutes]]
- 機序: [[../../exp_007/design.md]] / 統合計画: [[../../exp_007.5/research_plan]]
