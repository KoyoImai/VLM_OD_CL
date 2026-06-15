# ZiRa（Zero-interference Reparameterizable adaptation）要約

出典: ZiRa, Grounding DINO の継続学習（Incremental Vision-Language Object Detection, IVLOD）。
本プロジェクトと最も近い既存研究。

## タスク設定
- 事前学習済み VLODM（Grounding DINO / OV-DINO, Swin-T）を**複数ドメインに逐次適応**しつつ
  ゼロショット汎化を保持する継続学習。
- ベンチマーク: **ODinW-13**（13下流タスクを1つずつ逐次学習、順序はランダムシャッフル×3seed）。
- 評価指標:
  - **ZCOCO**: COCO 上のゼロショット mAP（汎用知識の保持度）
  - **Avg**: ODinW-13 全タスク平均 mAP（適応性能）
  - **hAP**: ZCOCO と Avg の調和平均（バランス）
- few-shot 設定（0/1/5/10/Full-shot）も評価。

## 入力スケール・データ拡張
- 論文に明記なし。Grounding DINO の標準実装に従うと推定（flip / 多スケール resize 程度）。

## 学習スケジュール（下流タスクごと）
| 項目 | Grounding DINO | OV-DINO |
|---|---|---|
| epochs/タスク | 2 | 2 |
| batch size | 2 | 2 |
| GPU | 2×RTX3090 | 同 |
| optimizer | AdamW | AdamW |
| 初期 lr | 1e-3 | 1e-4 |
| lr schedule | 1epoch後に ×0.1 | 同 |
| weight decay | 1e-4 | 1e-4 |

- **凍結**: 事前学習 VLODM 全体（backbone+言語+検出器）。
- **学習対象**: 追加した **Reparameterizable Dual Branch (RDB)** のみ。

## 手法の要点（忘却対策）
- **RDB**: 各適応箇所に2並列ブランチ。HLRB（高lr, 新タスク高速適応）と LLRB（低lr=η×, 既知知識保持）。
  タスク学習後に HLRB を LLRB へ統合し HLRB をゼロリセット → **メモリがタスク数に比例しない**。
- **Zero-interference Loss (ZiL)**: RDB 出力の L1 ノルムを抑制し、事前学習表現への干渉を最小化
  → ゼロショット性能（ZCOCO）を維持。
- 主要ハイパラ: λ(ZiL重み)=0.05〜0.10, η(LLRB lr比)=0.2, スケール初期値 s=0.1。

## 主要結果（Grounding DINO, Full-shot）
- ZiRa: ZCOCO=46.06 / Avg=59.73。CL-DETR比でゼロショット +13.91 AP、iDETR比 +8.74 AP。

## 本プロジェクトへの示唆
- 継続学習の評価設計（ZCOCO/Avg/hAP、逐次学習＋順序シャッフル）をそのまま参考にできる。
- 「全凍結＋小アダプタ＋ノルム正則化＋極小エポック(2)」が忘却抑制の鍵。
  我々の現行 config（20epochs 全FT）は単体性能向きで、忘却は大きく出る想定 → ベースライン比較に好適。
- 関連: [[MM-GroundingDINO_summary]], [[Roboflow100_summary]]。
