# exp_002.5: 現状まとめ（中間整理 / 実験なし）

作成日: 2026-06-14。exp_000〜exp_002 の結果と決定事項、論文調査、次の計画を一箇所に集約する中間整理。
**本ディレクトリでは実験を行わない。**

---

## 1. プロジェクトの目的（再掲）
事前学習済み VLM（MM-Grounding DINO）を対象に、**ドメインが増加する物体検出の継続学習**を扱う。
2つの課題に対処する：
1. 事前学習に乏しいドメインでの検出性能低下
2. 新ドメイン学習による破滅的忘却（汎用知識・過去ドメイン知識の消失）

---

## 2. これまでの実験と結果

### exp_000: underwater zero-shot 評価（単一GPU）
- 事前学習モデルを fine-tune なしで underwater valid 評価。
- 結果: **mAP=0.051**。継続学習の出発点（lower bound）を初取得。
- 観察: AP は低いが AR=0.21（位置は拾えるがクラス対照が弱い）、小物体が特に弱い。

### exp_001: COCO2017 + RF100 各ドメイン zero-shot 評価（4GPU 分散）
- COCO（汎用＝ZCOCO 相当）＋ RF100 6ドメインの lower bound 表を作成。
- 健全性チェック: COCO=0.504（文献 ≈0.506）、underwater 再評価=0.051（exp_000 一致）。
- **解像度は `FixScaleResize (800,1333) keep_ratio=True`。以後この設定を基準とする（→ 決定事項参照）。**

| 対象 | mAP |
|---|---:|
| COCO2017 | 0.504 |
| underwater | 0.051 |
| aerial | 0.034 |
| electromagnetic | 0.024 |
| videogames | 0.016 |
| documents | 0.006 |
| microscopic | 0.002 |

- 含意: COCO と専門ドメインで **約10〜250倍**の性能差。課題1を定量的に裏付け。
  microscopic はほぼゼロ（事前学習概念と最も乖離）。

### exp_002: 評価リサイズ感度分析（28評価、4GPU 分散）
- リサイズ 5 設定（baseline=800 / S1=640 / S2=1024 / S3=1333 / S4=640正方形固定）× 7対象を比較。
- 主な発見:
  - **解像度を変えると精度が下がる傾向**。COCO は 800→640 で 0.504→0.499、800→正方形640 で 0.478、
    短辺1333 では 0.021 と崩壊。RF100 ドメインでも 640/1024 はおおむね baseline 同等以下。
  - アップスケール／ダウンスケールいずれも明確な利得はなく、**事前学習が想定する短辺800（exp_001設定）が無難**。
  - 正方形固定（keep_ratio=False）は非正方形画像（COCO）で歪み悪化。RF100 は元640×640なので無影響。
    （S3 高解像度では aerial/videogames が評価後に終了時クラッシュ。mAP は記録済みで有効。）

---

## 3. 確定した決定事項
1. **入力解像度は exp_001 の設定 `FixScaleResize (800,1333) keep_ratio=True` をベースに進める。**
   - リサイズ（解像度変更）は精度に悪影響を及ぼすことを exp_002 で確認したため、
     事前学習が想定する短辺800（exp_001設定）を学習・評価で維持する。
   - 640 への縮小・1024 以上への拡大・正方形固定はいずれも採用しない。
2. **real_world ドメインは使用しない**（方針どおり）。対象は 6 ドメイン + COCO。
3. **多クラスのテキストトークン対処**: videogames(265 tokens) のみ `max_text_len=512`。他は 256 内。
4. **評価は 4GPU 分散推論**（`tools/dist_test.sh ... 4`）を標準とする。単一GPUと同値を確認済み。

---

## 3.5 評価プロトコル（明文化・固定）
評価設定は **データセット種別ごとに名前付き共通 eval ベース config** として固定する（exp_002.5 で確定）。
今後の評価はこれらを継承し、設定の揺れを排除する。両者とも検証済み。

### 共通プロトコル（COCO / RF100 とも同一）
- リサイズ: `FixScaleResize scale=(800,1333) keep_ratio=True`（exp_001 設定。解像度は変更しない）
- batch_size: 1 / dataset: `CocoDataset` / 評価指標: `CocoMetric (bbox)` = COCO mAP
- テキスト: クラス名を連結（`return_classes=True`）
- 実行: 4GPU 分散推論（`tools/dist_test.sh ... 4`）

### COCO 用: `configs/mm_grounding_dino/eval_base_coco.py`
- 単体で実行可能（val2017 / 80クラスを内包）。
- 例: `bash tools/dist_test.sh configs/mm_grounding_dino/eval_base_coco.py <ckpt> 4 --work-dir <dir>`

### RF100 用: `configs/mm_grounding_dino/eval_base_rf100.py`
- 事前学習 config を継承し評価プロトコルを固定した**単一ベース**。
  ドメイン eval config が `_base_='eval_base_rf100.py'` で継承し、data_root / metainfo /
  ann_file / num_classes（videogames は max_text_len=512）を与える。
- 注: 既存の学習用ドメイン config も同一プロトコルを満たすため、単発評価はそれを直接渡してもよい。
  本ベースは「プロトコルを明文化・固定」し将来の派生で必ず継承させるためのもの。
- ※ mmengine の制約上、`_base_=[domain, eval_base_rf100]` の 2 ベース併記は重複キーで不可。
  必ず eval_base_rf100 を単一ベースにして data を上書きする。

## 4. 論文調査の要約（papers/ に格納）
- [[../../papers/MM-GroundingDINO_summary]]: ベースモデル。学習拡張は flip+多スケール+crop+負例サンプリング。
  FT は短め（12ep 程度）・head 中心。重い拡張（mosaic/mixup）は非採用が流儀。
- [[../../papers/ZiRaGroundingDINO_summary]]: 最も近い既存研究（Grounding DINO の継続学習）。
  評価系 **ZCOCO / Avg / hAP**、逐次学習＋順序シャッフル。本体凍結＋小アダプタ(RDB)＋ノルム正則化で忘却抑制。
  2 epochs/タスク, batch 2, lr 1e-3→×0.1, AdamW, wd 1e-4。
- [[../../papers/Roboflow100_summary]]: データ基盤。RF100 は 640×640、RF100-VL の GroundingDINO は (640,1333)・
  追加拡張なし。ゼロショットは ODinW-13 49.2 → RF100-VL 15.7 と激減（課題1の裏付け）。

---

## 5. 作成済みの資産
- 派生 config 28 本: `experiments/exp_002/configs/<target>_<setting>.py`（リサイズ別、再現性用）。
  - `_base_` 相対継承で元 config は無変更。今後は exp_001 設定(800,1333)を使うため評価には用いない。
- 調査メモ: `experiments/exp_002/outputs/resize_aug_investigation.md`（学習拡張のリサイズ挙動）。
- 結果: `experiments/exp_001/results/zeroshot_summary.md`, `experiments/exp_002/results/resize_comparison.md`。

---

## 6. 現状の到達点
- **継続学習の「出発点（lower bound）」と「評価プロトコル（解像度=exp_001設定 (800,1333)）」が確定**した段階。
- 未着手: fine-tune による upper bound、忘却の定量化、継続学習手法（ZiRa 等）の導入、ドメイン順序設計。

---

## 7. 次の計画（未実施・要承認）
1. **exp_003: fine-tune ベースライン**
   - exp_001 設定 (800,1333) のもと、両極端（underwater=0.051 / microscopic=0.002）を fine-tune し
     zero-shot からの伸び＝適応余地（upper bound）を測定。
2. **忘却プロトコルの確定**
   - exp_001 の COCO=0.504 を ZCOCO 基準に固定。逐次 fine-tune 後の ZCOCO 低下を測る評価系を設計
     （ZiRa の Avg/hAP に準拠）。
3. **継続学習本体の設計**
   - ドメイン提示順序（高→低 / 低→高 / ランダム）の影響検証。
   - ZiRa 型（凍結＋小アダプタ＋ZiL）の MM-Grounding DINO への適用検討。

---

## 関連
- 出発点表: `../exp_001/results/zeroshot_summary.md`
- リサイズ比較: `../exp_002/results/resize_comparison.md`
- 論文要約: `../../papers/`
