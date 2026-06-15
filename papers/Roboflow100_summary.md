# Roboflow100 / Roboflow100-VL 要約（評価データ・プロトコル）

本プロジェクトのデータ基盤（RF100 を 7 ドメインに統合）。2論文をまとめて要約。

## Roboflow100（オリジナル）
- 100 データセット / 224,714 画像 / 805 クラス。多様な実世界ドメインの検出ベンチマーク。
- **入力解像度: 640×640**（"All images resized to 640x640 pixels"）。
  → 本プロジェクトのデータも 640×640 済みで**ベンチマーク標準に一致**。
- ベースライン: YOLOv5/YOLOv7 を **100 epochs / 640×640 / デフォルトハイパラ**で学習。
- 拡張: YOLO デフォルト（重め: mosaic 等）。指標は mAP@.50（GLIP は mAP@.50:.95）。

## Roboflow100-VL（VL 拡張版）
- オープン語彙/few-shot 検出のための VL ベンチマーク。100 カテゴリ群を CLIP で **7 スーパーカテゴリ**に
  自動クラスタリング（Aerial/Document/Flora&Fauna/Industrial/Medical/Sports/Other）。
- 評価レジーム: **Zero-shot / Few-shot / Semi-supervised / Fully-supervised**。
- 指標: **mAP@.50:.95（pycocotools）**。Ultralytics 評価は最大 2.7% 過大評価するため非採用。
- **Few-shot のデフォルトは 10-shot**（5-shot も）。
- **GroundingDINO の設定（重要・本件に直結）**:
  - リサイズ **(640, 1333)**
  - few-shot fine-tune: **batch 4 / lr 3e-4 / 1000 iterations / 追加データ拡張なし**
- 難易度: GroundingDINO は ODinW-13 で 49.2 mAP → RF100-VL では **15.7 mAP** に大幅低下。
  事前学習に乏しいドメイン（特に医療）でゼロショットが極端に弱い。

## 本プロジェクトへの示唆
- **解像度 640 が RF100 の自然な作業解像度**。短辺 800 への拡大はアップスケールで情報増なし。
- GroundingDINO の確立 FT プロトコルは **リサイズ(640,1333)＋追加拡張なし＋短い反復** と軽量。
- RF100-VL のゼロショット低下（49→15.7）は本研究の課題1（未知ドメインでの性能低下）の裏付け。
  exp_000 の underwater zero-shot=0.051 とも整合。
- 関連: [[MM-GroundingDINO_summary]], [[ZiRaGroundingDINO_summary]]。
