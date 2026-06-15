# exp_002: 評価時データリサイズの感度分析と学習・評価方法の計画

## 目的
zero-shot 評価の**入力リサイズ設定**を変えたときに精度がどう変わるかを測定し、
exp_001（`(800,1333)`）と比較する。これにより、**今後の学習・評価で採用するリサイズ方法を決定**する。

背景：RF100 のデータはネイティブ 640×640。exp_001 の `(800,1333)` は正方形画像を 800×800 に
アップスケールしている。アップスケールが小物体検出に効くのか、ネイティブ 640 で十分か、
さらに高解像度が有効か、アスペクト比無視（正方形固定）が有害かを切り分ける。

## 比較するリサイズ設定（計 5 種、うち4種を本実験で新規評価）
すべて `test_pipeline` の `FixScaleResize` を上書き（`--cfg-options`）。

| 設定 | scale | keep_ratio | 640×640画像での実効サイズ | 位置づけ |
|---|---|---|---|---|
| **baseline** | (800,1333) | True | 800×800 | exp_001 で取得済（参照） |
| S1 | (640,1333) | True | 640×640（リサイズなし） | ネイティブ / RF100-VL 公式 |
| S2 | (1024,1333) | True | 1024×1024 | アップスケール |
| S3 | (1333,2000) | True | 1333×1333 | 高解像度 |
| S4 | (640,640) | **False** | 640×640（比無視） | 正方形固定 / RF100 YOLO 条件 |

## 評価対象（COCO + 6 ドメイン = 7 対象）
real_world 除外。COCO は画像サイズ可変（640固定ではない）ため、S1/S4 では多くが縮小される点に留意。

| 対象 | config |
|---|---|
| COCO2017 | `grounding_dino_swin-t_pretrain_obj365.py` |
| underwater/aerial/videogames/microscopic/documents/electromagnetic | `..._finetune_8xb4_20e_<domain>.py` |

→ 新規評価数 = **4 設定 × 7 対象 = 28 評価**。baseline(800,1333) は exp_001 の値を流用。

## 実験設定
- モデル/重み: exp_001 と同一（事前学習 MM-Grounding DINO Swin-T、fine-tune なし）。
- 実行: **GPU 4 枚 分散推論**（`tools/dist_test.sh ... 4`）。
- 指標: COCO bbox mAP（特に **mAP_s** をリサイズ効果の主指標として注視）。
- **リサイズ設定は専用 config ファイルで管理する（再現性のため）。**
  `--cfg-options` は使わず、`experiments/exp_002/configs/<target>_<setting>.py` を `test.py` に渡す。
  各派生 config は「`_base_` で元 config を継承 ＋ `test_pipeline` の FixScaleResize のみ上書き」する薄いファイル。
  元 config（`configs/mm_grounding_dino/...`）は読み込み参照のみで一切変更されない。

## 使用する config ファイル（`experiments/exp_002/configs/`、計 28 本 = 7対象 × 4設定）
命名規則: `<target>_<setting>.py`。各 setting の中身は FixScaleResize の `scale` / `keep_ratio` のみ異なる。

| target | _base_（継承元） |
|---|---|
| coco | `../../../configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py` |
| underwater 他5ドメイン | `../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_<domain>.py` |

| setting 接尾辞 | FixScaleResize |
|---|---|
| `s1_640`    | scale=(640,1333),  keep_ratio=True |
| `s2_1024`   | scale=(1024,1333), keep_ratio=True |
| `s3_1333`   | scale=(1333,2000), keep_ratio=True |
| `s4_640sq`  | scale=(640,640),   keep_ratio=False |

例（`underwater_s1_640.py`）:
```python
_base_ = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py'
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=(640, 1333), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id','img_path','ori_shape','img_shape',
                    'scale_factor','text','custom_entities','tokens_positive'))
]
val_dataloader = dict(dataset=dict(return_classes=True, pipeline=test_pipeline))
test_dataloader = val_dataloader
```

## 実行コマンド（config ファイル方式）
```bash
bash tools/dist_test.sh \
  experiments/exp_002/configs/<target>_<setting>.py \
  <事前学習ckpt URL> \
  4 \
  --work-dir ./experiments/exp_002/<target>_<setting>
```
work-dir 命名規則: config と同じ `<target>_<setting>`（例: `underwater_s1_640`, `coco_s3_1333`）。

## 期待される成果物
- `experiments/exp_002/configs/` … リサイズ設定別 config 28 本（生成済み）
- `experiments/exp_002/<target>_<setting>/` … 各評価ログ
- `experiments/exp_002/results/resize_comparison.md` … 対象 × リサイズ設定の mAP 比較表（exp_001 baseline 込み）
- `experiments/exp_002/outputs/notes.md` … 考察 ＋ **今後の学習・評価リサイズの推奨と根拠**

## 想定される懸念点
1. **派生 config の妥当性**: ロード検証済み（リサイズ反映・データパス・クラス・return_classes・load_from の
   継承を確認）。最初の1本で実行時の img_shape も確認してから全体を回す。
2. **メモリ/時間**: S3(1333) は 1333×1333 で VRAM・時間が増大。A100 40GB で OOM しないか初回監視。
3. **COCO の扱い**: COCO は 640 固定でないため S1/S4 の意味が RF100 とは異なる。比較表では別枠で解釈する。
4. **学習側との整合**: 本実験は評価のみ。決定したリサイズは「学習・評価で統一」する前提で次実験に反映する。

## 成功基準
- 28 評価が完走し、exp_001 baseline と合わせて **対象7 × 設定5 の mAP 比較表**が揃うこと。
- リサイズと mAP（特に mAP_s）の関係から、**今後の学習・評価で採用するリサイズを根拠付きで決定**できること。
- その決定を notes.md に記し、次実験（fine-tune 系）の design に引き継ぐ計画を提示すること。
