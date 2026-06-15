# exp_003: underwater 単一ドメイン fine-tuning（適応 upper bound ＋ 忘却の初測定）

## 目的
事前学習済み MM-Grounding DINO を **underwater 単一ドメインで fine-tune** し、以下を測る。
1. **適応性能（upper bound）**: underwater mAP が zero-shot(0.051) からどこまで伸びるか＝ドメイン適応の余地。
2. **破滅的忘却の初測定**: fine-tune 後に COCO を評価し、ZCOCO が zero-shot(0.504) からどれだけ低下するか。

継続学習の2軸（適応 / 忘却）を単一ドメインで初めて同時に観測し、以降の継続学習手法の比較基盤とする。

## 背景・位置づけ
- exp_001 で zero-shot lower bound（underwater=0.051, COCO=0.504）を取得済み。
- exp_002.5 で評価プロトコルを (800,1333) に確定。本実験の評価もこれに従う。
- underwater を選定: zero-shot 中程度(0.051)、28クラスで 256 トークン内（max_text_len 変更不要）、
  train 12,633 枚と適度な規模で最初の fine-tune ベースラインに適する。

## 実験設定
### 学習
- config: **`experiments/exp_003/configs/underwater_finetune.py`**（元ドメイン config を継承し凍結範囲のみ変更）
  - 元 `configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py` は無変更。
- 実行: **GPU 4 枚 分散学習**（`tools/dist_train.sh ... 4`）。実効バッチ = 16(batch_size) × 4(GPU) = **64**。

#### ハイパーパラメータの方針（既存研究比較に基づく決定）
- **学習可能パラメータ**: backbone・language_model を **lr_mult=0.1 で学習**（元は凍結 lr_mult=0.0）。
  MM-GDINO の COCO fine-tune に準拠し、適応性能（upper bound）を狙う。検出 head/encoder/decoder/neck は通常 lr。
- **バッチサイズ**: per-GPU **16** × 4GPU = **実効64**（勾配累積は使わない）。
- **学習率**: **1e-4**（実効64。線形整合値 2e-4 より保守的に半分へ設定。安定性・忘却抑制を優先）。
- **エポック**: 現行維持（20 epochs, milestone[15] gamma0.1）。
- 比較した既存研究: MM-GDINO COCO FT(lr2e-4,12ep,bb0.1×,bs64) / ZiRa(全凍結+アダプタ,lr1e-3,2ep,bs2) /
  RF100-VL(lr3e-4,1000iter,bs4)。詳細は exp_003/outputs/notes.md に記録予定。
- ※ per-GPU 16 は VRAM 消費が大きく A100 40GB で OOM の可能性あり。学習開始直後に要監視。

#### 初期化・モデル
- 初期重み（load_from）:
  `grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`
- `model.bbox_head.num_classes = 28`（underwater クラス数）。`max_text_len` は既定 256（変更不要）。
- data_preprocessor: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], BGR→RGB。

#### 最適化（optim_wrapper）
- type: `OptimWrapper`
- optimizer: **AdamW, lr=1e-4 (0.0001), weight_decay=1e-4 (0.0001)**
- clip_grad: `max_norm=0.1, norm_type=2`
- paramwise_cfg.custom_keys（**exp_003 で変更**）:
  - `absolute_pos_embed`: decay_mult=0.0（位置埋め込みは weight decay 無効）
  - **`backbone`: lr_mult=0.1（0.1×lr=1e-5 で学習。元の凍結 0.0 から変更）**
  - **`language_model`: lr_mult=0.1（0.1×lr=1e-5 で学習。元の凍結 0.0 から変更）**
  - → 検出 head/encoder/decoder/neck は lr=1e-4、Swin backbone と BERT は lr=1e-5 で学習（全体 fine-tune）。
- auto_scale_lr: `enable=False`（base_batch_size=64。lr は自動スケールしない＝上記 lr をそのまま使用）

#### スケジュール
- train_cfg: `EpochBasedTrainLoop, max_epochs=20, val_interval=1`
- param_scheduler: `MultiStepLR, begin=0, end=20, by_epoch=True, milestones=[15], gamma=0.1`
  - epoch 0–14: lr=1e-4 / epoch 15–19: lr=1e-5（×0.1）。backbone・言語は各 0.1×（1e-5 → 1e-6）。

#### データ・拡張（train_pipeline）
- train_dataloader: batch_size=16（per-GPU, ×4GPU=実効64）, num_workers=4, persistent_workers=True,
  sampler=`DefaultSampler(shuffle=True)`, batch_sampler=`AspectRatioBatchSampler`
- dataset: `CocoDataset`, ann=`train/_annotations.coco.json`, return_classes=True,
  filter_cfg=`dict(filter_empty_gt=False, min_size=32)`
- pipeline: LoadImageFromFile → LoadAnnotations → **RandomFlip(prob=0.5)** →
  **RandomChoice（2分岐から1つ選択, すべて keep_ratio=True）**:
  - 分岐0: `RandomChoiceResize` 短辺∈{480,512,…,800}, 長辺≤1333
  - 分岐1: `RandomChoiceResize` (400/500/600, ≤4200) → `RandomCrop absolute_range (384,600)` →
    `RandomChoiceResize` 短辺∈{480,…,800}, 長辺≤1333
  - → PackDetInputs。**全リサイズ keep_ratio=True で歪みなし**（[[../exp_002/outputs/resize_aug_investigation]]）。

#### 評価（学習中・hooks）
- val_interval=1（毎 epoch 検証）。検証パイプラインは (800,1333) keep_ratio=True（exp_002.5 プロトコル）。
- default_hooks.checkpoint: `interval=1, max_keep_ckpts=1, save_best='auto'`
  → `best_coco_bbox_mAP_epoch_*.pth` を保持。
- default_hooks.logger: `interval=5`

### 評価（fine-tune 後、best チェックポイントを使用）
評価はすべて (800,1333) プロトコル・4GPU 分散。
1. **適応**: underwater valid（学習中の val でも得られるが、best ckpt で明示的に test）
2. **忘却**: COCO2017 val を `eval_base_coco.py` ＋ fine-tuned ckpt で評価

## 実行コマンド
```bash
CKPT_DIR=./experiments/exp_003/underwater_work_dir

# 1) 学習（4GPU 分散）
bash tools/dist_train.sh \
  experiments/exp_003/configs/underwater_finetune.py \
  4 --work-dir $CKPT_DIR

# 2) 適応評価（best ckpt）
BEST=$(ls $CKPT_DIR/best_coco_bbox_mAP_epoch_*.pth | tail -1)
bash tools/dist_test.sh \
  configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_underwater.py \
  "$BEST" 4 --work-dir ./experiments/exp_003/underwater_eval

# 3) 忘却評価（同じ best ckpt を COCO で）
bash tools/dist_test.sh \
  configs/mm_grounding_dino/eval_base_coco.py \
  "$BEST" 4 --work-dir ./experiments/exp_003/coco_forgetting_eval
```

## 比較の基準値（exp_001 zero-shot）
| 指標 | zero-shot | fine-tune 後（本実験で取得） |
|---|---:|---|
| underwater mAP（適応） | 0.051 | ↑（upper bound） |
| COCO mAP（忘却 / ZCOCO） | 0.504 | ↓（忘却量 = 0.504 − 測定値） |

## 期待される成果物
- `experiments/exp_003/underwater_work_dir/` … 学習ログ・best チェックポイント・scalars.json
- `experiments/exp_003/underwater_eval/`, `coco_forgetting_eval/` … 評価ログ
- `experiments/exp_003/results/finetune_summary.md` … 適応／忘却の数値表
- `experiments/exp_003/outputs/notes.md` … 考察・次の問い

## 想定される懸念点
1. **COCO 評価時のチェックポイント整合**: underwater(28クラス) で fine-tune した重みを COCO(80クラス) 評価に
   ロードする。grounding head はテキスト対照ベースで num_classes 非依存のため動作する想定だが、
   ロード時の警告（head の shape 等）を確認する。
2. **学習時間**: 12,633 枚 × 20 epochs / 4GPU。数十分〜1.5時間程度を見込む。OOM がないか初回監視。
3. **忘却の解釈**: backbone・言語モデルが凍結のため、忘却は検出 head/encoder/decoder の適応に起因。
   ZiRa 等の「全凍結＋アダプタ」とは凍結範囲が異なる点に留意（本実験は標準 fine-tune ベースライン）。

## 成功基準
- 学習が完走し、best チェックポイントが得られること。
- underwater 適応 mAP（zero-shot 0.051 からの伸び）と COCO 忘却 mAP（0.504 からの低下）の双方を記録すること。
- 適応と忘却のトレードオフを定量化し、次の問い（継続学習手法、凍結範囲、忘却抑制）を提示すること。
