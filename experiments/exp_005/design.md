# exp_005: 他ドメインへの凍結 fine-tune — 忘却の普遍性確認

## 目的
exp_004（underwater, backbone・言語 凍結）と**同一設定**で underwater 以外の各ドメインを fine-tune し、
**すべてのドメインで COCO 精度低下（事前学習知識の忘却）が起きること**を確認する。

- 検証したい主張: 忘却は underwater 固有ではなく、**ドメインに依らず普遍的に発生**する。
- 主指標: 各ドメイン fine-tune 後の **COCO mAP（zero-shot 0.504 からの低下）**。
- 副指標: 各ドメイン自身の適応 mAP（fine-tune が効いていることの確認）。

## 対象ドメイン（5ドメイン）
real_world 除外、underwater は exp_004 で実施済み。
aerial / videogames / microscopic / documents / electromagnetic。

| ドメイン | クラス数 | train枚数 | max_text_len | 想定学習時間（目安） |
|---|---:|---:|---:|---|
| aerial | 22 | 6,643 | 256 | 短（最小規模） |
| microscopic | 28 | 9,576 | 256 | 中 |
| videogames | 87 | 8,233 | **512** | 中（テキスト長でメモリ大） |
| documents | 59 | 17,866 | 256 | 長 |
| electromagnetic | 39 | 25,398 | 256 | 最長 |

## 実験設定（exp_004 を完全継承）
- 学習: **backbone・language_model 凍結（lr_mult=0.0）**、lr=1e-4、
  実効バッチ64（per-GPU8×4GPU×累積2）、20 epochs(milestone[15])、AdamW wd=1e-4、4GPU 分散。
- **乱数設定: `randomness = dict(seed=0, deterministic=False)`**（全ドメイン共通で seed 固定）。
  - 今後の実験は seed 固定で実施する方針（exp_003/exp_004 は seed 未固定のまま据え置き）。
  - deterministic=False: seed でデータ順序・拡張・初期化は再現。GPU 数値はわずかに揺れうるが速度優先（研究標準）。
- config: `experiments/exp_005/configs/<domain>_finetune_frozen.py`（各ドメイン config を継承し
  exp_004 と同一の凍結・batch・lr を上書き。検証済み）。
- 初期重み: 事前学習 `...grit9m_v3det...pth`（継承）。
- 評価: (800,1333) プロトコル、4GPU 分散。各ドメイン自身（適応）＋ COCO（忘却）。

## 実行コマンド（各ドメイン共通、5回繰り返し）
```bash
D=<domain>   # aerial / videogames / microscopic / documents / electromagnetic
WD=./experiments/exp_005/${D}_work_dir

# 1) 学習（凍結）
bash tools/dist_train.sh experiments/exp_005/configs/${D}_finetune_frozen.py 4 --work-dir $WD

# 2) 適応評価（best ckpt, 当該ドメイン）
BEST=$(ls $WD/best_coco_bbox_mAP_epoch_*.pth | tail -1)
bash tools/dist_test.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_${D}.py \
  "$BEST" 4 --work-dir ./experiments/exp_005/${D}_eval

# 3) 忘却評価（同 best ckpt, COCO）
bash tools/dist_test.sh configs/mm_grounding_dino/eval_base_coco.py \
  "$BEST" 4 --work-dir ./experiments/exp_005/${D}_coco_forgetting_eval
```

## 比較の基準値
| 指標 | zero-shot | underwater凍結FT (exp_004) | 本実験で取得 |
|---|---:|---:|---|
| 各ドメイン mAP（適応） | exp_001参照 | 0.316 | 5ドメイン分 |
| COCO mAP（忘却/ZCOCO） | 0.504 | 0.414（−17.9%） | 5ドメイン分 |

## 期待される成果物
- `experiments/exp_005/<domain>_work_dir/` … 各学習ログ・best ckpt
- `experiments/exp_005/<domain>_eval/`, `<domain>_coco_forgetting_eval/` … 評価ログ
- `experiments/exp_005/results/forgetting_across_domains.md` … 全ドメインの適応／COCO忘却 一覧表
  （underwater(exp_004) 込みの 6 ドメイン比較）
- `experiments/exp_005/outputs/notes.md` … 考察・次の問い

## 想定される懸念点
1. **学習時間が長い**: 5ドメインを 4GPU で逐次実行。electromagnetic(25,398枚)・documents(17,866枚) が
   特に長く、**全体で数十時間規模**になる見込み。1ドメインずつ実行し、完了ごとに結果を確認する。
2. **OOM リスク**: videogames は max_text_len=512 でテキスト分岐のメモリが増える。per-GPU8 でも
   ピークが 40GB に迫る可能性 → 学習開始直後に監視し、OOM 時は停止・報告（exp_003 同様）。
3. **COCO 評価のロード警告**: 各ドメイン(クラス数≠80) で学習した重みを COCO(80) 評価にロード。
   grounding head はテキスト対照ベースで動作する想定。

## 成功基準
- 5ドメインの学習・評価が完走し、各ドメインの適応 mAP と COCO 忘却 mAP を取得すること。
- **全ドメインで COCO mAP が zero-shot(0.504) から低下する**ことを確認し、忘却の普遍性を示すこと。
- ドメイン別の忘却量の違い（どのドメインが最も忘却を誘発するか）を考察し、次の問いを提示すること。
