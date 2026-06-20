# exp_006: 2ドメイン逐次学習 — 1ドメイン目(D1)の忘却確認

## 目的
事前学習モデルではなく **aerial(D1) を学習済みのモデル**から各 D2 を継続学習し、
D2 学習後に **1ドメイン目(aerial)の検出性能が低下する**こと（＝過去ドメインの破滅的忘却）を確認する。
exp_005 までで確認した「COCO（事前学習知識）の忘却」に加え、**「過去に学習したドメインの忘却」**も
起きることを示すのが本実験の主眼。

## 設計：D1 固定・D2 を振る
```
事前学習 →[aerial 学習]→ ckpt_aerial →[D2 を継続学習]→ ckpt_(aerial→D2)
            ↑ exp_005 を再利用              ↑ exp_006 で新規学習（5本）
評価(ckpt_(aerial→D2)): aerial(D1忘却) / D2(適応) / COCO(累積忘却)
```
- **D1 = aerial 固定**。起点は exp_005 の `aerial_work_dir/best_coco_bbox_mAP_epoch_17.pth`。
- **D2 = 残り5ドメイン**: underwater, microscopic, videogames, documents, electromagnetic。
- D1 を変えず D2 を振ることで、「どの D2 が aerial をどれだけ忘却させるか」を比較できる。

## 実験設定（exp_005 を継承、load_from のみ変更）
- 学習: backbone・言語凍結 / lr=1e-4 / 実効バッチ64（per-GPU8×4GPU×累積2）/ 20ep[15] /
  AdamW wd=1e-4 / seed=0,deterministic=False / 4GPU 分散。
- **唯一の差**: `load_from = aerial(D1) チェックポイント`（事前学習重みではなく D1 から継続）。
- config: `experiments/exp_006/configs/aerial_to_<D2>.py`（5本、検証済み）。
- 評価: (800,1333) プロトコル、4GPU 分散。
  - num_classes が D1/D2/COCO で異なるため評価時に dn_query_generator.label_embedding の
    shape 不一致警告が出るが、検出はテキスト対照ベースで eval に影響しない（exp_003/005 と同様）。

## 実行（各 D2 で繰り返し、計5回）
```bash
D2=<domain>
WD=./experiments/exp_006/aerial_to_${D2}_work_dir
# 1) aerial から D2 を継続学習
bash tools/dist_train.sh experiments/exp_006/configs/aerial_to_${D2}.py 4 --work-dir $WD
BEST=$(ls $WD/best_coco_bbox_mAP_epoch_*.pth | tail -1)
# 2) D1=aerial 忘却評価
bash tools/dist_test.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_aerial.py \
  "$BEST" 4 --work-dir ./experiments/exp_006/aerial_to_${D2}_AERIALeval
# 3) D2 適応評価
bash tools/dist_test.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_${D2}.py \
  "$BEST" 4 --work-dir ./experiments/exp_006/aerial_to_${D2}_D2eval
# 4) COCO 累積忘却評価
bash tools/dist_test.sh configs/mm_grounding_dino/eval_base_coco.py \
  "$BEST" 4 --work-dir ./experiments/exp_006/aerial_to_${D2}_COCOeval
```
実行順（学習データ量 昇順）: videogames → microscopic → underwater → documents → electromagnetic。

## 比較の基準値
| 指標 | 基準値 | 出典 |
|---|---:|---|
| aerial 適応（aerial 単独FT後） | 0.468 | exp_005 |
| COCO（aerial 単独FT後） | 0.318 | exp_005 |
| COCO（zero-shot） | 0.504 | exp_001 |
| 各 D2 適応（D2 単独FT後） | underwater0.316 / microscopic0.499 / videogames0.719 / documents0.478 / electromagnetic0.331 | exp_004/005 |

主眼の比較:
- **aerial(D1) 忘却**: aerial 単独FT後 0.468 → aerial→D2 後 で**どれだけ下がるか**（< 0.468 を期待）。
- D2 適応: D2 単独FT（事前学習起点）と、aerial 起点（本実験）で差があるか。
- COCO: zero-shot 0.504 → aerial後 0.318 → aerial→D2 後（累積でさらに変化）。

## 期待される成果物
- `experiments/exp_006/aerial_to_<D2>_work_dir/` … 各継続学習ログ・best ckpt
- `experiments/exp_006/aerial_to_<D2>_{AERIAL,D2,COCO}eval/` … 各評価ログ
- `experiments/exp_006/results/sequential_2domain.md` … D1忘却／D2適応／COCO累積忘却 一覧表
- `experiments/exp_006/outputs/notes.md` … 考察・次の問い

## 想定される懸念点
1. **学習時間**: 5本の継続学習。documents・electromagnetic が長く、全体で数十時間規模。1本ずつ実行。
2. **OOM**: videogames(max_text_len=512) は per-GPU8 でもメモリ厳しめ。監視し OOM 時は停止・報告。
3. **num_classes 不一致**: 上述のとおり eval に影響しない想定（要ロード警告確認）。

## 成功基準
- 5本の継続学習・評価が完走すること。
- **aerial(D1) の性能が、D2 学習後にいずれの D2 でも低下する**ことを確認し、過去ドメイン忘却を実証すること。
- D2 による D1 忘却量の差を考察し、6ドメイン逐次学習（次段）への示唆を提示すること。
