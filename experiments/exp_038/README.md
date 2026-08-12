# exp_038 実行手順（素の LoRA の ODinW-13 逐次学習）

design: [[design]]。同一条件の先行実験は [[../exp_034/design]]（ZiRa）と
[[../exp_037/design]]（DitHub）。

## 0. 状態

| | 状態 |
|---|---|
| 実装（`projects/lora_cl/`） | 完了。検証 10/10・10/10・10/10 |
| config・ドライバ・検証スクリプト | 生成済み |
| 実行前検証 9 項目 | 9/9 OK（2026-08-11） |
| 学習ループの試走（30 iter → merge → plain 評価） | OK（pistols mAP 0.7050） |
| 逐次学習・評価 | 実行中（2026-08-11 開始、承認済み） |

## 1. config の再生成（必要なとき）

タスク定義（`experiments/exp_034/odinw_official_tasks.py`）を変えた場合のみ。

```bash
cd /workspace/kouyou/mmdetection
python experiments/exp_038/gen_configs.py
```

`lora_odinw13_base.py` だけは手書きなので上書きされない。

## 2. 実行前検証

```bash
cd /workspace/kouyou/mmdetection
python experiments/exp_038/check_lora_odinw13_setup.py
python projects/lora_cl/check_lora_setup.py
python projects/lora_cl/check_mha_decompose.py
python projects/lora_cl/check_backbone_lora.py
```

順に 9/9・10/10・10/10・10/10 が出ること。**1 つでも NG があれば学習を始めない。**

## 3. 逐次学習と評価

GPU 1 枚で約 18 時間。並列実行はしない。

```bash
cd /workspace/kouyou/mmdetection
GPU=0 nohup bash experiments/exp_038/run_sequential_lora_odinw13.sh 42 \
  > experiments/exp_038/run_s42.log 2>&1 &

tail -f experiments/exp_038/run_s42.log
```

**途中で止まっても同じコマンドで再開できる。**θt
（`lora_s42_theta/theta_tNN_<task>.pth`）が出来ているタスクはスキップする。
t=0 の θ0 評価も `eval/odinw_theta0/` があればスキップする。

## 4. 進捗の確認

```bash
# どのタスクまで終わったか
ls experiments/exp_038/lora_s42_theta/

# 直近の学習ログ
tail -30 experiments/exp_038/lora_s42_t*_work_dir/*/*.log | tail -40

# LoRA が想定どおり入っているか（各タスクの冒頭に出る）
grep -h "\[LoRA\] r=" experiments/exp_038/lora_s42_t*_work_dir/*/*.log
```

期待する出力は各タスク 1 行ずつの
`[LoRA] r=16 alpha=8 付与層=232 ... 内訳={'backbone': 51, 'language_model': 72, 'encoder': 108, 'text_feat_map': 1} trainable=5,671,936/178,649,629 (3.17%)`。

## 5. 出力

| パス | 内容 |
|---|---|
| `eval/odinw_theta0/`・`eval/zcoco_theta0/` | t=0（θ0）のゼロショット評価 |
| `lora_s42_t{01..13}_<task>_work_dir/` | 学習ログと ckpt（base＋LoRA） |
| `lora_s42_theta/theta_t{01..13}_<task>.pth` | マージ後の θt（plain 構造） |
| `eval/odinw_after_t{01..13}/` | 学習済みタスクの評価 |
| `eval/zcoco_after_t{01..13}/` | ZCOCO の評価 |

## 6. 完了後

結果は `experiments/exp_038/results/summary.md` に事実のみ記録する（行動原理8）。
比較対象は exp_034（Avg 0.5942 / ZCOCO 0.4980）、exp_037（Avg 0.6466 / ZCOCO 0.4980）、
θ0（Avg 0.4971 / ZCOCO 0.5040）。

## 注意

- **`*_work_dir` 配下は編集しない**（禁止則）。
- 学習中に他の GPU 作業を同時に走らせない。
- ディスクは 学習 ckpt 13 本 ＋ θ 13 本で約 30 GB。
