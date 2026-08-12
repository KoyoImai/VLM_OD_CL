# exp_037 実行手順（DitHub の ODinW-13 再現）

design: [[design]]。ZiRa の同一条件の実験は [[../exp_034/design]]。

**実行は design.md の承認後**（行動原理3）。以下は承認後の手順である。

## 0. 現状

| | 状態 |
|---|---|
| 実装（修正 5 件） | 完了。回帰テスト 8/8 OK |
| config・ドライバ・検証スクリプト | 生成済み |
| 実行前検証 9 項目 | 9/9 OK（2026-08-10） |
| 逐次学習・評価 | **未実行（承認待ち）** |

## 1. config の再生成（必要なとき）

タスク定義（`experiments/exp_034/odinw_official_tasks.py`）を変えた場合のみ。

```bash
cd /workspace/kouyou/mmdetection
python experiments/exp_037/gen_configs.py
```

生成物は `configs/` 配下の学習 13 本・評価 1 本・ZCOCO 1 本・部分集合 13 本。
`dithub_odinw13_base.py` だけは手書きなので上書きされない。

## 2. 実行前検証

```bash
cd /workspace/kouyou/mmdetection
python experiments/exp_037/check_dithub_odinw13_setup.py
python experiments/exp_037/check_dithub_phase_fix.py
python experiments/exp_021/check_dithub_setup.py
```

順に 9/9・8/8・10/10 が出ること。**1 つでも NG があれば学習を始めない。**

## 3. 逐次学習と評価

GPU 1 枚で 12〜15 時間。並列実行はしない。

```bash
cd /workspace/kouyou/mmdetection
GPU=0 nohup bash experiments/exp_037/run_sequential_dithub_odinw13.sh 42 \
  > experiments/exp_037/run_s42.log 2>&1 &

tail -f experiments/exp_037/run_s42.log
```

**途中で止まっても同じコマンドで再開できる。**ライブラリ ckpt
（`dithub_s42_library/lib_after_tNN_<task>.pth`）が出来ているタスクはスキップする。

## 4. 進捗の確認

```bash
# どのタスクまで終わったか
ls experiments/exp_037/dithub_s42_library/

# 直近の学習ログ
tail -30 experiments/exp_037/dithub_s42_t*_work_dir/*/*.log | tail -40

# 実効 λ ではなくフェーズが正しく初期化されたか（各タスクの冒頭に出る）
grep -h "phase initialized" experiments/exp_037/dithub_s42_t*_work_dir/*/*.log
```

期待する出力は各タスク 1 行ずつの
`>>> DitHub: phase initialized to WARMUP (iter=0, epoch=0), warmup_lora_a reinit=True <<<`。
**SPECIALIZATION が出ていたらバグ①が再発している**（design §1.1）。

## 5. 出力

| パス | 内容 |
|---|---|
| `dithub_s42_t{01..13}_<task>_work_dir/` | 学習ログと ckpt |
| `dithub_s42_library/lib_after_t{01..13}_<task>.pth` | 成長ライブラリ（式4 適用済み） |
| `eval/odinw_after_t{01..13}/` | 学習済みタスクの評価 |
| `eval/zcoco_after_t{01..13}/` | ZCOCO の評価 |

## 6. 完了後

結果は `experiments/exp_037/results/summary.md` に事実のみ記録する（行動原理8）。
θ0 の起点は exp_034 で測定済みの値を使う（Avg 0.4971 / ZCOCO 0.5040）。再測定はしない。

## 注意

- **`*_work_dir` 配下は編集しない**（禁止則）。
- 学習中に他の GPU 作業を同時に走らせない（design §4.4）。
- ディスクは ckpt 13 本 ＋ ライブラリ 13 本で約 30 GB。
