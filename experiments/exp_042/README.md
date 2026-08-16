# exp_042 実行手順（本環境。本走 4 本を 4 GPU 並行）

design: [[design]]。実験ノート: [[../../experiment_notes/note12]]。
**実行は design.md の承認後**（行動原理3。2026-08-15 承認済み）。
実装本体と検証は `projects/ewc_cl/`（EWC）と `projects/lora_cl/`（InfLoRA）、
論文・公開実装との対応は `papers/EWC_implementation_notes.md` /
`papers/InfLoRA_implementation_notes.md`。

## 0. 状態

| | 状態 |
|---|---|
| 実装（EWC / InfLoRA）と検証 | 完了（5/5・8/8 ＋ 挙動検証 ＋ 公開品質監査。2026-08-15） |
| config 26 本・ドライバ 2 本・実行前検証 | 生成済み。検証 7/7 OK（--build --smoke 込み） |
| 本走（EWC 3 λ ＋ InfLoRA） | 未実行 |

## 1. 実行前の確認

```bash
cd /workspace/kouyou/mmdetection
python experiments/exp_042/gen_configs.py                       # 再生成（変更時のみ）
python experiments/exp_042/check_exp042_setup.py                # 静的 5 項目
CUDA_VISIBLE_DEVICES=0 python experiments/exp_042/check_exp042_setup.py --build --smoke
```

## 2. 本走（design.md §2.4–2.5。4 GPU 並行）

パイロットは行わない（2026-08-15 変更）。EWC は λ ∈ {10², 10³, 10⁴} の 3 水準を
3 GPU で並行し、InfLoRA を 4 枚目で走らせる。4 本は独立で、止まったものだけ
同じコマンドで再開できる。

```bash
cd /workspace/kouyou/mmdetection
GPU=0 LAM=100   nohup bash experiments/exp_042/run_main_ewc.sh     > experiments/exp_042/run_ewc_lam100.log   2>&1 &
GPU=1 LAM=1000  nohup bash experiments/exp_042/run_main_ewc.sh     > experiments/exp_042/run_ewc_lam1000.log  2>&1 &
GPU=2 LAM=10000 nohup bash experiments/exp_042/run_main_ewc.sh     > experiments/exp_042/run_ewc_lam10000.log 2>&1 &
GPU=3           nohup bash experiments/exp_042/run_main_inflora.sh > experiments/exp_042/run_inflora.log      2>&1 &
```

各タスクのサイクル:

- EWC: 学習（λ・前状態を --cfg-options で明示）→ Fisher 推定＋2 バッファ更新
  （`ewc_lam{λ}_states/state_tNN.pth`）→ 学習済みタスク＋ZCOCO 評価
  （`eval_ewc_lam{λ}/`）。ckpt は素の GroundingDINO 構造なので exp_038 の
  plain 評価 config をそのまま使う。
- InfLoRA: 設計（`inflora_designs/`）→ 学習（B のみ）→ DualGPM メモリ更新
  （`inflora_memory/`）→ マージ（`inflora_theta/`）→ 同評価（`eval_inflora/`）。
  DualGPM 閾値は公式 DomainNet 設定（0.95 → 1.0 線形）。

概算（design.md §5）: 並行実行なので実時間は最長の 1 本と同じ 14〜18 h 程度。

## 3. 進捗の確認

```bash
tail -f experiments/exp_042/run_ewc_lam1000.log        # 各ログ
grep -o "loss_ewc: [0-9.]*" experiments/exp_042/run_ewc_lam10000.log | tail -3
ls experiments/exp_042/ewc_lam*_states/ experiments/exp_042/inflora_theta/ 2>/dev/null
nvidia-smi
```

λ=10⁴ は序盤で loss が発散しないかを確認する（design.md §6。grad_norm と loss の推移）。

## 4. t=0（θ0）の扱い

評価系は exp_038 と同一の plain config なので、θ0 の値は exp_038 の実測
（`experiments/exp_038/eval/odinw_theta0` ほか。Avg 0.4971 / ZCOCO 0.5040、
exp_034 と一致確認済み）をそのまま t=0 として使う。再測定はしない。

## 5. 結果の整理

事実は `experiments/exp_042/results/summary.md` に記録する（行動原理8）。
判定材料は design.md §3（θ0 起点、適応・忘却・ZCOCO 推移、13×13 表、
ZiRa exp_034 / DitHub exp_037 / 素の LoRA exp_038 との相対、EWC の λ 3 水準の比較）。

## 6. 注意

- λ・state_path・design_path は**ドライバが --cfg-options で毎回明示**する
  （config の既定は不活性値。model. を付けない誤記は model に届かない —
  check_exp042_setup.py 項目 5 で確認済み）。
- `*_work_dir` 配下は編集しない（禁止則）。
- EWC の状態・InfLoRA のメモリは再開時に二重更新しない設計
  （状態/θ が既にあるタスクは学習ごとスキップ）。
