# exp_043 実行手順（クラスタ・新規 clone から結果転送まで）

design: [[design]]。実験ノート: [[../../experiment_notes/note13]]。
**実行は design.md の承認後**（行動原理3。2026-08-16 承認済み）。

クラスタ側の作業ディレクトリは**新規 clone の `/home/kouyou/VLM_OD_CL_exp043`**
（design.md §8。2026-08-16 指示）。

## 0. 状態

| | 状態 |
|---|---|
| config 12 本・ドライバ・sbatch 2 本・検証 | 生成済み（本環境） |
| 実行前検証 5 項目（実バッチ構成・1 step 込み） | 5/5 OK（2026-08-16、本環境） |
| クラスタへの push / clone / 投入 | 未実施 |

## 1. 本環境で検証してから push

```bash
cd /workspace/kouyou/mmdetection
python experiments/exp_043/check_exp043_setup.py                  # 静的 3 項目
CUDA_VISIBLE_DEVICES=0 python experiments/exp_043/check_exp043_setup.py --data --step

git add experiments/exp_043/ && git commit -m "add exp_043"
git push
git log --oneline -1
```

## 2. クラスタで新規 clone

```bash
ssh kouyou@192.168.170.100
git clone https://github.com/KoyoImai/VLM_OD_CL.git /home/kouyou/VLM_OD_CL_exp043
cd /home/kouyou/VLM_OD_CL_exp043
git log --oneline -1                        # §1 の commit と一致すること
ls experiments/exp_043/configs/ | wc -l     # 12 本
```

**転送が必要な `.pth` は無い**（θ0 から始めるため初期パラメータ不要。θ0・データ・
バッファ定義は既存の bind / git 管理物で揃う。kdE の t=1 の教師 θ0 は
`/home/kouyou/ckpt` の bind から解決される）。

## 3. 投入（2 ジョブ並行）

```bash
cd /home/kouyou/VLM_OD_CL_exp043
sbatch experiments/exp_043/sbatch_replay.sh   # 約 75〜85 h
sbatch experiments/exp_043/sbatch_kdE.sh      # 約 90〜100 h
squeue -u kouyou
```

同時実行 4 ジョブ制限に注意（他実験のジョブが 2 本を超えて走っている場合は空き待ち）。

## 4. 進捗の確認

```bash
squeue -u kouyou
tail -f /home/kouyou/logs/result_exp043_kdE_<JOBID>.txt

# どのドメインまで終わったか
ls -d /home/kouyou/VLM_OD_CL_exp043/experiments/exp_043/*_work_dir/

# 投入後 10 分：学習が始まったか / kdE の教師と蒸留損失
grep -m3 "Epoch(train)" /home/kouyou/logs/result_exp043_<COND>_<JOBID>.txt
grep -m3 "教師 θ^T" /home/kouyou/logs/result_exp043_kdE_<JOBID>.txt
grep -o "loss_kd: [0-9.]*" /home/kouyou/logs/result_exp043_kdE_<JOBID>.txt | head -3
```

VRAM の実測（本環境、batch 8/GPU）: replay 13.4 GB / kdE 18.1 GB（A6000 48GB に収まる）。

## 5. 結果の転送（クラスタ → 本環境）

**転送は rsync のみ。ジョブが終わってから行う。**本環境で実行:

```bash
cd /workspace/kouyou/mmdetection
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp043/experiments/exp_043
LOCAL=experiments/exp_043

# 評価結果（最優先・軽い）
rsync -avh --progress \
  --include='*/' --include='*eval*/***' --exclude='*' "$REMOTE/" "$LOCAL/"

# 学習ログ
rsync -avh --progress \
  --include='*/' --include='*.log' --include='scalars.json' --include='config.py' \
  --exclude='*' "$REMOTE/" "$LOCAL/"

# チェックポイント（各ドメインの epoch_20 のみ。2 条件 × 6 × 約 1.95 GB ≒ 23 GB）
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --include='*/' --include='epoch_20.pth' --exclude='*' "$REMOTE/" "$LOCAL/"
```

## 6. 完了後

結果は `experiments/exp_043/results/summary.md` に事実のみ記録する（行動原理8）。
判定材料は design.md §3（各 t の全ドメイン mAP と ZCOCO、replay vs kdE、
既存バッチ構成の系列 exp_027/028/035/040 との対照）。

## 7. 注意

- kdE の教師はドライバが `--cfg-options model.teacher_ckpt=` で毎タスク明示する
  （t=1 は θ0。**model. を付けない誤記は届かない** — 検証項目 3 で確認済み）。
- 蒸留対象はバッチ末尾のバッファ由来 4 枚（`num_buffer_per_batch=4`）。バッチ順序は
  毎イテレーション assert で検査される（崩れると学習が止まる設計）。
- `*_work_dir` 配下は編集しない（禁止則）。
- 途中で止まっても同じコマンドで再開できる（epoch_20.pth があるドメインはスキップ）。
