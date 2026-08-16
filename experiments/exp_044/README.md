# exp_044 実行手順（クラスタ・新規 clone から結果転送まで）

design: [[design]]。実験ノート: [[../../experiment_notes/note14]]。
**実行は design.md の承認後**（行動原理3。2026-08-16 承認済み）。

クラスタ側の作業ディレクトリは**新規 clone の `/home/kouyou/VLM_OD_CL_exp044`**
（design.md §8。2026-08-16 指示）。

## 0. 状態

| | 状態 |
|---|---|
| 検出器 `KDProjGroundingDINO`・config 3 本・ドライバ・sbatch・検証 | 実装済み（本環境） |
| 実行前検証 6 項目（等価性・全データ化・1 step 込み） | 6/6 OK（2026-08-16、本環境） |
| クラスタへの push / clone / 投入 | 未実施 |

## 1. 本環境で検証してから push

```bash
cd /workspace/kouyou/mmdetection
python experiments/exp_044/check_exp044_setup.py                  # 静的 2 項目
CUDA_VISIBLE_DEVICES=0 python experiments/exp_044/check_exp044_setup.py --build --step

git add experiments/exp_044/ mmdet/models/detectors/kd_proj_grounding_dino.py
git commit -m "add exp_044"
git push
git log --oneline -1
```

## 2. クラスタで新規 clone

```bash
ssh kouyou@192.168.170.100
git clone https://github.com/KoyoImai/VLM_OD_CL.git /home/kouyou/VLM_OD_CL_exp044
cd /home/kouyou/VLM_OD_CL_exp044
git log --oneline -1                       # §1 の commit と一致すること
ls experiments/exp_044/configs/ | wc -l    # 3 本
```

**転送が必要な `.pth` は無い**（θ0 から開始。t=1 の教師 θ0 は `/home/kouyou/ckpt` の
bind、データ・バッファは既存の bind / git 管理物で揃う）。

## 3. 投入（1 ジョブ）

```bash
cd /home/kouyou/VLM_OD_CL_exp044
sbatch experiments/exp_044/sbatch_kdEp.sh     # 約 35〜40 h（前半 3 ドメイン）
squeue -u kouyou
```

exp_043 の 2 ジョブと合わせて 3 ジョブ（同時実行 4 本の制限内）。

## 4. 進捗の確認

```bash
squeue -u kouyou
tail -f /home/kouyou/logs/result_exp044_kdEp_<JOBID>.txt

# どのドメインまで終わったか / 教師の受け渡し
ls -d /home/kouyou/VLM_OD_CL_exp044/experiments/exp_044/*_work_dir/
grep -m3 "教師 θ^T" /home/kouyou/logs/result_exp044_kdEp_<JOBID>.txt

# 蒸留損失の推移。**projector が標準初期化のため学習初期の loss_kd は大きい**
# （本環境の 1 step 実測: λ込みで ~830。grad clip 0.1 が効くので発散はしない想定。
#  design.md §6）。序盤で減少していくことを確認する。
grep -o "loss_kd: [0-9.]*" /home/kouyou/logs/result_exp044_kdEp_<JOBID>.txt | head
```

VRAM 実測（本環境、batch 6/GPU・教師込み）: 13.3〜14.5 GB（A6000 48GB に収まる）。

## 5. 結果の転送（クラスタ → 本環境）

ジョブ終了後、本環境で:

```bash
cd /workspace/kouyou/mmdetection
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp044/experiments/exp_044
LOCAL=experiments/exp_044

rsync -avh --progress \
  --include='*/' --include='*eval*/***' --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress \
  --include='*/' --include='*.log' --include='scalars.json' --include='config.py' \
  --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --include='*/' --include='epoch_20.pth' --exclude='*' "$REMOTE/" "$LOCAL/"
```

ckpt は 3 × 約 1.96 GB（projector 約 0.53M パラメータ分だけ従来よりわずかに大きい）。

## 6. 完了後

結果は `experiments/exp_044/results/summary.md` に事実のみ記録する（行動原理8）。
判定材料は design.md §3（各 t の全ドメイン mAP と ZCOCO、note07 の方法
＝exp_035（前半 3 ドメイン、同一範囲）との対照、リプレイのみ exp_027 との対照）。

## 7. 注意

- 教師はドライバが `--cfg-options model.teacher_ckpt=` で毎タスク明示（t=1 は θ0）。
- projector は state_dict（`kd_proj_*` 16 キー）に含まれ、load_from で次タスクへ
  引き継がれる。評価は plain config で行い projector は読まれない（unexpected keys は
  無害）。この ckpt を他実験の初期値に流用する場合も同様。
- `*_work_dir` 配下は編集しない（禁止則）。
- 途中で止まっても同じコマンドで再開できる（epoch_20.pth があるドメインはスキップ）。
