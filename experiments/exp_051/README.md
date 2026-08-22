# exp_051 実行手順（蒸留適用箇所アブレーション・クラスタ 3 ジョブ）

design: [[design]]。**実行は design.md の承認後（2026-08-22 承認済み）。**
push・clone・sbatch はユーザー実施（2026-08-22 の運用）。

条件: exp_043 kdE と同一枠（バッチ [現在4,汎用2,過去2]×8・λ=10・L2・バッファ由来
4 枚限定・教師 θ_{t-1}・20 epoch・lr 1e-4・seed 0・AMP 無し）で **kd.targets のみ**
kdA=['img'] / kdB=['txt'] / kdD=['fus']。6 ドメイン通し。
ckpt は **epoch_20 のみ・optimizer 状態なし**。

## 0. 状態

| | 状態 |
|---|---|
| config 18 本・ドライバ・sbatch 3 本・検証 | 生成済み |
| 実行前検証（check_exp051_setup.py --step） | **5/5 OK**（2026-08-22。kdE との diff = targets と ckpt 方針のみ・1 step 有限・診断値が対象と一致・buf=slice(4,8)） |
| クラスタ投入 | 未実施 |

## 1. 本環境で検証（実施済み・再現手順）

```bash
python experiments/exp_051/check_exp051_setup.py           # 静的 2 項目
CUDA_VISIBLE_DEVICES=0 python experiments/exp_051/check_exp051_setup.py --step   # 全 5 項目
```

## 2. クラスタ（ユーザー実施）

```bash
# 本環境: commit & push（feats 等の大物は無い。ckpt/pt 系は gitignore 済み）
git add experiments/exp_051/
git commit -m "add exp_051 (KD tap-point ablation, exp_043 frame)"
git push

# クラスタ: 専用クローン（実行中の他クローンに触れない。PULL_WHILE_RUNNING.md）
git clone https://github.com/KoyoImai/VLM_OD_CL.git /home/kouyou/VLM_OD_CL_exp051
cd /home/kouyou/VLM_OD_CL_exp051
git log --oneline -1
ls experiments/exp_051/configs/ | wc -l    # 18
sbatch experiments/exp_051/sbatch_kdA.sh   # 各 約 90〜120 h（--time=144:00:00）
sbatch experiments/exp_051/sbatch_kdB.sh
sbatch experiments/exp_051/sbatch_kdD.sh
squeue -u kouyou
```

- リプレイありのため **Objects365 の bind が必要**（sbatch に設定済み。exp_043 と同じ）。
- exp_050（DGS）等と合わせて同時実行 4 本制限に掛かる分はキュー待ち（自動開始）。

### 投入後 10 分の確認

```bash
LOG=/home/kouyou/logs/result_exp051_kdA_<JOBID>.txt   # kdB/kdD も同様
grep -m2 "t=1/6\|教師" $LOG
grep -m3 "Epoch(train)" $LOG
grep -m2 "kd_img\|kd_txt\|kd_fus" $LOG   # 条件に対応する診断値だけが出る
```

t=1 の初期 loss_kd の目安（本環境 1 step 実測・学生=教師=θ0）:
kdA 0.8 / kdB 37（テキスト項は BERT の dropout 差で大きい）/ kdD 0.009。
数百のオーダーが出たら load_from の渡し忘れを疑う。

## 3. 再開

同じ sbatch を再投入（学習は epoch_20.pth、評価はディレクトリ有無でスキップ。
**json の無い評価ディレクトリは消してから**）。

## 4. 結果の転送（クラスタ → 本環境）

```bash
cd /workspace/kouyou/mmdetection
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp051/experiments/exp_051
LOCAL=experiments/exp_051
rsync -avh --progress \
  --include='*/' --include='*eval*/***' --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress \
  --include='*/' --include='*.log' --include='scalars.json' --include='config.py' \
  --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --include='*/' --include='epoch_20.pth' --exclude='*' "$REMOTE/" "$LOCAL/"
# 容量: 0.7 GB × 6 ドメイン × 3 条件 ≒ 13 GB
```

## 5. 完了後

結果は `experiments/exp_051/results/summary.md` に事実のみ記録（行動原理8）。
判定材料は design §3（各 t の mAP・ZCOCO、対 exp_043 kdE/replay、対 exp_035 系 A/B/D）。
