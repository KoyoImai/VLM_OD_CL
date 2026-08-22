# exp_052 実行手順（蒸留E × 学習可能モジュールアブレーション・クラスタ 5 ジョブ）

design: [[design]]。**実行は design.md の承認後（2026-08-22 承認済み）。**
push・clone・sbatch はユーザー実施（2026-08-22 の運用）。

条件: exp_035 kdE と同一枠（旧バッチ [現在4,汎用1,過去1]×6・λ=10・L2・バッファ由来
2 枚限定・教師 θ_{t-1}・20 epoch・lr 1e-4・seed 0・AMP 無し）で **学習可能モジュール
のみ** 5 通りに制限（凍結 = paramwise lr_mult 0.0。境界は exp_015 の区分）。
前半 3 ドメイン（underwater → electromagnetic → videogames）。
ckpt は **epoch_20 のみ・optimizer 状態なし**。

| 条件 | 学習対象（lr>0 実測） |
|---|---|
| swinNeck | backbone(0.1)+neck: 191 個 29.64M |
| bertTfm | language_model(0.1)+text_feat_map: 199 個 109.09M |
| enhancer | encoder: 276 個 21.91M |
| qsel | memory_trans_fc/norm+level_embed+query_embedding: 6 個 0.30M |
| decoder | decoder: 174 個 11.05M |

## 0. 状態

| | 状態 |
|---|---|
| config 15 本・ドライバ・sbatch 5 本・検証 | 生成済み |
| 実行前検証（check_exp052_setup.py --step） | **4/4 OK**（2026-08-22。exp_035 kdE との diff = paramwise と ckpt 方針のみ・lr>0 集合が境界と厳密一致・1 step で凍結不変/学習対象のみ更新） |
| クラスタ投入 | 未実施 |

## 1. 本環境で検証（実施済み・再現手順）

```bash
python experiments/exp_052/check_exp052_setup.py           # 静的 3 項目
CUDA_VISIBLE_DEVICES=0 python experiments/exp_052/check_exp052_setup.py --step   # 全 4 項目
```

## 2. クラスタ（ユーザー実施）

```bash
# 本環境: commit & push（ckpt/pt 系は gitignore 済み）
git add experiments/exp_052/
git commit -m "add exp_052 (kdE trainable-module ablation, exp_035 frame)"
git push

# クラスタ: 専用クローン（実行中の他クローンに触れない。PULL_WHILE_RUNNING.md）
git clone https://github.com/KoyoImai/VLM_OD_CL.git /home/kouyou/VLM_OD_CL_exp052
cd /home/kouyou/VLM_OD_CL_exp052
git log --oneline -1
ls experiments/exp_052/configs/ | wc -l        # 15
sbatch experiments/exp_052/sbatch_swinNeck.sh  # 各 約 30〜45 h（--time=96:00:00）
sbatch experiments/exp_052/sbatch_bertTfm.sh
sbatch experiments/exp_052/sbatch_enhancer.sh
sbatch experiments/exp_052/sbatch_qsel.sh
sbatch experiments/exp_052/sbatch_decoder.sh
squeue -u kouyou
```

- リプレイありのため **Objects365 の bind が必要**（sbatch に設定済み。exp_035 と同じ）。
- 同時実行 4 本制限のため 5 本目（と他実験の分）はキュー待ち（自動開始）。

### 投入後 10 分の確認

```bash
LOG=/home/kouyou/logs/result_exp052_swinNeck_<JOBID>.txt   # 他条件も同様
grep -m2 "t=1/3\|教師" $LOG
grep -m3 "Epoch(train)" $LOG
grep -m2 "loss_kd" $LOG
```

t=1 の初期 loss（本環境 1 step 実測・学生=教師=θ0・バッチ 6）:
loss 26.8〜28.7 / loss_kd 10.4〜13.1（λ=10 込み。全条件ほぼ同水準）。
loss_kd が数百のオーダーなら load_from の渡し忘れを疑う。

## 3. 再開

同じ sbatch を再投入（学習は epoch_20.pth、評価はディレクトリ有無でスキップ。
**json の無い評価ディレクトリは消してから**）。

## 4. 結果の転送（クラスタ → 本環境）

```bash
cd /workspace/kouyou/mmdetection
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp052/experiments/exp_052
LOCAL=experiments/exp_052
rsync -avh --progress \
  --include='*/' --include='*eval*/***' --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress \
  --include='*/' --include='*.log' --include='scalars.json' --include='config.py' \
  --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --include='*/' --include='epoch_20.pth' --exclude='*' "$REMOTE/" "$LOCAL/"
# 容量: 0.7 GB × 3 ドメイン × 5 条件 ≒ 10 GB
```
