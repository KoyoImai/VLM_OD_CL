# exp_053 実行手順（蒸留E × 学習可能モジュールアブレーション・後半 3 ドメイン継続）

design: [[design]]。**実行は design.md の承認後（2026-08-25 承認済み）。**
push・clone・sbatch はユーザー実施。

条件: exp_052 の続き（同じ 5 条件・凍結表）。枠は exp_040 kdE
（旧バッチ [現在4,汎用1,過去プール1]×6・λ=10・バッファ由来 2 枚・教師 θ_{t-1}・
20 epoch・seed 0・AMP 無し）。aerial → microscopic → documents（t=4..6）。
**t=4 の初期重み・教師 = exp_052 の同一条件の `videogames_work_dir/epoch_20.pth`。**
ckpt は epoch_20 のみ・optimizer 状態なし。

## 0. 状態

| | 状態 |
|---|---|
| config 15 本・ドライバ・sbatch 5 本・検証 | 生成済み |
| 実行前検証（check_exp053_setup.py --step） | **4/4 OK**（2026-08-25。exp_040 kdE との diff = paramwise と ckpt 方針のみ・lr>0 集合が境界と厳密一致・aerial 実データ 1 step で凍結不変/学習対象のみ更新） |
| クラスタ投入 | 未実施（**exp_052 の完了確認が前提**） |

## 1. 本環境で検証（実施済み・再現手順）

```bash
python experiments/exp_053/check_exp053_setup.py           # 静的 3 項目
CUDA_VISIBLE_DEVICES=0 python experiments/exp_053/check_exp053_setup.py --step   # 全 4 項目
```

## 2. クラスタ（ユーザー実施）

```bash
# 0) 前提確認: exp_052 の各条件が t=3 まで完了していること
for c in swinNeck bertTfm enhancer qsel decoder; do
  ls -la /home/kouyou/VLM_OD_CL_exp052/experiments/exp_052/${c}_videogames_work_dir/epoch_20.pth
done
# 5 本すべて存在してから投入する（無い条件は完了を待つ。条件ごとに個別投入で良い）

# 本環境: commit & push
git add experiments/exp_053/
git commit -m "add exp_053 (kdE trainable-module ablation, latter 3 domains)"
git push

# クラスタ: 専用クローン（実行中の他クローンに触れない。PULL_WHILE_RUNNING.md）
git clone https://github.com/KoyoImai/VLM_OD_CL.git /home/kouyou/VLM_OD_CL_exp053
cd /home/kouyou/VLM_OD_CL_exp053
git log --oneline -1
ls experiments/exp_053/configs/ | wc -l        # 15
sbatch experiments/exp_053/sbatch_swinNeck.sh  # 各 約 60〜80 h（--time=96:00:00）
sbatch experiments/exp_053/sbatch_bertTfm.sh
sbatch experiments/exp_053/sbatch_enhancer.sh
sbatch experiments/exp_053/sbatch_qsel.sh
sbatch experiments/exp_053/sbatch_decoder.sh
squeue -u kouyou
```

- exp_052 クローンの `experiments/exp_052` は sbatch が **read-only bind** して参照する
  （コピー不要。exp_052 クローンの場所が既定と違う場合は `REPO52=` で指定）。
- clone の認証エラーが出たら exp_052 README §2.1 と同じ対処
  （VS Code のターミナルから実行するか、`GIT_ASKPASS` 系を unset して PAT 入力）。
- リプレイに Objects365、ZCOCO に MSCOCO の bind が必要（sbatch に設定済み）。
- 同時実行 4 本制限のため 5 本目以降はキュー待ち（自動開始）。

### 投入後 10 分の確認

```bash
LOG=/home/kouyou/logs/result_exp053_swinNeck_<JOBID>.txt   # 他条件も同様
grep -m2 "t=4/6\|教師" $LOG    # 教師が exp_052 の videogames epoch_20 を指すこと
grep -m3 "Epoch(train)" $LOG
grep -m2 "loss_kd" $LOG
```

t=4 初期の目安（本環境 1 step 実測・教師=学生。実運用の教師は exp_052 t=3 なので
kd の絶対値は多少ずれる）: loss 45〜50 / loss_kd 13〜14（λ=10 込み）。
数百のオーダーなら load_from の渡し忘れを疑う。

## 3. 再開

同じ sbatch を再投入（学習は epoch_20.pth、評価はディレクトリ有無でスキップ。
**json の無い評価ディレクトリは消してから**）。

## 4. 結果の転送（クラスタ → 本環境）

```bash
cd /workspace/kouyou/mmdetection
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp053/experiments/exp_053
LOCAL=experiments/exp_053
rsync -avh --progress \
  --include='*/' --include='*eval*/***' --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress \
  --include='*/' --include='*.log' --include='scalars.json' --include='config.py' \
  --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --include='*/' --include='epoch_20.pth' --exclude='*' "$REMOTE/" "$LOCAL/"
# 容量: 0.7 GB × 3 ドメイン × 5 条件 ≒ 10 GB
```
