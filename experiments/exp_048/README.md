# exp_048 実行手順（クラスタ・専用クローン）

design: [[design]]。実験ノート: [[../../experiment_notes/note17]]。
**実行は design.md の承認後**（行動原理3。2026-08-20 承認済み）。

条件は 2 つ（詳細は design.md §2）:

| 条件 | バッチ | 蒸留対象 | 継承元 |
|---|---|---|---|
| kdEonly | 現在ドメインのみ 4/GPU（DefaultSampler） | 全 4 枚 | exp_024／exp_040 replayfree |
| kdEall | [現在4, 汎用1, 過去1]×6/GPU（t=1 は [現在4, 汎用2]） | 全 6 枚 | exp_023／exp_040 replay |

両条件とも蒸留E（KDGroundingDINO・L2・λ=10・教師 θ_{t-1}、t=1 は θ0）、6 ドメイン逐次。

## 0. 状態

| | 状態 |
|---|---|
| config 12 本・ドライバ 2 本・sbatch 2 本・検証 | 生成済み（本環境。実装本体は既存 KDGroundingDINO を無変更で使用） |
| 実行前検証（実データ 1 step・全サンプル蒸留の確認込み） | **4/4 OK**（2026-08-20、本環境 A100 1 枚） |
| クラスタへの push / clone / 投入 | 未実施 |

VRAM 実測: 実ランナー（2 GPU・DDP・教師込み）で kdEonly 11.2 GB / kdEall 19.4 GB
（logger の allocated 値）。A6000 Ada 48GB に収まる。

**ckpt 保持方針（2026-08-20 ユーザー決定）**: **全 epoch 保持**（継承既定
`max_keep_ckpts=-1` のまま。改修不要）。クラスタ側の消費は 2 条件合計 約 480 GB
（2GB × 20 epoch × 6 ドメイン × 2）。ホーム約 3 TB に対して他実験の出力と合算で
圧迫し得るため、投入前に `quota` で空きを確認しておくとよい。

## 1. 本環境で検証してから push

```bash
cd /workspace/kouyou/mmdetection
python experiments/exp_048/check_exp048_setup.py                  # 静的 2 項目
CUDA_VISIBLE_DEVICES=0 python experiments/exp_048/check_exp048_setup.py --step   # 全 4 項目

git add experiments/exp_048/
git commit -m "add exp_048"
git push
git log --oneline -1
```

## 2. クラスタで専用クローンを作る

実行中の他クローン（exp_043/044/045/047 など）には触れない（`PULL_WHILE_RUNNING.md`）。

```bash
ssh kouyou@192.168.170.100
git clone https://github.com/KoyoImai/VLM_OD_CL.git /home/kouyou/VLM_OD_CL_exp048
cd /home/kouyou/VLM_OD_CL_exp048
git log --oneline -1                        # §1 の commit と一致すること
ls experiments/exp_048/configs/ | wc -l     # 12 本
ls experiments/exp_023/buffer/*.odvg.json | wc -l   # 14（kdEall のバッファ定義。git 管理下）
```

**転送が必要な `.pth` は無い**（θ0 から開始。θ0 は `/home/kouyou/ckpt` の bind）。
データもすべて転送済みのものを使う（rf100_domain・o365v1_stage・bert-base-uncased は
`/home/kouyou/datasets`、MSCOCO は `/dataset01`）。

## 3. 投入（2 ジョブ並行）

`sbatch_*.sh` の `REPO` 既定は `/home/kouyou/VLM_OD_CL_exp048`。
別クローンから走らせる場合のみ `REPO=<パス> sbatch ...` とする。

```bash
cd /home/kouyou/VLM_OD_CL_exp048
sbatch experiments/exp_048/sbatch_kdEonly.sh    # 見積り 60〜90 h
sbatch experiments/exp_048/sbatch_kdEall.sh     # 見積り 80〜110 h
squeue -u kouyou
```

同時実行は 4 ジョブまで。他実験のジョブと合わせて超える分はキュー待ちで自動開始される
（投入は 8 本まで可）。中止は `scancel <JOBID>`。

### 投入後 10 分の確認

```bash
LOG1=/home/kouyou/logs/result_exp048_kdEonly_<JOBID>.txt
LOG2=/home/kouyou/logs/result_exp048_kdEall_<JOBID>.txt
grep -m2 "t=1/6\|教師" $LOG1 $LOG2      # θ0 が教師になっている
grep -m3 "Epoch(train)" $LOG1           # iteration が進んでいる
grep -m3 "loss_kd" $LOG1                # 蒸留損失が乗っている（t=1 から出る）
```

t=1 の初期 `loss_kd` は小さい（学生=教師=θ0 で dropout 差のみ。本環境の実ランナー
実測: kdEonly 16.5 → 339 iter で 4.5、kdEall 15.8 → 180 iter で 6.1。
exp_035 t=1 の最初のログ 8.3 とも整合）。数百のオーダーが出たら学生の重みが
θ0 になっていない（load_from の渡し忘れ等）を疑う。

## 4. 進捗の確認

```bash
squeue -u kouyou
tail -f /home/kouyou/logs/result_exp048_kdEonly_<JOBID>.txt

# どのドメインまで終わったか（epoch_20.pth の有無でスキップ判定）
ls /home/kouyou/VLM_OD_CL_exp048/experiments/exp_048/kdEonly_*_work_dir/epoch_20.pth 2>/dev/null
ls /home/kouyou/VLM_OD_CL_exp048/experiments/exp_048/eval_kdEonly/ 2>/dev/null
```

途中で止まっても同じ `sbatch` で再開できる（学習は epoch_20.pth、評価は
出力ディレクトリの有無でスキップ）。

## 5. 結果の転送（クラスタ → 本環境）

ジョブ終了後、本環境で:

```bash
cd /workspace/kouyou/mmdetection
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp048/experiments/exp_048
LOCAL=experiments/exp_048

rsync -avh --progress \
  --include='*/' --include='*eval*/***' --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress \
  --include='*/' --include='*.log' --include='scalars.json' --include='config.py' \
  --exclude='*' "$REMOTE/" "$LOCAL/"
# ckpt（各ドメインの last のみ回収。全 epoch 回収は容量と相談）
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --include='*/' --include='epoch_20.pth' --exclude='*' "$REMOTE/" "$LOCAL/"
```

容量目安（epoch_20 のみ）: 2 GB × 6 ドメイン × 2 条件 ≒ 24 GB。

## 6. 完了後

結果は `experiments/exp_048/results/summary.md` に事実のみ記録する（行動原理8）。
判定材料は design.md §3（各 t の全ドメイン mAP と ZCOCO、
対照は replayfree exp_024＋040 / 蒸留＋リプレイ exp_035＋040 / リプレイのみ exp_027＋040）。

## 7. 注意

- 教師 θ_{t-1} は**ドライバが --cfg-options model.teacher_ckpt= で毎回明示**する
  （t=1 は θ0 の実パス。load_from にも同じパスを渡す規約）。
- `num_buffer_per_batch = バッチサイズ` により蒸留が全サンプルに掛かる。バッファ位置の
  毎 iteration 検査（_assert_buffer_slice）はこの設定では自動で無効になるため、
  全サンプル適用は check_exp048_setup.py 項目 3 で検証済み（先頭サンプル差し替えで
  loss_kd が変化）。
- checkpointing は継承既定のまま（num_cp=6・Swin with_cp=True）。KD 系は θ を forward 外で
  使わないため、EWC で起きた DDP 衝突は該当しない（exp_035/043/046 で実績あり）。
- `*_work_dir` 配下は編集しない（禁止則）。
