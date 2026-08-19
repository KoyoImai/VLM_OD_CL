# exp_045 実行手順（クラスタ・新規 clone から結果転送まで）

design: [[design]]。実験ノート: [[../../experiment_notes/note15]]。
**実行は design.md の承認後**（行動原理3。2026-08-16 承認済み）。

クラスタ側の作業ディレクトリは**新規 clone の `/home/kouyou/VLM_OD_CL_exp045`**
（design.md §5）。

## 0. 状態

| | 状態 |
|---|---|
| config 12 本・ドライバ 2 本・sbatch 2 本・検証 | 生成済み（本環境。実装本体は exp_042 で検証済みの projects/ewc_cl・projects/lora_cl） |
| 実行前検証 5 項目（ODVG 実データの一連込み） | 5/5 OK（2026-08-16、本環境） |
| クラスタへの push / clone / 投入 | 未実施 |

## 1. 本環境で検証してから push

```bash
cd /workspace/kouyou/mmdetection
python experiments/exp_045/check_exp045_setup.py                  # 静的 2 項目
CUDA_VISIBLE_DEVICES=0 python experiments/exp_045/check_exp045_setup.py --build --step

git add experiments/exp_045/
git commit -m "add exp_045"
git push
git log --oneline -1
```

## 2. クラスタで新規 clone

```bash
ssh kouyou@192.168.170.100
git clone https://github.com/KoyoImai/VLM_OD_CL.git /home/kouyou/VLM_OD_CL_exp045
cd /home/kouyou/VLM_OD_CL_exp045
git log --oneline -1                        # §1 の commit と一致すること
ls experiments/exp_045/configs/ | wc -l     # 12 本
```

**転送が必要な `.pth` は無い**（θ0 から開始。θ0 は `/home/kouyou/ckpt` の bind。
バッファ不使用なので Objects365 の bind も不要）。

## 3. 投入（2 ジョブ並行）

```bash
cd /home/kouyou/VLM_OD_CL_exp045
sbatch experiments/exp_045/sbatch_ewc.sh       # 約 40〜45 h（λ=1000 はスクリプト既定）
sbatch experiments/exp_045/sbatch_inflora.sh   # 約 30〜40 h
squeue -u kouyou
```

exp_043（2）・exp_044（1）と合わせて 5 本になるため、**1 本は同時実行 4 本制限で
キュー待ち**になる（投入は 8 本まで可。空き次第自動で開始）。

## 3.5 障害と修正・EWC の再開手順（2026-08-19）

EWC が t=2（electromagnetic）開始直後に
`RuntimeError: Expected to mark a variable ready only once` で停止した。
原因は「EWC ペナルティ（θ の forward 外使用）× fairscale **再入型** checkpointing
（encoder num_cp=6）× DDP」の組み合わせで、t=2 でペナルティが活性化した時点で
勾配が 2 経路になり顕在化した（t=1 はペナルティ不活性のため通る。最小再現で
cp あり=エラー / cp なし=正常を実証済み）。

**修正**: EWC config に `encoder=dict(num_cp=0)` を追加（2026-08-19、ユーザー決定）。
InfLoRA は影響なし（ペナルティを持たない）で num_cp=6 のまま。
checkpointing の有無による数値条件の但し書きは design.md §6。

**t=1 の成果物はすべて有効**なので、underwater はやり直さない。エラーが t=2 の学習中に
出たこと自体が「t=1 の学習 → Fisher（state_t1.pth）→ 評価」まで完了している証拠である
（ドライバはこの順で t=2 に進む）。ドライバは `ewc_states/state_t{t}.pth` の有無で
タスクをスキップするため、再投入だけで t=2 から再開される。

### 再開手順

**(1) 本環境**: 修正済み config を push する。

```bash
cd /workspace/kouyou/mmdetection
python experiments/exp_045/check_exp045_setup.py     # 2/2 OK を確認（2026-08-19 実施済み）
git add experiments/exp_045/
git commit -m "fix exp_045 ewc: encoder num_cp=0 (reentrant cp x EWC penalty x DDP)"
git push
git log --oneline -1
```

**(2) クラスタ**: pull して EWC ジョブだけ再投入する。

```bash
ssh kouyou@192.168.170.100
cd /home/kouyou/VLM_OD_CL_exp045
squeue -u kouyou          # inflora ジョブが同クローンで実行中でも、変更は ewc 側のみなので pull は安全
git pull && git log --oneline -1     # (1) の commit と一致すること
ls experiments/exp_045/ewc_states/   # state_t1.pth があること（= t=2 から再開される）
sbatch experiments/exp_045/sbatch_ewc.sh
squeue -u kouyou
```

**(3) 再開後 10 分の確認**: t=1 がスキップされ、t=2 の学習が始まり、ペナルティが
乗っていること。

```bash
grep -m2 "スキップ\|t=2/6" /home/kouyou/logs/result_exp045_ewc_<新JOBID>.txt
grep -m3 "Epoch(train)" /home/kouyou/logs/result_exp045_ewc_<新JOBID>.txt
grep -m3 "loss_ewc" /home/kouyou/logs/result_exp045_ewc_<新JOBID>.txt   # t>=2 で必ず出る
```

**記録**: この再開により **t=1 は num_cp=6、t=2 以降は num_cp=0** という混在になる
（2026-08-19 判断: t=1 はペナルティ不活性＝素のフル FT と同条件であり、num_cp=6 は
むしろ replayfree と揃っている。差は浮動小数点レベルのため取り直しはしない）。

## 4. 進捗の確認

```bash
squeue -u kouyou
tail -f /home/kouyou/logs/result_exp045_ewc_<JOBID>.txt
tail -f /home/kouyou/logs/result_exp045_inflora_<JOBID>.txt

# どのドメインまで終わったか
ls /home/kouyou/VLM_OD_CL_exp045/experiments/exp_045/ewc_states/ 2>/dev/null
ls /home/kouyou/VLM_OD_CL_exp045/experiments/exp_045/inflora_theta/ 2>/dev/null

# EWC: t>=2 で loss_ewc がログに乗る / InfLoRA: 設計・メモリ更新の進行
grep -o "loss_ewc: [0-9.]*" /home/kouyou/logs/result_exp045_ewc_<JOBID>.txt | tail -3
grep -E "\[InfLoRA\]" /home/kouyou/logs/result_exp045_inflora_<JOBID>.txt | tail -5
```

VRAM 実測（本環境、batch 4/GPU）: EWC 5.1 GB / InfLoRA 5.7 GB（A6000 48GB に余裕）。
参考: 8 枚収集でのテキスト側ランク不足は 5/232 層（本走は 1,000 枚でさらに減る見込み。
検証ログ exp045_check）。

## 5. 結果の転送（クラスタ → 本環境）

ジョブ終了後、本環境で:

```bash
cd /workspace/kouyou/mmdetection
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp045/experiments/exp_045
LOCAL=experiments/exp_045

rsync -avh --progress \
  --include='*/' --include='*eval*/***' --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress \
  --include='*/' --include='*.log' --include='scalars.json' --include='config.py' \
  --exclude='*' "$REMOTE/" "$LOCAL/"
# ckpt / 状態 / メモリ（EWC は epoch_20 と状態、InfLoRA は merged θ とメモリで再現可能）
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --include='*/' --include='epoch_20.pth' --include='state_t*.pth' \
  --include='theta_t*.pth' --include='memory_t*.pth' --include='design_t*.pth' \
  --exclude='*' "$REMOTE/" "$LOCAL/"
```

容量目安: EWC ckpt 6×2.0 GB＋状態 6×1.4 GB、InfLoRA work ckpt 6×2.0 GB＋
merged 6×2.0 GB ≒ **約 45 GB**。

## 6. 完了後

結果は `experiments/exp_045/results/summary.md` に事実のみ記録する（行動原理8）。
判定材料は design.md §3（各 t の全ドメイン mAP と ZCOCO、リプレイフリー
exp_024＋exp_040 との中心対照、バッファ使用系列との横並び、exp_042 との手法内対照）。

## 7. 注意

- EWC の λ（=1000）と状態、InfLoRA の design_path は**ドライバが --cfg-options で
  毎回明示**する（誤記は model に届かない — 検証項目 2 で確認済み）。
- InfLoRA の DualGPM は **T=6**（ODinW の 13 と違う。ドライバ既定に設定済み）。
- `*_work_dir` 配下は編集しない（禁止則）。
- 途中で止まっても同じコマンドで再開できる（EWC は状態、InfLoRA は θ_t の有無で
  スキップ判定）。
