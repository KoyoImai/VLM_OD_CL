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

## 3.6 障害と修正・EWC の再開手順（2026-08-20）

§3.5 の修正後に再投入した EWC が、**再び t=2（electromagnetic）の学習開始直後**に
同じ `RuntimeError: Expected to mark a variable ready only once` で停止した。
今度は落ちたパラメータが backbone 側である
（`Parameter at index 169 with name backbone.stages.3.blocks.1.ffn.layers.1.bias`）。

**原因**: §3.5 の `encoder=dict(num_cp=0)` は **encoder の fairscale 再入型 cp を切るだけ**で、
**Swin backbone の `with_cp=True`**（事前学習 config `grounding_dino_swin-t_pretrain_obj365.py`
44 行目。`mmdet/models/backbones/swin.py:375` が `cp.checkpoint(...)` を `use_reentrant`
未指定＝再入型で呼ぶ）はそのまま残っていた。EWC ペナルティが θ に 2 経路目の勾配を
作るため、再入型 checkpoint の入れ子 backward と DDP の組で同じ機序が backbone 側で
顕在化する（t=1 はペナルティ不活性のため通る）。

**検証（本環境、A100 40GB × 2、batch 4/GPU、λ=10³、デバッグ用 Fisher 状態）**:
クラスタと同一のパラメータ・同一 index でエラーを再現したうえで、2 案を実測した。

| 案 | 結果 | allocated ピーク | iteration 時間 |
|---|---|---:|---:|
| `backbone.with_cp=False` | **iter 13 で OOM**（2 回再現。確保総量 39.42 GiB） | 28,119 MiB | 1.60 s |
| `static_graph=True`（cp 維持） | **303 iteration 完走**（`loss_ewc` 全 iteration に出現） | 31,226 MiB | 1.49 s |

**修正**: EWC config に
`model_wrapper_cfg = dict(type='MMDistributedDataParallel', static_graph=True)` を追加
（2026-08-20、ユーザー決定）。再入型 checkpointing を DDP 下で使うための PyTorch 公式の
構成であり、学習の計算そのもの（cp あり）は t=1・対照（replayfree / InfLoRA / LoRA）と
同一のまま、変わるのは DDP の勾配集約の記帳だけである。`encoder=dict(num_cp=0)` は
§3.5 のまま残す。InfLoRA は影響なし（ペナルティを持たない）。但し書きは design.md §6。

**t=1 の成果物は §3.5 と同様にすべて有効**で、`ewc_states/state_t1.pth` があるため
再投入だけで t=2 から再開される（ドライバは状態ファイルの有無でスキップ判定）。

**メモリの注意**: §4 に記した 5.1 GB はペナルティ不活性の 1 step 検証値である。
ペナルティ活性時の実測は **31.2 GB（batch 4/GPU）**で、A6000 Ada 48GB には収まるが
余裕は約 1.5 倍しかない。他ジョブと GPU を共有しない前提で投入すること。

### 再開手順

**(1) 本環境**: 修正済み config を検証して push する（実施済みの検証結果を併記）。

```bash
cd /workspace/kouyou/mmdetection
python experiments/exp_045/gen_configs.py            # ewc_*.py 6 本を再生成
python experiments/exp_045/check_exp045_setup.py     # 2/2 OK（2026-08-20 実施済み）

git add experiments/exp_045/
git commit -m "fix exp_045 ewc: DDP static_graph=True (reentrant cp in Swin backbone x EWC penalty)"
git push
git log --oneline -1
```

**(2) クラスタ**: 専用クローンで pull して EWC ジョブだけ再投入する。

```bash
ssh kouyou@192.168.170.100
cd /home/kouyou/VLM_OD_CL_exp045
squeue -u kouyou                      # 走っているジョブの確認（変更は ewc 側のみなので pull は安全）
git pull && git log --oneline -1      # (1) の commit と一致すること
python3 - <<'EOF'
import re
s = open('experiments/exp_045/configs/ewc_electromagnetic.py').read()
print('static_graph:', 'static_graph=True' in s.replace(' ', ''))
print('num_cp=0    :', 'num_cp=0' in s.replace(' ', ''))
EOF
ls experiments/exp_045/ewc_states/    # state_t1.pth があること（= t=2 から再開される）
sbatch experiments/exp_045/sbatch_ewc.sh
squeue -u kouyou
```

`sbatch_ewc.sh` は `LAM=1000 GPUS=4` でドライバを呼ぶ器で、パーティションは
`a6000_ada`（`--exclude=node03`。node03 は `/local_cache` が無く mkdir で落ちる）、
rf100_domain を `/local_cache/$SLURM_JOB_ID` にステージングし、MSCOCO・θ0・
bert-base-uncased を bind する。ジョブ側は何も変更していない。

**(3) 再開後 10 分の確認**: t=1 がスキップされ、t=2 の学習が iteration を刻み、
ペナルティが乗っていること。§3.5 と同じく次の 3 点を見る。

```bash
LOG=/home/kouyou/logs/result_exp045_ewc_<新JOBID>.txt
grep -m2 "スキップ\|t=2/6" $LOG
grep -m3 "Epoch(train)" $LOG          # iteration が進んでいること（前回はここで停止した）
grep -m3 "loss_ewc" $LOG              # t>=2 で必ず出る（ペナルティが有効）
grep -c "marked as ready twice" $LOG  # 0 であること
```

エラーが出るとすれば従来どおり t=2 の最初の backward なので、**この確認は
再投入から 10 分以内に行えば十分**である（前回・前々回とも学習開始直後に停止した）。

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
