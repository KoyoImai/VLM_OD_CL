# exp_047 実行手順（クラスタ・新規 clone から結果転送まで）

design: [[design]]。実験ノート: [[../../experiment_notes/note15]]。
**実行は design.md の承認後**（行動原理3。2026-08-18 承認済み）。

クラスタ側の作業ディレクトリは**新規 clone の `/home/kouyou/VLM_OD_CL_exp047`**
（design.md §5）。

## 0. 状態

| | 状態 |
|---|---|
| config 6 本・ドライバ・sbatch・検証 | 生成済み（実装本体は exp_038 で検証済みの `GroundingDINOLoRA`） |
| 実行前検証 | 静的 1 項目 OK（2026-08-18）。**--step（GPU）は exp_046 の完了後に実施**（本環境の GPU が満杯のため。相乗りは exp_046 を OOM で落とすリスクがある） |
| クラスタへの push / clone / 投入 | 未実施（--step 完了後） |

## 1. 本環境で検証してから push

```bash
cd /workspace/kouyou/mmdetection
python experiments/exp_047/check_exp047_setup.py                  # 静的（GPU 不要）
CUDA_VISIBLE_DEVICES=0 python experiments/exp_047/check_exp047_setup.py --step  # exp_046 完了後

git add experiments/exp_047/
git commit -m "add exp_047"
git push
git log --oneline -1
```

## 2. クラスタで新規 clone

```bash
ssh kouyou@192.168.170.100
git clone https://github.com/KoyoImai/VLM_OD_CL.git /home/kouyou/VLM_OD_CL_exp047
cd /home/kouyou/VLM_OD_CL_exp047
git log --oneline -1                       # §1 の commit と一致すること
ls experiments/exp_047/configs/ | wc -l    # 6 本
```

**転送が必要な `.pth` は無い**（θ0 から開始。θ0 は `/home/kouyou/ckpt` の bind。
バッファ不使用なので Objects365 の bind も不要）。

## 3. 投入（1 ジョブ）

```bash
cd /home/kouyou/VLM_OD_CL_exp047
sbatch experiments/exp_047/sbatch_lora.sh     # 約 30〜38 h
squeue -u kouyou
```

同時実行 4 本を超える分はキュー待ちで自動開始される。

## 4. 進捗の確認

```bash
squeue -u kouyou
tail -f /home/kouyou/logs/result_exp047_lora_<JOBID>.txt
ls /home/kouyou/VLM_OD_CL_exp047/experiments/exp_047/lora_theta/ 2>/dev/null
```

## 5. 結果の転送（クラスタ → 本環境）

ジョブ終了後、本環境で:

```bash
cd /workspace/kouyou/mmdetection
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp047/experiments/exp_047
LOCAL=experiments/exp_047

rsync -avh --progress \
  --include='*/' --include='*eval*/***' --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress \
  --include='*/' --include='*.log' --include='scalars.json' --include='config.py' \
  --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --include='*/' --include='theta_t*.pth' --exclude='*' "$REMOTE/" "$LOCAL/"
```

merged θ は 6 × 約 2 GB（再評価・分析はこれで足りる。work ckpt が必要なら別途）。

## 6. 完了後

結果は `experiments/exp_047/results/summary.md` に事実のみ記録する（行動原理8）。
判定材料は design.md §3（InfLoRA exp_045 との対比較＝A の扱いのみの差、
リプレイフリー exp_024＋exp_040 との対照＝低ランク制約の効果）。

## 7. 注意

- LoRA 設定（232 層・r=16・alpha=16）は exp_045 の InfLoRA と完全に同一。
  アダプタの学習率は RF100 枠の paramwise のまま（Swin/BERT 内 1e-5、
  encoder・text_feat_map 1e-4。design.md §2.2、2026-08-18 決定）。
- `*_work_dir` 配下は編集しない（禁止則）。
- 途中で止まっても同じコマンドで再開できる（θ_t があるドメインはスキップ）。
