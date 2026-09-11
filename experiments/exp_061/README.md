# exp_061 実行手順（バッファサイズ感度（Ours/ER・内訳 [4,1,1]）・クラスタ）

design: [[design]]。**実行は design.md の承認後（2026-09-11 承認済み）**。
push・clone・sbatch はユーザー実施。

## 0. 最重要: バッファ依存（クローンに必ず含める）

config は **exp_058/buffer/**（入れ子拡張バッファ、seed 0）を絶対パスで参照する。
このディレクトリは **git 未管理（46MB・56 ファイル）** のため、通常の clone には入らない。
クラスタ側のクローンに以下のいずれかで必ず配置すること:

- push 前に本環境で `git add experiments/exp_058/buffer && git commit` してから clone、または
- clone 後に本環境からバッファだけ rsync:
  ```bash
  rsync -avh experiments/exp_058/buffer/ \
    kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp061/experiments/exp_058/buffer/
  ```

## 1. 本環境で検証（実施済み）

```bash
python experiments/exp_061/check_exp061_setup.py   # config 本数・内訳・バッファ実在（ALL OK）
```
- config 48 本（手法 2 × サイズ 4 × ドメイン 6）生成済み。
- 実データ 1 step 検証済み（2026-09-11、GPU 1）: Ours=蒸留あり / ER=蒸留なし、
  内訳 [4,1,1] bs6、loss 有限・VRAM ≤11.4GB。

## 2. クラスタ（ユーザー実施）

```bash
# 本環境: push（exp_058/buffer を含めること。§0 参照）
git add experiments/exp_061 experiments/exp_058/buffer
git commit -m "add exp_061 (バッファサイズ感度（Ours/ER・内訳 [4,1,1]）)"
git push

# クラスタ: 専用クローン
git clone https://github.com/KoyoImai/VLM_OD_CL.git /home/kouyou/VLM_OD_CL_exp061
cd /home/kouyou/VLM_OD_CL_exp061
ls experiments/exp_058/buffer/*.json | wc -l     # 56 を確認（§0）
ls experiments/exp_061/configs/ | wc -l          # 48

# 投入（手法 × サイズ。同時実行 4 本制限。超過分はキュー待ち）
for m in ours er; do for c in b2 b3 b4 b5; do sbatch experiments/exp_061/sbatch_${m}_${c}.sh; done; done
squeue -u kouyou
```

- リプレイに Objects365、ZCOCO に MSCOCO、教師 θ_{t-1} に /home/kouyou/ckpt の bind が必要（sbatch 設定済み）。
- サイズ: b2..b5（base=1× は既存実測を再利用）。

## 3. 再開

同じ sbatch を再投入（epoch_20.pth のあるドメインは学習スキップ、json のある eval はスキップ。
json 無しの eval ディレクトリは消してから）。

## 4. 結果の回収・記録

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp061/experiments/exp_061
LOCAL=experiments/exp_061
rsync -avh --progress --include='*/' --include='eval_*/***' --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress --partial --include='*/' --include='*_work_dir/epoch_20.pth' --exclude='*' "$REMOTE/" "$LOCAL/"
```
回収後 `results/summary.md` に Ours・ER の base〜b5 推移を事実記録（行動原理8）。
