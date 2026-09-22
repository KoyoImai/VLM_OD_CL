# exp_062 実行手順（バッチ内訳感度（Ours/ER・内訳 [4,2,2]）・クラスタ）

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
    kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp062/experiments/exp_058/buffer/
  ```

## 1. 本環境で検証（実施済み）

```bash
python experiments/exp_062/check_exp062_setup.py   # config 本数・内訳・バッファ実在（ALL OK）
```
- config 60 本（手法 2 × サイズ 5 × ドメイン 6）生成済み。
- 実データ 1 step 検証済み（2026-09-11、GPU 1）: Ours=蒸留あり / ER=蒸留なし、
  内訳 [4,2,2] bs8、loss 有限・VRAM ≤14.2GB。

## 2. クラスタ（ユーザー実施）

```bash
# 本環境: push（exp_058/buffer を含めること。§0 参照）
git add experiments/exp_062 experiments/exp_058/buffer
git commit -m "add exp_062 (バッチ内訳感度（Ours/ER・内訳 [4,2,2]）)"
git push

# クラスタ: 専用クローン
git clone https://github.com/KoyoImai/VLM_OD_CL.git /home/kouyou/VLM_OD_CL_exp062
cd /home/kouyou/VLM_OD_CL_exp062
ls experiments/exp_058/buffer/*.json | wc -l     # 56 を確認（§0）
ls experiments/exp_062/configs/ | wc -l          # 60

# 投入（手法 × サイズ。同時実行 4 本制限。超過分はキュー待ち）
for m in ours er; do for c in base b2 b3 b4 b5; do sbatch experiments/exp_062/sbatch_${m}_${c}.sh; done; done
squeue -u kouyou
```

- リプレイに Objects365、ZCOCO に MSCOCO、教師 θ_{t-1} に /home/kouyou/ckpt の bind が必要（sbatch 設定済み）。
- サイズ: base..b5（全サイズ新規。内訳が [4,1,1] と異なるため base も学習）。

## 3. 再開

同じ sbatch を再投入（epoch_20.pth のあるドメインは学習スキップ、json のある eval はスキップ。
json 無しの eval ディレクトリは消してから）。

## 4. 結果の回収・記録

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp062/experiments/exp_062
LOCAL=experiments/exp_062
rsync -avh --progress --include='*/' --include='eval_*/***' --exclude='*' "$REMOTE/" "$LOCAL/"
rsync -avh --progress --partial --include='*/' --include='*_work_dir/epoch_20.pth' --exclude='*' "$REMOTE/" "$LOCAL/"
```
回収後 `results/summary.md` に Ours・ER の base〜b5 推移を事実記録（行動原理8）。

## 5. バグ修正と全面再実行（2026-09-22）

### 発生した問題
初回投入で `ours_b3` が electromagnetic（t=2）の学習開始直後に
`_assert_buffer_slice` の AssertionError（「バッファ位置の前提が崩れています」）で停止した。

**原因**: gen_configs.py の内訳上書きが `sampler.batch_size=8` のみを書き換え、
**DataLoader 本体の `batch_size` が 6 のまま**だった。サンプラーは [4,2,2] の 8 周期で
インデックス列を吐くが、DataLoader が 6 個ずつに切り直すため、**2 バッチ目以降で内訳が
ずれる**（例: 2 バッチ目 = [過 過 現 現 現 現] → 末尾 2 件が現在ドメインになり検査が発火）。

- 事前の 1-step 検証は初回バッチ（[現現現現参参]・末尾が o365 参照）しか見ないため
  原理的に検出不能だった。
- t=1（underwater）は上書き対象外のため正常。t=2 の electromagnetic が最初の発火点。
- **ER 側にはこの検査が無く、落ちずに「実効バッチ 6・内訳崩れ」で黙って学習が進む**
  ため、初回投入分の er_* の結果は無効。

### 修正（2026-09-22 承認済み）
1. gen_configs.py: t>=2 で `dl['batch_size'] = 8` も設定（sampler と本体を必ず一致させる）。
   60 本を再生成済み。
2. check_exp062_setup.py を強化: 静的検査に「本体と sampler の batch_size 一致」を追加し、
   **動的検査**（実 dataloader を構築して 5 バッチ分の位置内訳を検査:
   先頭 4=現在ドメイン / 末尾 2=現在ドメイン以外）を新設。
   ours_b3_electromagnetic / er_b3_electromagnetic / ours_base_documents で ALL OK（2026-09-22）。

### 再実行
**初回投入の結果（work_dir・eval）はクラスタ・ローカルとも全削除して全条件を再実行する**
（ユーザー決定 2026-09-22。t=1 は条件不変だが一律削除で仕切り直し）。
手順は §2 と同じ（push → クラスタ側 pull → sbatch 一括投入）。クラスタ側は
`experiments/exp_062/configs/` の差し替え（git pull）を忘れないこと。
