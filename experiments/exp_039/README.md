# exp_039 実行手順（クラスタ・新規 clone から結果転送まで）

design: [[design]]。**実行は design.md の承認後**（行動原理3）。

**前提**: クラスタで exp_036 が実行中のため、既存の `/home/kouyou/VLM_OD_CL` では
`git pull` できない。**別ディレクトリへ新規 clone** して、そちらから投入する。
exp_036 のジョブ・出力・データには一切触れない。

## 0. 状態

| | 状態 |
|---|---|
| config 12 本・ドライバ・sbatch・検証 | 生成済み（本環境） |
| 実行前検証 6 項目 | 6/6 OK（2026-08-12、本環境） |
| クラスタへの clone | 実施済み（`/home/kouyou/VLM_OD_CL_exp039`） |
| ZiRa | 投入済み |
| DitHub | 2026-08-12 に DDP エラーで停止 → 修正済み。**§10 の手順で再投入する** |

## 1. 本環境で検証してから push（clone 元を最新にする）

**実行前検証は本環境でしか行えない。**クラスタのマスターノードには mmdet の実行環境が無く
（python はコンテナ内にしかない）、計算ノードへは Slurm 経由でしか入れないため、
`check_exp039_setup.py` はクラスタでは動かない。**push する前に本環境で通しておく。**

```bash
cd /workspace/kouyou/mmdetection
python experiments/exp_039/check_exp039_setup.py           # 5/5 OK
python experiments/exp_039/check_exp039_setup.py --build   # 6/6 OK（GPU 1 枚）
```

クラスタが clone するのは remote なので、**先に push が要る**。

```bash
git status --short experiments/exp_039/
git add experiments/exp_039/ && git commit -m "add exp_039"
git push
git log --oneline -1
```

## 2. クラスタで新規 clone

`/home/kouyou/VLM_OD_CL` は exp_036 が使用中なので触らない。**別名で clone** する。

```bash
ssh kouyou@192.168.170.100
cd /home/kouyou
git clone https://github.com/KoyoImai/VLM_OD_CL.git VLM_OD_CL_exp039
cd VLM_OD_CL_exp039 && git log --oneline -1        # §1 の commit と一致すること
ls experiments/exp_039/configs/ | wc -l            # 12 本
```

clone 先を変えても、**Singularity が repo を `/workspace/kouyou/mmdetection` に bind する**ため
コンテナ内のパスは同じである。変わるのはホスト側の `REPO` だけで、投入時に環境変数で渡す。

`.pth` は `.gitignore` されているので clone には含まれない。exp_039 は θ0 から始まるため
それで問題ない（θ0 は `/home/kouyou/ckpt` にあり bind される）。

## 3. データの確認（転送済みのはず）

exp_027 / exp_028 と同じものを使う。**新規に転送するものは無い**。

```bash
ls /home/kouyou/datasets/rf100_domain/                    # 7 ドメイン
ls /home/kouyou/datasets/o365v1_stage/Objects365_v1/      # リプレイ参照元
ls /home/kouyou/datasets/bert-base-uncased/
ls /dataset01/MSCOCO/                                     # ZCOCO（読取専用）
ls /home/kouyou/ckpt/                                     # θ0
```

## 4. 投入

`REPO` に clone 先を渡す。**2 ジョブを並列に投入できる**（同時実行 4 ジョブ制限内、
exp_036 が 1 枠使っていても収まる）。

```bash
cd /home/kouyou/VLM_OD_CL_exp039

REPO=/home/kouyou/VLM_OD_CL_exp039 sbatch experiments/exp_039/sbatch_zira.sh
REPO=/home/kouyou/VLM_OD_CL_exp039 sbatch experiments/exp_039/sbatch_dithub.sh

squeue -u kouyou
```

各ジョブは「リプレイ無し → リプレイ有り」を直列に流す（design.md §5.3、約 54 h/ジョブ）。
条件を分けて 4 ジョブにしたい場合は次のようにする。

```bash
REPO=/home/kouyou/VLM_OD_CL_exp039 RUN_STAGE=replayfree sbatch experiments/exp_039/sbatch_zira.sh
REPO=/home/kouyou/VLM_OD_CL_exp039 RUN_STAGE=replay     sbatch experiments/exp_039/sbatch_zira.sh
```

**exp_036 のジョブを止めたり、`/home/kouyou/VLM_OD_CL` で作業したりしないこと。**

## 5. 進捗の確認

```bash
squeue -u kouyou
tail -f /home/kouyou/logs/result_exp039_zira_<JOBID>.txt

# どのドメインまで融合済みか
ls /home/kouyou/VLM_OD_CL_exp039/experiments/exp_039/*_merged/

# DitHub のフェーズ切替が epoch 10 で起きたか
grep -h "phase initialized" \
  /home/kouyou/VLM_OD_CL_exp039/experiments/exp_039/dithub_*_work_dir/*/*.log
```

期待する出力は各ドメイン 1 行ずつの
`>>> DitHub: phase initialized to WARMUP (iter=0, epoch=0), warmup_lora_a reinit=True <<<`。
**SPECIALIZATION が出ていたらフェーズ初期化のバグが再発している**（exp_037 で修正した箇所）。

ZiRa は Rep+ が `merge_hlrb.py` 側で走るので、融合後 ckpt の
`max|HLRB| = 1e-8` / `s = 0.1` を転送後に本環境で確認する（§7）。

## 6. 結果の転送（クラスタ → 本環境）

**転送は rsync のみで行う。git は使わない**（2026-07-26 の方針。他実験の `RETRIEVE.md` と同じ）。
**ジョブが終わってから行う**（実行中に rsync すると書き込み途中のファイルを掴む）。

```bash
squeue -u kouyou        # exp039 のジョブが消えていること
```

本環境で実行する。

```bash
cd /workspace/kouyou/mmdetection
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_exp039/experiments/exp_039
LOCAL=experiments/exp_039
```

### 6.1 評価結果（最優先・軽い）

```bash
rsync -avh --progress \
  --include='*/' --include='*eval*/***' --exclude='*' \
  "$REMOTE/" "$LOCAL/"
```

### 6.2 学習ログ（ckpt は除く）

```bash
rsync -avh --progress \
  --include='*/' --include='*.log' --include='scalars.json' --include='config.py' \
  --exclude='*' "$REMOTE/" "$LOCAL/"
```

### 6.3 チェックポイント（重い。必要なものだけ）

融合済み ckpt（`*_merged/`）だけで再評価も追加分析もできる。各ドメインの
`*_work_dir/` 内の生 ckpt は原則転送しない。

```bash
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --include='*/' --include='*_merged/***' --exclude='*' \
  "$REMOTE/" "$LOCAL/"
```

`--partial-dir` は必須。付けないと中断時に途中までのファイルが最終的な名前のまま残る。
容量は 4 条件 × 3 ドメイン × 約 1.95 GB ≒ **24 GB**。

### 6.4 転送できたかの確認

```bash
ssh kouyou@192.168.170.100 \
  'find /home/kouyou/VLM_OD_CL_exp039/experiments/exp_039 -name "*.pth" | wc -l'
find experiments/exp_039 -name '*.pth' | wc -l
```

クラスタ側の `*_merged/` の数と本環境の数が一致すること。rsync は差分のみを送るので、
足りなければ同じコマンドを再実行すればよい。

## 7. 転送後の確認（本環境）

```bash
# ZiRa: Rep+ が効いているか（HLRB=1e-8 / s=0.1 / LLRB が累積）
# DitHub: base が θ0 のまま / クラス別 A が和集合で増えているか
python - <<'PY'
import glob, torch
for f in sorted(glob.glob('experiments/exp_039/*_merged/*.pth')):
    sd = torch.load(f, map_location='cpu')['state_dict']
    hl = [k for k in sd if k.endswith('.hlrb.weight')]
    A  = [k for k in sd if '.per_class_lora_A.' in k]
    if hl:
        s = {round(float(sd[k].reshape(-1)[0]), 4) for k in sd if k.endswith('.scaling')}
        print(f, 'ZiRa  max|HLRB|=%.1e s=%s' % (max(float(sd[k].abs().max()) for k in hl), s))
    elif A:
        cls = {k.split('.per_class_lora_A.')[1] for k in A}
        print(f, f'DitHub  {len(cls)} クラス')
PY
```

## 8. 完了後

結果は `experiments/exp_039/results/summary.md` に事実のみ記録する（行動原理8）。
比較対象はリプレイ（exp_027）と蒸留（exp_028）の実測値、および exp_023 の論文準拠設定での
ZiRa / DitHub。

## 9. 注意

- **`/home/kouyou/VLM_OD_CL` には触らない**（exp_036 が使用中）。作業は
  `/home/kouyou/VLM_OD_CL_exp039` に閉じる。
- `*_work_dir` 配下は編集しない（禁止則）。
- `/local_cache/${SLURM_JOB_ID}` はジョブ終了で自動削除される。出力をそこに置かない。
- exp_039 が終わって転送も済んだら、`/home/kouyou/VLM_OD_CL_exp039` は削除してよい
  （削除前チェックは [[../CLUSTER_RECLONE]] の §1・§2 に準じる）。

## 10. 修正の取り込みと DitHub の再投入（2026-08-12）

### 10.1 何が起きたか

クラスタ 4GPU の DitHub 学習が次で落ちた。

```
RuntimeError: Expected to mark a variable ready only once
Parameter at index 1051 with name encoder.layers.5.ffn.layers.1.shared_lora_b
has been marked as ready twice.
```

原因は exp_039 の DitHub config に `encoder=dict(num_cp=0)` が無かったこと。事前学習
config の既定 `num_cp=6` により fairscale の `checkpoint_wrapper`（**再入版**）が encoder に
適用され、`DitHubGroundingDINO` が `encoder_cp=6` で掛ける非再入版と二重になった。再入版は
backward が二度走るため DDP の ready マークが二度立つ。継承元を exp_023 の DitHub base から
リプレイ・蒸留側へ変えた際に該当行が落ちていた。

修正は 3 点（commit `a23a7ba`）。

| | 内容 |
|---|---|
| config | `gen_configs.py` の `DITHUB_BODY` に `encoder=dict(num_cp=0)` を追加し 6 本を再生成 |
| ガード | `dithub_grounding_dino.py` の二重適用 assertion を実効化。fairscale は `module.forward` を `functools.partial` に差し替えて同じモジュールを返すため型名では検出できず、従来のガードは発火していなかった |
| 検証 | `check_exp039_setup.py` に `encoder.num_cp == 0` と `encoder_cp > 0` を追加 |

`encoder_cp=6`（非再入版）はそのまま使う（exp_023 と同条件。design.md §7 の caveat を参照）。
ZiRa は `encoder_cp` を使わないため影響を受けない。**走っているならそのまま継続してよい。**

### 10.2 本環境で push

```bash
cd /workspace/kouyou/mmdetection
git log --oneline -2          # 9c8ce62 / a23a7ba があること
git push
```

### 10.3 クラスタで pull

`VLM_OD_CL_exp039` は clone 済みなので、消さずに `git pull` でよい。変更したのは config と
コードだけで `.pth` は触らないため、既存の出力とは競合しない。

```bash
ssh kouyou@192.168.170.100
cd /home/kouyou/VLM_OD_CL_exp039
git status --short            # 変更が無いこと（あれば先に確認）
git pull
git log --oneline -1          # 9c8ce62 と一致すること
grep -n "num_cp" experiments/exp_039/configs/dithub_replay_underwater.py
```

`num_cp=0` が入っていること。

### 10.4 検証してから再投入

**`check_exp039_setup.py` はクラスタでは実行しない。**マスターノードには mmdet の実行環境が
無く（python はコンテナ内にしかない）、計算ノードへは Slurm 経由でしか入れない。
**実行前検証は §10.2 の push より前に本環境で済ませておく**（design.md §5.4）。

クラスタ側で行うのは、pull が反映されたかのシェルでの確認だけでよい。

```bash
cd /home/kouyou/VLM_OD_CL_exp039
grep -l "num_cp=0" experiments/exp_039/configs/dithub_*.py | wc -l    # 6
```

6 本すべてに入っていることを確認したら投入する。

```bash
REPO=/home/kouyou/VLM_OD_CL_exp039 sbatch experiments/exp_039/sbatch_dithub.sh
squeue -u kouyou
```

失敗した DitHub の `*_work_dir` は融合まで到達していないので、ドライバの再開判定
（融合済み ckpt の有無）ではスキップされない。t=1 から走り直す。

### 10.5 同種の失敗を早く捕まえる

DDP 特有の問題は単一 GPU の build 検証では出ない。投入後 10 分で最初の
`Epoch(train)` 行が出ているかを確認する。

```bash
grep -m3 "Epoch(train)" /home/kouyou/logs/result_exp039_dithub_<JOBID>.txt
```

出ていなければ `error_exp039_dithub_<JOBID>.txt` を見る。
