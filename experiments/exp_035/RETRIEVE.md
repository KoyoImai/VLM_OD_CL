# exp_035（λ = 10.0）の結果をクラスタから本環境へ持ち帰る手順

作成: 2026-08-06

方向: クラスタ `kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments/`
　　→ 本環境 `/workspace/kouyou/mmdetection/experiments/`

**結果の転送は rsync のみで行う。git は使わない**（2026-07-26 の方針）。
汎用手順は [RETRIEVE_FROM_CLUSTER.md](../RETRIEVE_FROM_CLUSTER.md)。

---

## 0. 事前確認

### 0.1 転送元のクローンを確かめる

投入したクローンによってパスが変わる（exp_033 は `VLM_OD_CL_new` から投入した）。

```bash
ssh kouyou@192.168.170.100 'ls -d /home/kouyou/VLM_OD_CL*/experiments/exp_035/*_work_dir 2>/dev/null | sed "s|/experiments.*||" | sort -u'
```

出力が `/home/kouyou/VLM_OD_CL_new` なら、以下の `REMOTE` をそちらに読み替える。

### 0.2 ジョブが終わっているか（**最優先**）

**実行中に rsync すると書き込み途中のファイルを掴む。**

```bash
ssh kouyou@192.168.170.100 'squeue -u kouyou'
```

exp_035 のジョブ名は次の2つ。

```
exp035a_kdAB_condA_l2w100     exp035b_kdDE_condA_l2w100
```

途中経過だけ見たい場合は転送せずログを読む。

```bash
ssh kouyou@192.168.170.100 'tail -30 /home/kouyou/logs/result_exp035a_kdAB_condA_l2w100_*.txt'
```

### 0.3 λ が効いていたかをログで先に確認する（**exp_035 固有・重要**）

exp_033 は λ が config から上書きされて実効 1.0 で学習されていた（design §0）。
exp_035 は `train_val_{a,b}.sh` で `export LAMBDA=10.0` を明示し、起動時にガードを入れてある。
**ジョブログの冒頭にこの行が出ているかを、転送前に確認する。**

```bash
ssh kouyou@192.168.170.100 'grep -h "実効 λ" /home/kouyou/logs/result_exp035*_*.txt'
```

期待する出力は `[exp_035] 実効 λ = 10.0（config と一致）`。出ていない場合は結果を使わない。

### 0.4 容量

```bash
df -h /workspace/kouyou
ssh kouyou@192.168.170.100 'du -sh /home/kouyou/VLM_OD_CL/experiments/exp_035'
```

見積りは **約 480 GB**（4条件 × 3ドメイン × 20 epoch × 2.0GB）。
本環境 `/workspace/kouyou` の空きは 1.8 TB（2026-08-06 時点）。

---

## 1. 転送

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments

rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --exclude='/configs/' --exclude='/*.sh' --exclude='/*.py' --exclude='/*.md' \
  "$REMOTE/exp_035/" "$LOCAL/exp_035/"
```

`--exclude` の先頭の `/` は転送元のルート（＝`exp_035/` 直下）を指す。**直下のコードだけを外し**、
`*_work_dir/` 配下の `.py`（実行時に保存された config の複製）は残す。コードは本環境が正本。

`--partial-dir=.rsync-partial` は必須。付けないと中断時に途中までのファイルが最終的な名前のまま
残り、**完全なファイルに見えてしまう**（2026-07-27 に 1.9GB の一時ファイルを git が巻き込んで
push が失敗した実例がある）。

### 1.1 評価結果だけ先に取る（軽量・数 MB）

数値だけ見たい場合はこれで足りる。

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments

rsync -avh --progress \
  --include='*/' --include='*eval_after_*/***' --exclude='*' \
  "$REMOTE/exp_035/" "$LOCAL/exp_035/"
```

### 1.2 学習ログ（loss の推移）だけ取る

λ が効いているかを `loss_kd` で確かめたい場合はこれで足りる。

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments

rsync -avh --progress \
  --include='*/' --include='*.log' --include='scalars.json' --include='config.py' \
  --exclude='*' "$REMOTE/exp_035/" "$LOCAL/exp_035/"
```

### 1.3 長時間になるので切断に強い形で

```bash
cd /workspace/kouyou/mmdetection
nohup bash -c '
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --exclude="/configs/" --exclude="/*.sh" --exclude="/*.py" --exclude="/*.md" \
  kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments/exp_035/ \
  /workspace/kouyou/mmdetection/experiments/exp_035/
' > experiments/retrieve_exp035.log 2>&1 &

tail -f experiments/retrieve_exp035.log
```

**中断したら同じコマンドを再実行すればよい。** 転送済みはスキップされ、途中のファイルは
`.rsync-partial` から続きを再開する。

---

## 2. 転送後の確認

### 2.1 個数の照合

```bash
echo -n "クラスタ ckpt: "; ssh kouyou@192.168.170.100 \
  "find /home/kouyou/VLM_OD_CL/experiments/exp_035 -name '*.pth' | wc -l"
echo -n "本環境   ckpt: "; find /workspace/kouyou/mmdetection/experiments/exp_035 -name '*.pth' | wc -l
du -sh /workspace/kouyou/mmdetection/experiments/exp_035
```

全条件が完走していれば ckpt は **240 個**（4条件 × 3ドメイン × 20 epoch）。

### 2.2 実効 λ が 10.0 だったか（**必ず確認する**）

work_dir に残る実効 config を読む。exp_033 の再発を防ぐための確認である。

```bash
python3 - <<'PY'
import glob, re
for f in sorted(glob.glob('/workspace/kouyou/mmdetection/experiments/exp_035/*_work_dir/*/vis_data/config.py')):
    m = re.search(r"loss_weight=([\d.]+),\n *num_buffer", open(f).read())
    name = f.split('/')[-4].replace('_work_dir', '')
    print(f'{name:36s} loss_weight = {m.group(1) if m else "?"}')
PY
```

**すべて 10.0 であること。** 1.0 が出たら λ が効いていないので、その条件は使わない。

### 2.3 途中ファイルが残っていないか

```bash
find /workspace/kouyou/mmdetection/experiments/exp_035 \
  \( -name '.*.pth.*' -o -name '.rsync-partial' -o -name '*.pth.*' \) | head
```

何か出たら転送が完了していない。**この状態で `git add` しない。**

### 2.4 評価値が読めるか

```bash
grep -h "coco/bbox_mAP:" \
  /workspace/kouyou/mmdetection/experiments/exp_035/kdA_condA_l2w100_eval_after_videogames/*/*/*.log \
  | tail -4
```

---

## 3. 注意

- **`*_work_dir` 配下は編集しない**（禁止則）。rsync で上書きするのは構わないが、手で直さない。
- 転送方向は常に **クラスタ → 本環境**。コードを送るのは `git push` / `git pull`。
- 蒸留B（λ=10.0 で検出損失の 13.5 倍）は学習が破綻する可能性がある。ckpt が 20 個
  揃っていない条件があれば、まずジョブログの末尾を読む。
