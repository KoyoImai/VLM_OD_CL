# exp_036（λ = 2.0 / 3.0 / 5.0）の結果をクラスタから本環境へ持ち帰る手順

作成: 2026-08-06

方向: クラスタ `kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments/`
　　→ 本環境 `/workspace/kouyou/mmdetection/experiments/`

**結果の転送は rsync のみで行う。git は使わない**（2026-07-26 の方針）。
汎用手順は [RETRIEVE_FROM_CLUSTER.md](../RETRIEVE_FROM_CLUSTER.md)。

---

## 0. 事前確認

### 0.1 転送元のクローンを確かめる

```bash
ssh kouyou@192.168.170.100 'ls -d /home/kouyou/VLM_OD_CL*/experiments/exp_036/*_work_dir 2>/dev/null | sed "s|/experiments.*||" | sort -u'
```

出力が `/home/kouyou/VLM_OD_CL_new` なら、以下の `REMOTE` をそちらに読み替える。

### 0.2 ジョブが終わっているか（**最優先**）

```bash
ssh kouyou@192.168.170.100 'squeue -u kouyou'
```

exp_036 のジョブ名は次の6つ。

```
exp036a_kdAB_condA_l2w20    exp036b_kdDE_condA_l2w20
exp036c_kdAB_condA_l2w30    exp036d_kdDE_condA_l2w30
exp036e_kdAB_condA_l2w50    exp036f_kdDE_condA_l2w50
```

途中経過だけ見たい場合は転送せずログを読む。

```bash
ssh kouyou@192.168.170.100 'tail -30 /home/kouyou/logs/result_exp036a_kdAB_condA_l2w20_*.txt'
```

### 0.3 λ が効いていたかをログで先に確認する（**exp_036 の存在理由**）

exp_036 は、exp_033 で λ が実効 1.0 になっていた不具合を直した再実行である（design §0）。
**ジョブログの冒頭に実効 λ の行が出ているかを、転送前に必ず確認する。**

```bash
ssh kouyou@192.168.170.100 'grep -h "実効 λ" /home/kouyou/logs/result_exp036*_*.txt'
```

期待する出力は次の6行。

```
[exp_036_a] 実効 λ = 2.0（config と一致）
[exp_036_b] 実効 λ = 2.0（config と一致）
[exp_036_c] 実効 λ = 3.0（config と一致）
[exp_036_d] 実効 λ = 3.0（config と一致）
[exp_036_e] 実効 λ = 5.0（config と一致）
[exp_036_f] 実効 λ = 5.0（config と一致）
```

出ていないジョブの結果は使わない。

### 0.4 容量

```bash
df -h /workspace/kouyou
ssh kouyou@192.168.170.100 'du -sh /home/kouyou/VLM_OD_CL/experiments/exp_036'
```

見積りは **約 1.4 TB**（12条件 × 3ドメイン × 20 epoch × 2.0GB）。
本環境 `/workspace/kouyou` の空きは 1.8 TB（2026-08-06 時点）で**余裕が小さい**。
足りない場合は λ ごとに分けて取り、解析が済んだものから判断する（§1.1）。

---

## 1. 転送

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments

rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --exclude='/configs/' --exclude='/*.sh' --exclude='/*.py' --exclude='/*.md' \
  "$REMOTE/exp_036/" "$LOCAL/exp_036/"
```

`--exclude` の先頭の `/` は転送元のルート（＝`exp_036/` 直下）を指す。**直下のコードだけを外し**、
`*_work_dir/` 配下の `.py`（実行時に保存された config の複製）は残す。コードは本環境が正本。

`--partial-dir=.rsync-partial` は必須。付けないと中断時に途中までのファイルが最終的な名前のまま
残り、**完全なファイルに見えてしまう**（2026-07-27 の実例）。

### 1.1 λ ごとに分けて転送する（容量が厳しい場合）

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments

# λ=2.0 だけ（約 470 GB）
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --include='*/' --include='*l2w20*/***' --exclude='*' \
  "$REMOTE/exp_036/" "$LOCAL/exp_036/"
```

`l2w20` を `l2w30`（λ=3.0）/ `l2w50`（λ=5.0）に変えて繰り返す。

### 1.2 評価結果だけ先に取る（軽量・十数 MB）

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments

rsync -avh --progress \
  --include='*/' --include='*eval_after_*/***' --exclude='*' \
  "$REMOTE/exp_036/" "$LOCAL/exp_036/"
```

### 1.3 学習ログ（loss_kd の推移）だけ取る

λ が実際に効いているかは `loss_kd` の大きさで確かめられる（λ=5.0 なら λ=1.0 の5倍）。

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments

rsync -avh --progress \
  --include='*/' --include='*.log' --include='scalars.json' --include='config.py' \
  --exclude='*' "$REMOTE/exp_036/" "$LOCAL/exp_036/"
```

### 1.4 長時間になるので切断に強い形で

```bash
cd /workspace/kouyou/mmdetection
nohup bash -c '
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --exclude="/configs/" --exclude="/*.sh" --exclude="/*.py" --exclude="/*.md" \
  kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments/exp_036/ \
  /workspace/kouyou/mmdetection/experiments/exp_036/
' > experiments/retrieve_exp036.log 2>&1 &

tail -f experiments/retrieve_exp036.log
```

**中断したら同じコマンドを再実行すればよい。**

---

## 2. 転送後の確認

### 2.1 個数の照合

```bash
echo -n "クラスタ ckpt: "; ssh kouyou@192.168.170.100 \
  "find /home/kouyou/VLM_OD_CL/experiments/exp_036 -name '*.pth' | wc -l"
echo -n "本環境   ckpt: "; find /workspace/kouyou/mmdetection/experiments/exp_036 -name '*.pth' | wc -l
du -sh /workspace/kouyou/mmdetection/experiments/exp_036
```

全条件が完走していれば ckpt は **720 個**（12条件 × 3ドメイン × 20 epoch）。

### 2.2 実効 λ が config どおりだったか（**必ず確認する**）

exp_033 と同じ取り違えが起きていないかを、work_dir に残る実効 config で確かめる。

```bash
python3 - <<'PY'
import glob, re
bad = []
for f in sorted(glob.glob('/workspace/kouyou/mmdetection/experiments/exp_036/*_work_dir/*/vis_data/config.py')):
    m = re.search(r"loss_weight=([\d.]+),\n *num_buffer", open(f).read())
    name = f.split('/')[-4].replace('_work_dir', '')
    want = {'l2w20': '2.0', 'l2w30': '3.0', 'l2w50': '5.0'}[
        [k for k in ('l2w20', 'l2w30', 'l2w50') if k in name][0]]
    got = m.group(1) if m else '?'
    ok = (got == want)
    if not ok:
        bad.append(name)
    print(f'[{"OK" if ok else "NG"}] {name:40s} loss_weight = {got}（期待 {want}）')
print('\nNG:', bad if bad else 'なし')
PY
```

**NG が出た条件は使わない。**

### 2.3 loss_kd が λ に比例しているか

exp_028（λ=1.0）と比べて、λ=2.0 なら約2倍、λ=5.0 なら約5倍になっていること。

```bash
python3 - <<'PY'
import glob, re, statistics
def ep20(pat):
    logs = sorted(glob.glob(pat))
    if not logs: return None
    v = [float(m.group(1)) for l in open(logs[-1], errors='ignore')
         if 'Epoch(train) [20]' in l for m in [re.search(r'loss_kd: ([\d.e+-]+)', l)] if m]
    return statistics.mean(v) if v else None
base = '/workspace/kouyou/mmdetection/experiments'
for kd in 'ABDE':
    r = ep20(f'{base}/exp_028/kd{kd}_condA_l2_underwater_work_dir/*/*.log')
    row = [f'λ=1.0: {r:.4f}' if r else 'λ=1.0: —']
    for w, lam in (('20', 2), ('30', 3), ('50', 5)):
        v = ep20(f'{base}/exp_036/kd{kd}_condA_l2w{w}_underwater_work_dir/*/*.log')
        row.append(f'λ={lam}.0: {v:.4f}（{v/r:.2f}倍）' if v and r else f'λ={lam}.0: —')
    print(f'蒸留{kd}  ' + '  '.join(row))
PY
```

### 2.4 途中ファイルが残っていないか

```bash
find /workspace/kouyou/mmdetection/experiments/exp_036 \
  \( -name '.*.pth.*' -o -name '.rsync-partial' -o -name '*.pth.*' \) | head
```

何か出たら転送が完了していない。**この状態で `git add` しない。**

---

## 3. 注意

- **`*_work_dir` 配下は編集しない**（禁止則）。
- 転送方向は常に **クラスタ → 本環境**。コードを送るのは `git push` / `git pull`。
- 蒸留B は λ=5.0 で検出損失の 6.7 倍になる。ckpt が 20 個揃っていない条件があれば、
  まずジョブログの末尾を読む（破綻した可能性がある）。
- exp_033 の結果は**別物として残す**。exp_036 で上書きしない。
