# exp_033 の結果をクラスタから本環境へ持ち帰る手順

作成: 2026-08-02

**転送元は別クローン `/home/kouyou/VLM_OD_CL_new` である。** exp_033 は実行中の exp_029 を
避けるため2つ目のクローンから投入した（[SETUP_SECOND_CLONE.md](../../SETUP_SECOND_CLONE.md)）。
exp_028/029 とは転送元が違うので注意する。

---

## 0. 転送元の確認（最初に必ず）

出力がどちらのクローンにあるかを確かめる。

```bash
ssh kouyou@192.168.170.100 'ls -d /home/kouyou/VLM_OD_CL*/experiments/exp_033/*_work_dir 2>/dev/null | head'
```

`/home/kouyou/VLM_OD_CL_new/...` が出れば新クローン側。以降の `REMOTE` はそれに合わせる。

## 1. ジョブが終わっているか

**実行中に rsync すると書き込み途中のファイルを掴む。**

```bash
ssh kouyou@192.168.170.100 'squeue -u kouyou'
```

exp_033 のジョブ名は次の6つ。

```
exp033a_kdAB_condA_l2w20    exp033b_kdDE_condA_l2w20
exp033c_kdAB_condA_l2w30    exp033d_kdDE_condA_l2w30
exp033e_kdAB_condA_l2w50    exp033f_kdDE_condA_l2w50
```

途中経過だけ見たい場合は転送せずログを読む。

```bash
ssh kouyou@192.168.170.100 'tail -30 /home/kouyou/logs/result_exp033a_kdAB_condA_l2w20_*.txt'
```

## 2. 容量

| | |
|---|---:|
| 本環境 `/workspace/kouyou` の空き（2026-08-02 時点） | **1.9 TB** |
| exp_033 の見積り（12条件 × 3ドメイン × 20 epoch × 2.0GB） | **約 1.4 TB** |

**余裕が小さい。**転送前に必ず確認する。

```bash
df -h /workspace/kouyou
ssh kouyou@192.168.170.100 'du -sh /home/kouyou/VLM_OD_CL_new/experiments/exp_033'
```

足りない場合は、λ ごと（`l2w20` / `l2w30` / `l2w50`）に分けて転送し、解析が済んだものから
判断するとよい（§3.1）。

---

## 3. 転送

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_new/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments

rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --exclude='/configs/' --exclude='/*.sh' --exclude='/*.py' --exclude='/*.md' \
  "$REMOTE/exp_033/" "$LOCAL/exp_033/"
```

`--exclude` の先頭の `/` は転送元のルート（＝`exp_033/` 直下）を指す。**直下のコードだけを外し**、
`*_work_dir/` 配下の `.py`（実行時に保存された config の複製）は残す。コードは本環境が正本。

`--partial-dir=.rsync-partial` は必須。付けないと中断時に途中までのファイルが最終的な名前のまま
残り、**完全なファイルに見えてしまう**（2026-07-27 に 1.9GB の一時ファイルを git が巻き込んで
push が失敗した実例がある）。

### 3.1 λ ごとに分けて転送する場合

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_new/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments

# λ=2.0 だけ（約 470GB）
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --include='*/' --include='*l2w20*/***' --exclude='*' \
  "$REMOTE/exp_033/" "$LOCAL/exp_033/"
```

`l2w20` を `l2w30` / `l2w50` に変えて繰り返す。

### 3.2 評価結果だけ先に取る（軽量・数MB）

数値だけ先に見たい場合はこれで足りる。

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_new/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments

rsync -avh --progress \
  --include='*/' --include='*eval_after_*/***' --exclude='*' \
  "$REMOTE/exp_033/" "$LOCAL/exp_033/"
```

### 3.3 長時間になるので切断に強い形で

```bash
cd /workspace/kouyou/mmdetection
nohup bash -c '
rsync -avh --progress --partial --partial-dir=.rsync-partial \
  --exclude="/configs/" --exclude="/*.sh" --exclude="/*.py" --exclude="/*.md" \
  kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL_new/experiments/exp_033/ \
  /workspace/kouyou/mmdetection/experiments/exp_033/
' > experiments/retrieve_exp033.log 2>&1 &

tail -f experiments/retrieve_exp033.log
```

**中断したら同じコマンドを再実行すればよい。** 転送済みはスキップされ、途中のファイルは続きから再開する。

---

## 4. 転送後の確認

### 4.1 個数の照合

```bash
echo -n "クラスタ ckpt: "; ssh kouyou@192.168.170.100 \
  "find /home/kouyou/VLM_OD_CL_new/experiments/exp_033 -name '*.pth' | wc -l"
echo -n "本環境   ckpt: "; find /workspace/kouyou/mmdetection/experiments/exp_033 -name '*.pth' | wc -l
du -sh /workspace/kouyou/mmdetection/experiments/exp_033
```

全条件が完走していれば ckpt は **720 個**（12条件 × 3ドメイン × 20 epoch）。

### 4.2 途中ファイルが残っていないか

```bash
find /workspace/kouyou/mmdetection/experiments/exp_033 \
  -name '.rsync-partial' -o -name '.*.pth.*'
```

**何も出なければ完了。** 出たら rsync を再実行する。

### 4.3 評価結果の抽出

**json は1ファイルに複数レコードが改行なしで連結されている場合がある。**
`json.load` は `Extra data` で失敗するので `raw_decode` を使う。

```bash
cd /workspace/kouyou/mmdetection
python3 - <<'PY'
import glob, json, os, re
dec = json.JSONDecoder()
def recs(f):
    s = open(f).read(); out = []; i = 0
    while i < len(s):
        while i < len(s) and s[i] in ' \t\r\n': i += 1
        if i >= len(s): break
        try: o, j = dec.raw_decode(s, i); out.append(o); i = j
        except Exception: break
    return out
res = {}
for f in glob.glob('experiments/exp_033/**/*.json', recursive=True):
    if 'eval' not in f or 'vis_data' in f: continue
    rs = [r for r in recs(f) if isinstance(r, dict) and 'coco/bbox_mAP' in r]
    if not rs: continue
    p = os.path.relpath(f, 'experiments/exp_033').split(os.sep)
    res[os.sep.join(p[:-2])] = rs[-1]['coco/bbox_mAP']
print(f'{len(res)} 件 / 完走なら 108 件（12条件 × 9評価）')
for k in sorted(res): print(f'  {k:58s} {res[k]}')
PY
```

### 4.4 λ が意図どおり効いていたかの確認

学習ログの `loss_kd` と `loss` を見る。**λ を上げたのに `loss_kd` が比例して増えていなければ、
条件の取り違えを疑う。**

```bash
cd /workspace/kouyou/mmdetection
python3 - <<'PY'
import glob, json
for f in sorted(glob.glob('experiments/exp_033/*_work_dir/*/vis_data/scalars.json')):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    tr = [r for r in rows if 'loss_kd' in r]
    if not tr: continue
    a, b = tr[0], tr[-1]
    tag = f.split('/')[2]
    print(f'{tag:48s} loss_kd {a["loss_kd"]:8.3f} -> {b["loss_kd"]:8.3f}   '
          f'loss {a.get("loss",0):7.3f} -> {b.get("loss",0):7.3f}')
PY
```

**λ = 5.0 の蒸留B は検出損失の 6.7 倍になる見込み**（design §3）。学習が破綻していないかを
`loss` の推移で確認する。

---

## 5. 注意点

- **rsync 実行中に `git add -A` を使わない。** 転送中の一時ファイルを巻き込む。`.gitignore` に
  除外を入れてあるが、rsync の完了を待ってからコミットするのが確実。
- **exp_028/029 の転送元は元のリポジトリ**（`/home/kouyou/VLM_OD_CL`）。exp_033 とは別。
  手順は [RETRIEVE_EXP028_029.md](../RETRIEVE_EXP028_029.md)。
- 回収と照合が済んだら、クラスタ側の新クローンを消すかどうかを判断する
  （[SETUP_SECOND_CLONE.md](../../SETUP_SECOND_CLONE.md) §5）。**削除は元に戻せない。**
