# exp_028 / exp_029 の結果をクラスタから本環境へ持ち帰る手順

対象: **exp_028**（条件A + L2 蒸留 + リプレイ）／**exp_029**（条件B + L2 蒸留 + リプレイ）
方向: クラスタ `kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments/`
　　→ 本環境 `/workspace/kouyou/mmdetection/experiments/`

**結果の転送は rsync のみで行う。git は使わない**（2026-07-26 の方針）。
汎用手順は [RETRIEVE_FROM_CLUSTER.md](./RETRIEVE_FROM_CLUSTER.md)。本書は exp_028/029 固有の
パスと注意点をまとめたもの。

---

## 0. 事前確認

### 0.1 ジョブが終わっているか（**最優先**）

**実行中のジョブがある状態で rsync すると、書き込み途中のファイルを掴む。**

```bash
ssh kouyou@192.168.170.100 'squeue -u kouyou'
```

`exp028a_kdAB_condA_l2` / `exp028b_kdDE_condA_l2` / `exp029a_kdAB_condB_l2` /
`exp029b_kdDE_condB_l2` が残っていないことを確認する。

途中経過だけ見たい場合は転送せず、ログを直接読むほうが安全。

```bash
ssh kouyou@192.168.170.100 'tail -30 /home/kouyou/logs/result_exp028a_kdAB_condA_l2_*.txt'
```

### 0.2 容量

| | |
|---|---:|
| 本環境 `/workspace/kouyou` の空き（2026-07-31 時点） | **2.6 TB** |
| exp_028 の見積り（4条件 × 3ドメイン × 20 epoch × 2.0GB） | 約 480 GB |
| exp_029 の見積り | 約 480 GB |
| **合計** | **約 960 GB** |

収まるが、exp_030〜032 まで含めると 1.9 TB 追加になる。転送前に確認する。

```bash
df -h /workspace/kouyou
ssh kouyou@192.168.170.100 'du -sh /home/kouyou/VLM_OD_CL/experiments/exp_02{8,9}'
```

---

## 1. 転送

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments

for e in exp_028 exp_029; do
  rsync -avh --progress --partial --partial-dir=.rsync-partial \
    --exclude='/configs/' --exclude='/*.sh' --exclude='/*.py' --exclude='/*.md' \
    "$REMOTE/$e/" "$LOCAL/$e/"
done
```

`--exclude` の先頭の `/` は転送元のルート（＝実験ディレクトリ直下）を指す。
**直下のコードだけを外し**、`*_work_dir/` 配下の `.py`（実行時に保存された config の複製）は残す。
コードは本環境が正本で、クラスタへは git で配っている。

`--partial-dir=.rsync-partial` を付けると、転送中のファイルが隠しディレクトリに退避される。
これを付けないと、**中断時に途中までのファイルが最終的なファイル名のまま残り、完全なファイルに
見えてしまう**（2026-07-27 に `.epoch_12.pth.NNQjeZ` を git が巻き込んで push が失敗した件）。

### 長時間になるので切断に強い形で

```bash
cd /workspace/kouyou/mmdetection
nohup bash -c '
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments
for e in exp_028 exp_029; do
  rsync -avh --progress --partial --partial-dir=.rsync-partial \
    --exclude="/configs/" --exclude="/*.sh" --exclude="/*.py" --exclude="/*.md" \
    "$REMOTE/$e/" "$LOCAL/$e/"
done
' > experiments/retrieve_exp028_029.log 2>&1 &
```

進捗の確認。

```bash
tail -f /workspace/kouyou/mmdetection/experiments/retrieve_exp028_029.log
df -h /workspace/kouyou
```

**中断したら同じコマンドを再実行すればよい。** 転送済みはスキップされ、途中のファイルは続きから再開される。

---

## 2. 転送されるもの

| パス | 中身 |
|---|---|
| `exp_028/kd{A,B,D,E}_condA_l2_{dom}_work_dir/` | ckpt（epoch_1〜20）、学習ログ、`vis_data/scalars.json`、config の複製、`last_checkpoint` |
| `exp_028/kd{A,B,D,E}_condA_l2_eval_after_{dom}/on_{evd}/` | 過去・現在ドメインの評価 |
| `exp_028/kd{A,B,D,E}_condA_l2_eval_after_{dom}/zcoco/` | ZCOCO の評価 |
| `exp_029/kd{A,B,D,E}_condB_l2_*` | 同上（条件B） |
| `exp_02{8,9}/calibration/` | λ 校正の記録（`CALIBRATE=1` で実行した場合のみ） |

`{dom}` と `{evd}` は `underwater` / `electromagnetic` / `videogames`。

---

## 3. 転送後の確認

### 3.1 個数とサイズの照合

```bash
for e in exp_028 exp_029; do
  echo "=== $e ==="
  echo -n "  クラスタ: "; ssh kouyou@192.168.170.100 \
    "find /home/kouyou/VLM_OD_CL/experiments/$e -name '*.pth' | wc -l"
  echo -n "  本環境  : "; find /workspace/kouyou/mmdetection/experiments/$e -name '*.pth' | wc -l
  ssh kouyou@192.168.170.100 "du -sh /home/kouyou/VLM_OD_CL/experiments/$e"
  du -sh /workspace/kouyou/mmdetection/experiments/$e
done
```

完走していれば ckpt は **各実験 240 個**（4条件 × 3ドメイン × 20 epoch）。

### 3.2 途中ファイルが残っていないか

```bash
find /workspace/kouyou/mmdetection/experiments/exp_02{8,9} -name '.rsync-partial' -o -name '.*.pth.*'
```

**何も出力されなければ完了。** 出た場合は rsync を再実行する。

### 3.3 評価結果の抽出

**評価の json は1ファイルに複数レコードが改行なしで連結されている場合がある。**
`json.load` は `Extra data` で失敗するので、`raw_decode` で全レコードを読んで最後を採る。

```bash
cd /workspace/kouyou/mmdetection
python3 - <<'PY'
import glob, json, os
dec = json.JSONDecoder()
def recs(f):
    s = open(f).read(); out = []; i = 0
    while i < len(s):
        while i < len(s) and s[i] in ' \t\r\n': i += 1
        if i >= len(s): break
        try: o, j = dec.raw_decode(s, i); out.append(o); i = j
        except Exception: break
    return out
for e in ['exp_028', 'exp_029']:
    res = {}
    for f in glob.glob(f'experiments/{e}/**/*.json', recursive=True):
        if 'eval' not in f or 'vis_data' in f: continue
        rs = [r for r in recs(f) if isinstance(r, dict) and 'coco/bbox_mAP' in r]
        if not rs: continue
        p = os.path.relpath(f, f'experiments/{e}').split(os.sep)
        res[os.sep.join(p[:-2])] = rs[-1]['coco/bbox_mAP']
    print(f'===== {e}  ({len(res)} 件 / 完走なら 36 件) =====')
    for k in sorted(res): print(f'  {k:55s} {res[k]}')
PY
```

完走していれば **各実験 36 件**（4条件 × 各9評価）。

### 3.4 蒸留損失の推移を見る

学習ログには `loss_kd` と、診断用の `kd_img` / `kd_txt` / `kd_fus` / `kd_fus_img` / `kd_fus_txt` が
残る（診断値は合計損失に含まれない）。**λ = 1.0 が強すぎたか弱すぎたかの判断材料**になる。

```bash
cd /workspace/kouyou/mmdetection
python3 - <<'PY'
import glob, json
for f in sorted(glob.glob('experiments/exp_02[89]/*_work_dir/*/vis_data/scalars.json')):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    tr = [r for r in rows if 'loss_kd' in r]
    if not tr: continue
    a, b = tr[0], tr[-1]
    tag = f.split('/')[1] + '/' + f.split('/')[2]
    print(f'{tag:50s} loss_kd {a["loss_kd"]:.4f} -> {b["loss_kd"]:.4f}  '
          f'loss {a.get("loss",0):.3f} -> {b.get("loss",0):.3f}')
PY
```

---

## 4. 注意点

### 4.1 rsync 中に `git add -A` を使わない

転送中の一時ファイル（`.<名前>.<ランダム6文字>`）を巻き込む。`.gitignore` に除外を入れてあるが
（163〜167行目）、**rsync の完了を待ってからコミットするのが確実**。

2026-07-27 に 1.9GB の一時ファイルをコミットして push が GitHub に拒否された実例がある。

### 4.2 クラスタ側の `git pull` が衝突する

本環境で結果をコミットして push すると、クラスタが自分で生成した未追跡ファイルと衝突して
`git pull` が中断する。対処は [RECOVER_CLUSTER_PULL.md](../RECOVER_CLUSTER_PULL.md)。

**そもそも結果をコミットしなければ起きない。**転送方針（rsync のみ）を徹底するなら
`.gitignore` に次を足すという選択肢がある（既存コミット分の扱いは別途判断が必要）。

```
experiments/*/*_work_dir/
experiments/*/*eval*/
```

### 4.3 クラスタ側の容量

exp_028〜032 をすべて実行すると、クラスタのホーム（約3TB）に約 2.4 TB の ckpt が溜まる。
回収と照合（§3.1）が済んだ実験は、クラスタ側から削除することを検討する。
**削除は元に戻せない**ので、照合を終えてから判断する。
