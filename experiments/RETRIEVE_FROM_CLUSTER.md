# クラスタの実験結果を本環境へ持ち帰る手順（rsync）

対象: exp_024 / exp_025 / exp_026 / exp_027（クラスタで実行）
方向: クラスタ `kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments/`
　　→ 本環境 `/workspace/kouyou/mmdetection/experiments/`

すべて**本環境側**で実行する。

---

## 0. 事前確認：容量

| | 空き |
|---|---:|
| 本環境 `/workspace/kouyou`（`/dev/sda1` 7.3T） | **2.0T** |
| 全 ckpt の見積り（exp_024/025/026/027 合計） | 約 790G |

収まる。ただし exp_023 が本環境で実行中で書き足しているため、転送前に再確認する。

```bash
df -h /workspace/kouyou
```

転送元の実サイズを先に見ておく。

```bash
ssh kouyou@192.168.170.100 'du -sh /home/kouyou/VLM_OD_CL/experiments/exp_02{4,5,6,7}'
```

---

## 1. 転送（実験ごと）

**結果の転送は rsync だけで行う（git は使わない）。** チェックポイント・ログ・
`vis_data/*.json`・学習時に保存された config のコピーまで、実験ディレクトリの中身をすべて運ぶ。

除外するのは**実験ディレクトリ直下のコードだけ**（`configs/`、`*.sh`、`*.py`、`*.md`）。
これらは本環境が正本で、クラスタ側は git で配った複製にすぎない。転送すると本環境側の
未コミットの編集を上書きしてしまうため外す。

```bash
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments

for e in exp_024 exp_025 exp_026 exp_027; do
  rsync -avh --progress --partial \
    --exclude='/configs/' --exclude='/*.sh' --exclude='/*.py' --exclude='/*.md' \
    "$REMOTE/$e/" "$LOCAL/$e/"
done
```

オプションの意味。

| | |
|---|---|
| `-a` | 構成・タイムスタンプを保持 |
| `-v -h --progress` | 進捗表示 |
| `--partial` | **中断しても途中まで残り、再実行で続きから**（2GB×数百ファイルなので重要） |
| `--exclude='/...'` | 先頭の `/` は転送元のルート（＝実験ディレクトリ直下）を指す。<br>直下のコードだけを外し、`*_work_dir/` 配下の `.py`（学習時に保存された config の複製）は残す |

**既定で全部持ってくる形にしてある**ので、今後 `zeroshot_eval` のような新しい出力ディレクトリが
増えても自動的に対象になる。逆に「出力ディレクトリ名を列挙して include する」書き方は、
名前が想定外だったときに黙って漏れるので採らない。

`-z`（圧縮）は付けない。`.pth` は既に zip 形式で圧縮効果が乏しく、CPU を食うだけになる。

### 長時間になるため、切断に強い形で流す

790G は LAN 速度次第で数時間かかる。ssh が切れても続くように `nohup` で流す。

```bash
cd /workspace/kouyou/mmdetection
nohup bash -c '
REMOTE=kouyou@192.168.170.100:/home/kouyou/VLM_OD_CL/experiments
LOCAL=/workspace/kouyou/mmdetection/experiments
for e in exp_024 exp_025 exp_026 exp_027; do
  rsync -avh --progress --partial \
    --exclude="/configs/" --exclude="/*.sh" --exclude="/*.py" --exclude="/*.md" \
    "$REMOTE/$e/" "$LOCAL/$e/"
done
' > experiments/retrieve_from_cluster.log 2>&1 &
```

進捗の確認。

```bash
tail -f /workspace/kouyou/mmdetection/experiments/retrieve_from_cluster.log
df -h /workspace/kouyou
```

**中断した場合は同じコマンドをもう一度実行すればよい。** 転送済みのファイルはスキップされ、
途中のファイルは `--partial` により続きから再開される。

---

## 2. 転送後の確認

ckpt の個数とサイズが一致するかを両側で比較する。

```bash
# クラスタ側
ssh kouyou@192.168.170.100 'find /home/kouyou/VLM_OD_CL/experiments/exp_027 -name "*.pth" | wc -l; du -sh /home/kouyou/VLM_OD_CL/experiments/exp_027'

# 本環境側
find /workspace/kouyou/mmdetection/experiments/exp_027 -name "*.pth" | wc -l
du -sh /workspace/kouyou/mmdetection/experiments/exp_027
```

評価結果が揃っているかは mAP を並べて確認する。

```bash
cd /workspace/kouyou/mmdetection
for f in experiments/exp_027/*eval*/*/*/*.json; do
  printf "%-70s %s\n" "$f" "$(python3 -c "import json,sys;print(json.load(open('$f')).get('coco/bbox_mAP'))" 2>/dev/null)"
done
```

---

## 3. 参考：optimizer 状態を落とす場合（任意・現時点では不要）

ckpt 1個 2.07GB の内訳は以下。

| 要素 | サイズ |
|---|---:|
| `state_dict`（モデル重み） | 0.69 GB |
| `optimizer`（AdamW の状態） | 1.38 GB |
| その他 | 0.01 GB |

重みだけを見る分析であれば `optimizer` は不要で、落とせば**総量が約 1/3（790G → 約 270G）**になる。
ただし学習の再開ができなくなる。今回は容量に余裕があるためそのまま持ち帰る方針。

必要になった場合の変換（**クラスタ側の原本は残したまま**、本環境で軽量版を作る）。

```bash
python3 - <<'EOF'
import glob, torch, os
for p in glob.glob('/workspace/kouyou/mmdetection/experiments/exp_027/**/*.pth', recursive=True):
    ck = torch.load(p, map_location='cpu')
    if 'optimizer' not in ck:
        continue
    out = p.replace('.pth', '.slim.pth')
    torch.save({'meta': ck.get('meta'), 'state_dict': ck['state_dict']}, out)
    print(f'{out}  {os.path.getsize(out)/1e9:.2f} GB')
EOF
```

---

## 注意

- `*_work_dir` 配下は編集しない（禁止則）。転送はコピーなので問題ない。
- **クラスタ → 本環境の結果の転送に git は使わない**（2026-07-26 方針）。指標が入った
  `vis_data/*.json` も含め、すべて rsync で運ぶ。
- コード（`configs/` やスクリプト）の向きは逆で、本環境が正本。クラスタへは git で配る。
  そのため rsync では実験ディレクトリ直下のコードを除外している。
