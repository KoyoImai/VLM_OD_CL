# クラスタ側の `VLM_OD_CL` を消して clone し直す前の確認手順

作成: 2026-08-06

`git pull` の代わりに、クラスタの `/home/kouyou/VLM_OD_CL` を削除して `git clone` し直す場合の
確認事項をまとめる。**削除して困るのは git 管理外のファイルだけ**なので、そこを潰せば安全に行える。

---

## 0. 何が clone で戻り、何が戻らないか

| | git 管理 | 削除したら |
|---|---|---|
| コード（`mmdet/`, `tools/`, `configs/`, `experiments/*/configs`, `*.sh`, `*.py`, `*.md`） | **管理下** | clone で戻る |
| リプレイのバッファ定義 `experiments/exp_023/buffer/*.odvg.json`（14ファイル） | **管理下**（確認済み 2026-08-06） | clone で戻る |
| 学習ログ `*_work_dir/*/vis_data/{scalars.json, config.py}`、`*.log` | **管理下**（コミット済みの分のみ） | コミット済みなら戻る |
| **チェックポイント `*.pth`** | **管理外**（`.gitignore` 121行目 `*.pth`） | **戻らない。完全に失われる** |
| 評価の生ログ（未コミット分） | 管理外 | 戻らない |

репо 外にあるもの（`/home/kouyou/datasets`、`/home/kouyou/ckpt`、`/home/kouyou/logs`、
`/home/kouyou/sif`）は `VLM_OD_CL` を消しても影響を受けない。**θ0 と RF100 と Objects365 と
ジョブログは安全**である。

---

## 1. 未転送の出力が残っていないか（**最重要**）

クラスタ側にあって本環境に無い ckpt があれば、消した時点で失われる。

```bash
ssh kouyou@192.168.170.100 'for d in /home/kouyou/VLM_OD_CL/experiments/exp_*; do
  n=$(find "$d" -name "*.pth" 2>/dev/null | wc -l); [ "$n" -gt 0 ] && echo "$(basename $d) $n"; done'
```

本環境側の同じ集計。

```bash
cd /workspace/kouyou/mmdetection
for d in experiments/exp_*; do n=$(find "$d" -name '*.pth' 2>/dev/null | wc -l); [ "$n" -gt 0 ] && echo "$(basename $d) $n"; done
```

**クラスタ側の数が本環境より多い実験があれば、先に転送する**（各実験の `RETRIEVE.md` を参照）。
本環境の 2026-08-06 時点の数は次のとおり。

```
exp_023 138   exp_024 60   exp_025 60   exp_026 143   exp_027 120
exp_028 240   exp_029 165  exp_033 29   exp_034 26
```

特に注意する実験を挙げる。

- **exp_030 / exp_031 / exp_032**: 本環境に ckpt が 0 個。クラスタで実行済みなら**全部が未転送**。
- **exp_033**: 本環境に 29 個しかない。ただし exp_033 は別クローン `VLM_OD_CL_new` から
  実行したので、`VLM_OD_CL` を消しても影響しない（§3 で確認する）。

## 2. 未コミット・未 push の変更がないか

クラスタ側で直接編集したファイルがあれば、それも失われる。

```bash
ssh kouyou@192.168.170.100 'cd /home/kouyou/VLM_OD_CL && git status --short | head -40'
ssh kouyou@192.168.170.100 'cd /home/kouyou/VLM_OD_CL && git log --oneline @{u}..HEAD'
ssh kouyou@192.168.170.100 'cd /home/kouyou/VLM_OD_CL && git stash list'
```

- 1つ目でコードの変更（`M`）や未追跡ファイル（`??`）が出ないこと。`??` に `*_work_dir` が
  大量に出るのは正常（出力なので追跡していない）。
- 2つ目が空であること（空でなければクラスタ側にしかないコミットがある）。
- 3つ目が空であること。

## 3. もう一方のクローン `VLM_OD_CL_new` の位置づけ

exp_033 はこちらから実行した。**消すのは `VLM_OD_CL` だけで、`_new` は残す**。

```bash
ssh kouyou@192.168.170.100 'ls -d /home/kouyou/VLM_OD_CL*'
ssh kouyou@192.168.170.100 'du -sh /home/kouyou/VLM_OD_CL /home/kouyou/VLM_OD_CL_new'
ssh kouyou@192.168.170.100 'for d in /home/kouyou/VLM_OD_CL_new/experiments/exp_*; do
  n=$(find "$d" -name "*.pth" 2>/dev/null | wc -l); [ "$n" -gt 0 ] && echo "$(basename $d) $n"; done'
```

## 4. ジョブが動いていないこと

実行中のジョブが書き込んでいるディレクトリを消すとジョブごと壊れる。

```bash
ssh kouyou@192.168.170.100 'squeue -u kouyou'
```

空であること。動いていれば終わるまで待つ。

## 5. 削除してよい条件（チェックリスト）

- [ ] §1 で、クラスタ側の ckpt 数が本環境以下（＝未転送の出力が無い）
- [ ] §2 で、`git status` に変更が無く、`@{u}..HEAD` が空、`stash` が空
- [ ] §3 で、`VLM_OD_CL_new` を消す対象に含めていない
- [ ] §4 で、実行中のジョブが無い
- [ ] 本環境で `git push` 済み（クラスタが clone する先に最新が入っている）

---

## 6. 削除して clone し直す

```bash
ssh kouyou@192.168.170.100
cd /home/kouyou
mv VLM_OD_CL VLM_OD_CL_old          # いきなり rm せず、まず退避する
git clone <リポジトリURL> VLM_OD_CL
cd VLM_OD_CL && git log --oneline -3
```

**`rm -rf` ではなく `mv` で退避する。** 動作確認が済んでから消す。ディスクが足りない場合は
先に §1・§2 を通してから `rm -rf VLM_OD_CL_old` する。

## 7. clone 後の動作確認

### 7.1 コンテナ内で mmdet が読めるか

editable install（`pip install -e .`）は `/workspace/kouyou/mmdetection` を指している。
clone 先のパスが同じなので通るはずだが、確認する。

```bash
ssh kouyou@192.168.170.100
singularity exec --nv \
  --bind /home/kouyou/VLM_OD_CL:/workspace/kouyou/mmdetection \
  /home/kouyou/sif/docker-image-of-mmdetection4singularity.sif \
  python -c "import mmdet, os; print(mmdet.__version__, os.path.dirname(mmdet.__file__))"
```

`/workspace/kouyou/mmdetection/mmdet` が出れば正しい。別の場所（site-packages 内）が出た場合は
`pip install -e .` をやり直す必要がある。

### 7.2 リプレイのバッファ定義が揃っているか

```bash
ssh kouyou@192.168.170.100 'ls /home/kouyou/VLM_OD_CL/experiments/exp_023/buffer/*.odvg.json | wc -l'
```

**7 個**（reference_o365v1_1000 と 6 ドメイン分）であること。0 なら clone が正しく済んでいない。

### 7.3 投入したい実験のスクリプトが存在するか

```bash
ssh kouyou@192.168.170.100 'ls /home/kouyou/VLM_OD_CL/experiments/exp_03{5,6}/*.sh'
```

---

## 8. そもそも clone し直す必要があるか

**exp_035 / exp_036 を回すだけなら `git pull` で足りる。** 今回 push するのは新規ファイル
（`experiments/exp_034`, `exp_035`, `exp_036`）が中心で、クラスタ側で同じファイルを触って
いなければコンフリクトは起きない。

clone し直す利点は、過去に起きた pull のコンフリクト（2026-07-27）を避けられることと、
`VLM_OD_CL` 配下の巨大な出力ごと消えてクラスタの空きが増えることである。
後者を目的とするなら、**§1 の転送を済ませてから**行う。
