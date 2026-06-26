# LaTeX 環境の使い方（導入〜コンパイル）

本プロジェクトの資料作成用 LaTeX 環境について、**実際に行った手順どおり**にまとめる。
**LaTeX 関係の資料はすべて `experiments/latex/` 配下に置く**こと。
- `experiments/latex/latex.md` … 本ファイル
- `experiments/latex/template/` … テンプレート（MPRG Work Document, pLaTeX系）
- `experiments/latex/template.zip` … テンプレートの配布元 zip
- `experiments/latex/<資料名>/` … 個別の資料（例: `survey_doc/`）

- 使用エンジン: **pLaTeX**（`platex` + `pbibtex` + `dvipdfmx`）。文書クラスは `jarticle`（日本語）。
- 文献スタイル: `splncs04`（Springer LNCS）。`ref.bib` を `pbibtex` で処理。
- 自動ビルド: `latexmk`（テンプレ同梱の `latexmkrc` がエンジンを指定）。
- **日本語フォントは IPAex（IPAexMincho/IPAexGothic）に統一**する（テンプレ `root.pdf` と同一・埋め込み）。
  導入: `apt-get install -y fonts-ipaexfont` → `kanji-config-updmap-sys ipaex`（詳細は §6）。
- **テンプレに無いパッケージは安易に足さない**（体裁を崩さないため）。テンプレの読込は
  `nag` / `mprg` / `datetime` / `ascmac` のみ。表は `mprg` が読む `tabularx` と `\hline` で組む
  （`booktabs` は使わない）。

---

## 0. 前提と「ML環境を壊さない」理由
- **ML環境は conda（`/opt/conda`）に隔離**されている（Python/CUDA/mmdetection）。
- **LaTeX（TeX Live）は apt でシステム領域（`/usr/bin`, `/usr/share/texlive`）に入る別系統**。
  conda・pip・CUDA・mmdetection には一切触れない。apt は「追加」のみで、ML関連の削除/更新は起きない。
- 共有資源はディスク容量のみ（導入は約1.5〜2GB、空きは十分）。
- 環境: Ubuntu 24.04, root, apt 利用可。

---

## 1. 導入（インストール）
### 1-1. apt リポジトリ更新
```bash
apt-get update
```

### 1-2. TeX Live（日本語＋必要コンポーネント）を導入
テンプレートが必要とするものをカバーするターゲット導入（`texlive-full` は不要）。
```bash
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  texlive-lang-japanese \
  texlive-latex-recommended \
  texlive-latex-extra \
  texlive-fonts-recommended \
  texlive-publishers \
  texlive-science \
  latexmk
```
各パッケージの役割:
| パッケージ | 提供物（テンプレで使うもの） |
|---|---|
| `texlive-lang-japanese` | `platex`, `pbibtex`, 日本語フォント, `ascmac` |
| `texlive-latex-recommended` | 基本 LaTeX, `fancyhdr`, `caption` 等 |
| `texlive-latex-extra` | `subfigure`, `datetime`, `onlyamsmath`, `cases`, `nag`, `subcaption` 等 |
| `texlive-fonts-recommended` | 標準フォント |
| `texlive-publishers` | `splncs04.bst`（LNCS文献スタイル） |
| `texlive-science` | 数式関連の追加 |
| `latexmk` | 自動ビルドツール |

> ダウンロード/展開で数分かかる。バックグラウンド実行可（ログを `/tmp/texlive_install.log` 等に出す）。

---

## 2. 導入の検証
### 2-1. コマンドが入ったか
```bash
for c in platex pbibtex dvipdfmx latexmk uplatex; do
  printf "%-10s " "$c"; command -v $c || echo "(none)"; done
```
期待: すべて `/usr/bin/...` が表示される。

### 2-2. 必須スタイル/BSTファイルの所在確認
```bash
for f in splncs04.bst ascmac.sty onlyamsmath.sty cases.sty datetime.sty \
         fancyhdr.sty subfigure.sty nag.sty tabularx.sty; do
  printf "%-18s " "$f"; kpsewhich $f || echo "(MISSING)"; done
```
- すべてパスが出れば OK。
- **`mprg.sty` は `kpsewhich` で出なくて正常**（テンプレ同梱のローカルファイルで、システムには未インストール）。
  テンプレートディレクトリ内でコンパイルすれば参照される。

### 2-3. ML環境が無事か（任意）
```bash
python3 -c "import mmdet; print('mmdet', mmdet.__version__)"
```

---

## 3. テンプレートの構成
`experiments/latex/template/`:
```
main.tex      # 本体（jarticle, \usepackage{mprg}）。ここを編集して資料を書く
mprg.sty      # MPRG 独自スタイル（余白・ヘッダ・数式・図表設定）
latexmkrc     # ビルド設定（platex+pbibtex+dvipdfmx を指定）
ref.bib       # 参考文献データベース（BibTeX）
img/          # 画像（\includegraphics で参照）
root.pdf      # 参考の完成例 PDF
```
`latexmkrc` の中身（このためエンジン指定不要で `latexmk` だけで回る）:
```perl
$latex   = 'platex';
$bibtex  = 'pbibtex';
$dvipdf  = 'dvipdfmx %O -o %D %S';
$pdf_mode = 3;   # dvipdfmx 経由で PDF 生成
```

---

## 4. コンパイル（PDF生成）
### 4-1. 推奨: latexmk（全自動）
テンプレートディレクトリ内で実行する。`latexmk` が
「platex → （必要なら）pbibtex → platex 再実行 → dvipdfmx」を自動で回す。
```bash
cd experiments/latex/template
latexmk main.tex          # → main.pdf を生成
```
出力例（成功時）:
```
main.dvi -> main.pdf
Latexmk: All targets (main.dvi main.pdf) are up-to-date
```
生成物: `main.dvi`, `main.pdf`（`%PDF-1.5` で始まる有効な PDF）。

### 4-2. 中間ファイルの掃除
```bash
latexmk -c     # .aux/.log/.dvi/.bbl 等の中間ファイルを削除（main.pdf は残す）
latexmk -C     # main.pdf も含めて全削除
```

### 4-3. 参考: 手動で実行する場合（latexmk を使わない）
```bash
cd experiments/latex/template
platex main.tex        # .aux, .dvi を生成（文献の相互参照のため複数回必要なことがある）
pbibtex main           # ref.bib を処理して .bbl を生成（\cite を使う場合）
platex main.tex        # 文献ラベルを反映
platex main.tex        # 相互参照を確定
dvipdfmx main.dvi      # .dvi → main.pdf
```
> 通常は 4-1 の `latexmk` で十分。手動はトラブル時の理解用。

---

## 5. 新しい資料を作るとき
**新しい LaTeX 資料は `experiments/latex/<資料名>/` に作る**（テンプレを資料ごとに複製する運用）:
```bash
cp -r experiments/latex/template experiments/latex/<資料名>
cd experiments/latex/<資料名>
# main.tex を編集（必要なら別名 .tex にしてもよい）→ latexmk でビルド
```
`template/` から最低限 `mprg.sty` `latexmkrc` `ref.bib` `img/` と本文 `.tex` が揃っていればよい。
実例: `experiments/latex/survey_doc/`（既存研究サーベイ。`survey.tex` + `figs/`）。

最低限の編集箇所（`main.tex`）:
```latex
\title{タイトル\\ \Large{サブタイトル}}
\author{TP26801 今井　孝洋}
\date{2026年xx月xx日}
...
\section{...}   % 本文
\bibliography{ref}   % ref.bib を使う場合
```

---

## 6. トラブルシューティング
- **`! LaTeX Error: File 'xxx.sty' not found.`**
  → 不足パッケージ。`kpsewhich xxx.sty` で確認し、対応する texlive-* を apt で追加導入。
  どの texlive パッケージに含まれるか不明なら `apt-get install apt-file && apt-file update && apt-file search xxx.sty`。
- **日本語が出ない/文字化け**: 本テンプレは `platex`(EUC/utf8) 前提。`pdflatex` ではなく `platex`+`dvipdfmx`（=`latexmk`）を使う。
- **PDFで日本語が「空白」になる／フォントがテンプレと違う**: 日本語フォントが**未埋め込み**（dvipdfmx 既定 `noEmbed`＝Ryumin/GothicBBB 参照）か，別フォント（Harano Aji 等）になっている。
  **テンプレ（`root.pdf`）と同じ IPAex に統一**して再ビルドする:
  ```bash
  kanji-config-updmap-sys status      # CURRENT family を確認（ipaex でなければ要変更）
  apt-get install -y fonts-ipaexfont  # IPAex フォント本体（未導入なら）
  kanji-config-updmap-sys ipaex       # 日本語フォントを IPAex に切替（システム全体・埋め込み）
  cd <texdir> && latexmk -gg <file>.tex
  pdffonts <file>.pdf                 # IPAexMincho/IPAexGothic が emb=yes なら成功（要 poppler-utils）
  ```
  ※ `kanji-config-updmap-sys ipaex` が `ipaex not available` となる場合は `fonts-ipaexfont` 未導入。
  ※ `pdffonts` が無ければ `apt-get install -y poppler-utils`。いずれも TeX 設定のみで ML 環境に影響しない。
- **文献（\cite）が `[?]` になる**: `pbibtex` 実行後に `platex` を2回。`latexmk` なら自動。
- **`mprg.sty` not found**: テンプレートディレクトリの中で実行しているか確認（カレントディレクトリに `mprg.sty` が必要）。
- **生成物をリポジトリに含めたくない**: `.gitignore` に
  `*.aux *.log *.dvi *.bbl *.blg *.out *.fls *.fdb_latexmk` を追加。

---

## 付録: 確認済みバージョン
- TeX Live 2023/Debian（`e-upTeX 3.141592653-p4.1.0-...`, `dvipdfmx 20220710`）。
- 導入〜`main.tex` のコンパイル成功までを本手順で確認済み（2026-06-22）。
</content>
