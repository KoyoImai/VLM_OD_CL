# Objects365 v1 ダウンロード手順（exp_023 参照データ）

- 作成日: 2026-07-22
- 目的: リプレイの参照データ（バッファ）用に Objects365 **v1** を取得する。保存先 `/workspace/kouyou/datasets/objects365v1/`。学習には一部（1,000サンプル）のみ使用。
- 入手元: OpenDataLab（`OpenDataLab/Objects365_v1`）。**認証（AK/SK）が必要**。no-auth の公開ミラーは v2 のみで、v1 は無かった（別途調査）。

---

## 0. 環境への影響（重要）

`opendatalab` / `openxlab` は共有パッケージを古い版に固定するため、**本環境（conda）に入れると以下がダウングレードされる**。

| パッケージ | 元の版 | 導入後 |
|---|---|---|
| urllib3 | 2.2.2 | 1.26.20 |
| setuptools | 72.1.0 | 60.2.0 |
| requests | 2.32.3 | 2.28.2 |
| rich | 15.0.0 | 13.4.2 |
| tqdm | 4.66.4 | 4.65.2 |
| filelock | 3.29.0 | 3.14.0 |

- 核心の ML スタック（torch 2.1.2 / mmengine 0.10.3 / mmdet 3.3.0 / mmcv 2.1.0）は**変更されない**（学習は動く）。
- 症状: `typer requires rich>=13.8.0` 等の不整合警告。
- **推奨は別 venv での実行**（下記5）。今回は本環境に入れてしまったため、ダウンロード後に復元する（下記4）。

## 1. インストール

```bash
pip install opendatalab   # openxlab も依存として入る
```

## 2. ログイン（AK/SK 方式・対話）

GitHub 連携アカウントはパスワードが無いため、`odl login`（パスワード方式・非推奨）ではなく **openxlab の AK/SK ログイン**を使う。

```bash
# 認証情報（AK/SK）はチャットに貼らず、! プレフィックスで自セッション実行
openxlab login          # AK → SK の順で入力（発行済みキー）
# 入れ直す場合: openxlab login -r
```

## 3. ダウンロード

```bash
openxlab dataset get -r "OpenDataLab/Objects365_v1" -t "/workspace/kouyou/datasets/"
```

- v1 全量（約600K画像・数十GB）。空き 238G。
- repo 名が異なる場合は `openxlab dataset info -r "OpenDataLab/Objects365_v1"` で確認。
- 認証情報はホームに保存されるため、ログイン後の取得は非対話で実行可能。

## 4. 環境の復元（ダウンロード完了後）

opendatalab は urllib3<1.27 を要求するので、**先に復元すると opendatalab が動かない**。必ず「ダウンロード → 復元」の順。

```bash
pip uninstall -y opendatalab openxlab oss2 aliyun-python-sdk-core aliyun-python-sdk-kms crcmod jmespath pycryptodome colorama pytz
pip install "setuptools==72.1.0" "requests==2.32.3" "urllib3==2.2.2" "rich==15.0.0" "tqdm==4.66.4" "filelock==3.29.0"
```

## 5. 次回のための推奨（別 venv 方式）

本環境を汚さないための正攻法。ダウンロードした画像・JSON は環境非依存なので、venv を消しても残る。

```bash
python -m venv /workspace/odl_env
source /workspace/odl_env/bin/activate
pip install opendatalab
openxlab login                                            # AK/SK
openxlab dataset get -r "OpenDataLab/Objects365_v1" -t "/workspace/kouyou/datasets/"
deactivate
rm -rf /workspace/odl_env                                 # 用済み後に削除（本環境は無傷）
```

## 6. ダウンロード後の配置・変換（本環境・非対話）

```bash
# 想定構造に整形:
#   /workspace/kouyou/datasets/objects365v1/objects365_train.json
#   /workspace/kouyou/datasets/objects365v1/train/  (画像)
# リポジトリのツール用にシンボリックリンク:
mkdir -p data
ln -s /workspace/kouyou/datasets/objects365v1 data/objects365v1
# OD-ODVG 変換（o365v1 の label_map を正しく生成）:
python tools/dataset_converters/coco2odvg.py data/objects365v1/objects365_train.json -d o365v1
#   -> o365v1_train_od.json, o365v1_label_map.json
```

## セキュリティ注意

- AK/SK はチャットに貼らない。`! ` プレフィックスで自分のセッションに直接入力する。
