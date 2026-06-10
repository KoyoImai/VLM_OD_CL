# VLM_OD_CL
事前学習済みVision Language Modelを用いた物体検出の継続学習

## 構築手順
事前学習済みVision Language ModelのMM-Grouding DINOをも対象に，学習と評価を動かす環境を整えることを目的とします．


### Dockerコンテナの作成
Dockerfileを使用してDockerイメージを作成します．
Dockerfileはこのリポジトリにあるものを使用しています．
以下のコマンドを実行してDickerイメージを作成してください．
```
docker build ./ --force-rm --no-cache -t mmdet-grounding:cuda12.8-torch2.1
```
上記コマンドを実行することで，イメージ名`mmdet-grounding`，タグ名`cuda12.8-torch2.1`としています．
Dockerイメージを作成後，Dockerコンテナを作成します．
以下のコマンドを実行してください．
```
docker run -it --gpus all --shm-size=128G -v /data1/kouyou:/workspace/kouyou -p 7979:7979 --name mmdet-grounding mmdet-grounding:cuda12.8-torch2.1
```
ホスト側の`/data1/kouyou`ディレクトリをコンテナ側の`/workspace/kouyou`ディレクトリにマウントします．
コンテナ名は`mmdet-grounding`としています．
 

### Dockerコンテナ内での環境構築

#### ライブラリのバージョン修正
numpyのバージョンとtorchのバージョンが合っていなかったので，ここを修正します．
以下のコマンドを実行してください．
```
cd /workspace/kouyou/mmdetection
pip install --no-cache-dir -e .
pip install --no-cache-dir "numpy<2"
pip install --no-cache-dir "opencv-python==4.9.0.80"
pip install --no-cache-dir fairscale jsonlines
```

#### MM-Grouding DINO関連のライブラリをインストール
[mm_grounding_dino/usage.md](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/usage.md)の手順に従って，必要なライブラリをインストールします．
以下のコマンドを実行してください．
```
pip install -r requirements/multimodal.txt
pip install emoji ddd-dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

#### BERTの重みをダウンロード
BERTの事前学習済み重みをダウンロードします．
以下のコマンドを実行してください．
```
python -c "
from transformers import BertConfig, BertModel
from transformers import AutoTokenizer

config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, config=config)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

config.save_pretrained('/workspace/kouyou/datasets/bert-base-uncased')
model.save_pretrained('/workspace/kouyou/datasets/bert-base-uncased')
tokenizer.save_pretrained('/workspace/kouyou/datasets/bert-base-uncased')
"
```

#### NLTK重みのダウンロード
NLTKの重みをダウンロードします．
以下のコマンドを実行してください．
```
python -c "
import nltk
nltk.download('punkt', download_dir='/workspace/kouyou/datasets/nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='/workspace/kouyou/datasets/nltk_data')
"
export NLTK_DATA=/workspace/kouyou/datasets/nltk_data
```

#### MM-GroudingDINOの重みをダウンロード
```
wget load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth' # noqa
```



