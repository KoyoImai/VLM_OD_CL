# 構築手順
事前学習済みVision Language ModelのMM-Grouding DINOをも対象に，学習と評価を動かす環境を整えることを目的とします．


## Dockerコンテナの作成
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
 

## Dockerコンテナ内での環境構築

### ライブラリのバージョン修正
numpyのバージョンとtorchのバージョンが合っていなかったので，ここを修正します．
以下のコマンドを実行してください．
```
cd /workspace/kouyou/mmdetection
pip install --no-cache-dir -e .
pip install --no-cache-dir "numpy<2"
pip install --no-cache-dir "opencv-python==4.9.0.80"
pip install --no-cache-dir fairscale jsonlines
```

### MM-Grouding DINO関連のライブラリをインストール
[mm_grounding_dino/usage.md](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/usage.md)の手順に従って，必要なライブラリをインストールします．
以下のコマンドを実行してください．
```
pip install -r requirements/multimodal.txt
pip install emoji ddd-dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

### BERTの重みをダウンロード
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

### NLTK重みのダウンロード
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
```
python -c "
import nltk
nltk.download('punkt_tab', download_dir='/workspace/kouyou/datasets/nltk_data')
"
```
```
python -c "
import nltk
nltk.download('averaged_perceptron_tagger_eng', download_dir='/workspace/kouyou/datasets/nltk_data')
"
```

### MM-GroudingDINOの重みをダウンロード
```
wget load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth' # noqa
```


### デモの実行
```
# Closed-Set Object Detection
python demo/image_demo.py demo_images/animals.png \
        configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
        --texts '$: coco'

python demo/image_demo.py demo_images/animals.png \
        configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
        --texts '$: lvis'  --chunked-size 70 \
        --palette random
```

```
# オープンボキャブラリー物体検出
python demo/image_demo.py demo_images/animals.png \
        configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
        --texts 'zebra. giraffe' -c
```

### 評価の実行
```
# ゼロショットCOCO2017
python tools/test.py configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py \
        grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
```


### 学習の実行（cat_dataset）
cat_datasetを用いて学習を行えるかを確認する．
まず，cat_datasetのダウンロードを行います．
以下のコマンドを実行してください．
```
cd /workspace/kouyou/datasets
wget https://download.openmmlab.com/mmyolo/data/cat_dataset.zip
unzip cat_dataset.zip
mkdir cat_dataset
mv ./images/ ./cat_dataset
mv ./labels/ ./cat_dataset
mv ./annotations/ ./cat_dataset
```
続いて単一画像での可視化を行います．
```
export NLTK_DATA=/workspace/kouyou/datasets/nltk_data
python demo/image_demo.py /workspace/kouyou/datasets/cat_dataset/images/IMG_20211205_120756.jpg configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth --texts cat.
```
cat_datasetでファインチューニングを行います．
以下のコマンドを実行することで，1gpu，20エポックの学習が行われます．
```
python tools/train.py configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py --work-dir cat_work_dir
```
学習結果は以下のとおりです．
```
DONE (t=0.05s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.897
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.966
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.897
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.953
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.960
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.960
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.960
06/10 03:42:17 - mmengine - INFO - bbox_mAP_copypaste: 0.897 1.000 0.966 -1.000 -1.000 0.897
```
4gpuでの学習は以下のコマンドで実行します．
```
./tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py 4 --work-dir cat_work_dir
```



