# コード分析
## データセット関連のメモ
### CocoDatasetクラス関連
CocoDatasetクラスのload_data_list()で用意したdata_list（slef.data_list）の中身に関するメモ
```
{
    # --- 画像の基本情報（処理2で詰めた） ---
    'img_path': 'images/IMG_xxx.jpg',   # 画像のパス
    'img_id': 12,                        # 画像ID
    'seg_map_path': None,                # セグメンテーションマスクのパス（使わないのでNone）
    'height': 480,                       # 画像の高さ
    'width': 640,                        # 画像の幅

    # --- Grounding DINO 用のテキスト情報（処理3で詰めた、return_classes=True のため） ---
    'text': ('cat',),                    # クラス名（モデルに渡すテキスト）
    'caption_prompt': None,              # プロンプト設定（通常None）
    'custom_entities': True,             # カスタムエンティティのフラグ

    # --- 各物体の情報（処理4で詰めた） ---
    'instances': [
        {
            'bbox': [100, 50, 300, 280],   # 物体の枠 [左上x, 左上y, 右下x, 右下y]
            'bbox_label': 0,               # ラベル番号（catは0）
            'ignore_flag': 0,              # 無視フラグ（通常は0）
            # 'mask': ...                  # セグメンテーションがあれば（catは通常なし）
        },
        # 画像に複数の猫がいれば、ここに複数のinstanceが並ぶ
    ]
}
```
