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

### データローダー作成までの流れ
```
【train.py】
 main() の中で runner = Runner.from_cfg(cfg)  → runner.train() を呼ぶ
        │
        ▼
【Runner.from_cfg】（runner.py）
 config を Runner.__init__ の引数に振り分ける
 このとき __init__ に train_dataloader（configの設定）を渡す
        │
        ▼
【Runner.__init__】（runner.py）
 self._train_dataloader = train_dataloader  ← 辞書のまま保持（まだ作らない）
 self._train_loop       = train_cfg          ← 辞書のまま保持（まだ作らない）
        │
        ▼
【runner.train()】（runner.py）
 self._train_loop = self.build_train_loop(self._train_loop) を呼ぶ
        │
        ▼
【build_train_loop】（runner.py）
 LOOPS.build(..., default_args=dict(dataloader=self._train_dataloader))
 → EpochBasedTrainLoop を構築。このとき dataloader（まだ辞書）を渡す
        │
        ▼
【EpochBasedTrainLoop.__init__】（loops.py）
 super().__init__(runner, dataloader)  ← 親クラスに dataloader を渡すだけ
        │
        ▼
【BaseLoop.__init__】（base_loop.py）
 if isinstance(dataloader, dict):                         ← 辞書なので
     self.dataloader = runner.build_dataloader(dataloader, ...)  ← ここで変換！
        │
        ▼
【build_dataloader】（runner.py）★データローダー作成の本体
 ① dataset = DATASETS.build(dataset_cfg)  → CocoDataset を構築
 ②   └ dataset.full_init()  → アノテーション読み込み
 ③ sampler = DATA_SAMPLERS.build(...)  → サンプラー構築
 ④ collate_fn など
 ⑤ data_loader = DataLoader(...)  → 完成
        │
        ▼
 完成した DataLoader が BaseLoop の self.dataloader になる
```