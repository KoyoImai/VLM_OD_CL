# =============================================================================
# exp_026-4: オラクル（条件A＝全モジュール）/ 3ドメイン同時学習
#   underwater + electromagnetic + videogames の train を ConcatDataset で結合し、
#   1モデルを同時に学習する（＝忘却が起きない上限の基準線）。
#
#   note04 の指定により **Objects365（参照データ）は含めない**。
#
#   各画像は「自ドメインのクラス名のみ」をプロンプトとする:
#     ODVGDataset は data_root ごとに label_map_file を持つため、RandomSamplingNegPos の
#     負例サンプリングは各画像の所属ドメインの label_map に閉じる。全ドメインのクラスを
#     1プロンプトに連結しないので、テキスト長制限（max_text_len=256）は発生しない。
#
#   規模不均衡は**均衡化しない**（design.md §4 の決定・2026-07-24）。
#     実測 train 画像数: underwater 12,633（27%）/ electromagnetic 25,398（55%）/
#     videogames 8,233（18%）、合計 46,264。最大:最小 ≈ 3.1:1 と偏りは中程度のため、
#     RepeatDataset 等の倍率ハイパラを導入せず、素の ConcatDataset で自然な分布のまま学習する。
#
#   スケジュール（epoch=20 / milestone[15] / lr=1e-4 / batch4 / seed=0 / init_cfg=None）は
#   exp_024 のベースを継承して逐次学習と揃える。結合により1エポックのデータ量は約3倍に
#   なるが、エポック数の定義を揃えることを優先する（design.md §4）。
#   既存ファイルは無変更。
# =============================================================================
_base_ = '../../exp_024/configs/fullft_replayfree_base.py'

train_pipeline = _base_.train_pipeline  # ODVG＋負例サンプリング（事前学習から継承）

_rf100 = '/workspace/kouyou/datasets/rf100_domain/'


def _domain(name):
    """ドメイン名から ODVGDataset の設定を作る（label_map はドメイン別）。"""
    return dict(
        type='ODVGDataset',
        data_root=_rf100 + name + '/',
        ann_file=name + '_train_od.json',
        label_map_file=name + '_label_map.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False),
        return_classes=True,
        pipeline=train_pipeline)


# 3ドメインを結合（均衡化なし・自然な分布のまま）
train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            _domain('underwater'),
            _domain('electromagnetic'),
            _domain('videogames'),
        ]))
