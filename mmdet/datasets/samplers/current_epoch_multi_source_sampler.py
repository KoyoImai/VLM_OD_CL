"""現在ドメイン基準で epoch を区切る MultiSourceSampler（既存無変更・新規）.

問題: 親 MultiSourceSampler は EpochBasedTrainLoop と相性が悪い。
  (1) __len__ が len(全ConcatDataset) を返し world_size で割らないため、表示上の
      1 epoch の iter 数が world_size 倍に膨らむ。
  (2) さらに __iter__ が `while True` の無限サンプラーであり、EpochBasedTrainLoop の
      `for data in dataloader` が終端しないため、**epoch が実際に区切られず epoch1 の
      まま無限に進む**（epoch が増えず、per-epoch の checkpoint も保存されない）。
  親は iter ベース（IterBasedTrainLoop, max_iters 固定）で使う設計であり、ZiRa/DitHub の
  +replay（iter ベース）は影響を受けないが、full FT/条件A（epoch ベース）で上記が顕在化した。

対処（ユーザー方針 2026-07-23）: バッチ構成（毎バッチ各ソースを混ぜる比率）・データの
rank シャード・バッファの無限循環は親のまま保ちつつ、epoch を「現在ドメイン（ソース0）が
1周する長さ」で区切る。具体的には:
  - __len__ を「現在ドメイン1周を world_size で分割した rank あたりサンプル数」に上書き。
  - __iter__ を「親の無限イテレータから __len__ 個だけ切り出す」有限版に上書き。
    これで DataLoader が epoch 末で StopIteration を返し、EpochBasedTrainLoop が epoch を
    進める。バッファ側ソースは epoch 中に循環して重複使用される（許容）。

    iters/epoch = ceil( len(source0) / (world_size × num_per_source[0]) )
    __len__（rank あたりのサンプル数）= iters/epoch × batch_size

ソース0 が現在ドメインである前提（config は datasets=[現在, 参照, 過去…] の順で統一）。
既存の multi_source_sampler.py は一切変更しない。
"""
import itertools
import math
from typing import Iterator

from mmdet.registry import DATA_SAMPLERS
from mmdet.datasets.samplers.multi_source_sampler import MultiSourceSampler


@DATA_SAMPLERS.register_module()
class CurrentEpochMultiSourceSampler(MultiSourceSampler):
    """1 epoch = 現在ドメイン(ソース0)の1周（world_size で分割）で区切る有限版.

    バッチ構成・混合比率・データのシャード（source2inds）は親のまま用い、
    __len__（epoch 長）と __iter__（有限化）だけを上書きする。source2inds は
    __init__ で作られる共有の無限イテレータで、epoch を跨いで循環し続けるため、
    epoch ごとに異なるデータが供給される（バッファ側は重複使用を許容）。
    """

    def __len__(self) -> int:
        n_current = len(self.dataset.datasets[0])
        iters_per_epoch = math.ceil(
            n_current / (self.world_size * self.num_per_source[0]))
        return iters_per_epoch * self.batch_size

    def __iter__(self) -> Iterator[int]:
        # 親の無限イテレータから __len__ 個（＝現在ドメイン1周分のバッチ×batch_size、
        # batch_size の倍数なのでバッチ境界で止まる）だけ切り出して有限化する。
        return itertools.islice(super().__iter__(), len(self))
