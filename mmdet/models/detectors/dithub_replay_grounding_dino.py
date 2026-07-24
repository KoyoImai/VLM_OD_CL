"""exp_023: DitHub+リプレイ用の新規サブクラス（既存 dithub コードは一切無変更）.

問題: DitHub の specialization 学習 forward である DitHubLinear._delta_per_sample は
`torch.stack([self.per_class_lora_A[k] for k in keys])` を無フィルタで引くため、
ミニバッチに参照(Objects365)・過去ドメイン画像（現在ドメインの dithub_classes に無い
クラス）が入ると KeyError で落ちる（評価パスは `if k in ...` でフィルタ済みだが学習は未）。

対処（ユーザー選択: warmup_lora_a 経路, 2026-07-22）: ライブラリ非登録キーの画像は
specialization 中も warmup_lora_a（クラス共通 A）＋共有 B を通す。これによりクラッシュを
避けつつ、リプレイ画像が汎用 A・B の学習に寄与する。評価時は既存フィルタにより参照専用
クラスはモジュールを持たず凍結モデルを通る（§1 特則は評価で成立）。

実装方針: 既存の dithub_layers.py / dithub_grounding_dino.py を変更せず、
メソッドのみ差し替えるサブクラスを新規に定義する。DitHubReplayLinear は DitHubLinear の
属性・パラメータを完全に継承し `_delta_per_sample` だけを上書きするため、
DitHubReplayGroundingDINO は親の _apply_dithub が生成した DitHubLinear の __class__ を
差し替えるだけでよい（新パラメータを増やさない挙動のみのサブクラス）。
"""
import random
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.layers.dithub_layers import DitHubLinear, canonical_key, STATE
from mmdet.models.detectors.dithub_grounding_dino import DitHubGroundingDINO


class DitHubReplayLinear(DitHubLinear):
    """specialization 中、ライブラリ非登録キーの画像は warmup_lora_a を使う.

    親 DitHubLinear._delta_per_sample の唯一の差分は lora_a スタックの構築で、
    キーがライブラリに無ければ warmup_lora_a を代替に用いる（それ以外の
    bmm/転置ロジックは親と同一）。属性・パラメータは追加しない。
    """

    def _delta_per_sample(self, x: Tensor, keys: List[str]) -> Tensor:
        lora_a = torch.stack([
            self.per_class_lora_A[k]
            if k in self.per_class_lora_A else self.warmup_lora_a
            for k in keys
        ])
        if x.dim() == 2:
            h = torch.bmm(x.unsqueeze(1), lora_a.transpose(1, 2)).squeeze(1)
            return F.linear(h, self.shared_lora_b) * self.scaling
        assert x.dim() == 3, f'unsupported input dim {x.dim()}'
        transposed = False
        if x.shape[0] != len(keys):
            # seq-first (L, bs, E) のケース（テキスト self-attn 内部）
            assert x.shape[1] == len(keys), (
                f'batch mismatch: x={tuple(x.shape)}, classes={len(keys)}')
            x = x.transpose(0, 1)
            transposed = True
        h = torch.bmm(x, lora_a.transpose(1, 2))
        delta = F.linear(h, self.shared_lora_b) * self.scaling
        if transposed:
            delta = delta.transpose(0, 1)
        return delta


@MODELS.register_module()
class DitHubODVGGroundingDINO(DitHubGroundingDINO):
    """DitHub の loss() を ODVG（キャプション文字列）の text 形式に対応させた版.

    背景: 既存 DitHubGroundingDINO.loss() は画像ごとのクラス名を
    `names = list(ds.text)` で取り出すが、これは CocoDataset のように
    `ds.text` がクラス名のリスト/タプルであることを前提にしている。
    ODVG 学習（RandomSamplingNegPos）では `ds.text` がキャプション文字列
    （"shark. pipe. fish. ..."）なので、`list()` が1文字ずつに割ってしまい、
    `names[label]` が 'p' 等の1文字となって per_class_lora_A に無く KeyError で落ちる
    （実測: exp_023 の DitHub リプレイなし学習でクラッシュ）。

    修正: text が文字列なら `". "` で分割してクラス名列を復元する（GT ラベルは
    キャプション内の位置インデックスに再マップ済みなので names[label] が正しい
    クラス名になる。実データで検証: label→'pipe'→class_pipe が in_lib=True）。
    リスト/タプル（CocoDataset）の場合は従来通り list()。評価パス predict() は
    既存実装が list/tuple 分岐済みで問題ないため上書きしない。
    既存の dithub_grounding_dino.py は変更しない。
    """

    def loss(self, batch_inputs, batch_data_samples):
        keys = []
        for ds in batch_data_samples:
            text = ds.text
            names = text.split('. ') if isinstance(text, str) else list(text)
            labels = ds.gt_instances.labels.unique().tolist()
            if not labels:
                labels = [0]  # 公式実装と同じフォールバック
            label = random.choice(labels)
            keys.append(canonical_key(names[label]))
        STATE.train_class_keys = keys
        # DitHubGroundingDINO.loss（＝壊れたキー抽出）を飛ばして
        # GroundingDINO.loss を直接呼ぶ。
        return super(DitHubGroundingDINO, self).loss(batch_inputs,
                                                     batch_data_samples)


@MODELS.register_module()
class DitHubReplayGroundingDINO(DitHubODVGGroundingDINO):
    """DitHub 検出器の DitHubLinear を挙動のみ差し替えた DitHubReplayLinear に置換.

    親の _apply_dithub が Linear を DitHubLinear へ変換した直後に、生成された
    DitHubLinear インスタンスの __class__ を DitHubReplayLinear へ差し替える。
    新パラメータを増やさないメソッド専用サブクラスなので __class__ 差し替えで安全。
    isinstance(m, DitHubLinear) を用いる親の init_weights / enable_per_class /
    freeze は subclass でも True のまま機能する。loss() は ODVG 対応の
    DitHubODVGGroundingDINO のものを継承する。
    """

    def _apply_dithub(self, lora_r: int, lora_alpha: int) -> list:
        replaced = super()._apply_dithub(lora_r, lora_alpha)
        for m in self.modules():
            if type(m) is DitHubLinear:
                m.__class__ = DitHubReplayLinear
        return replaced
