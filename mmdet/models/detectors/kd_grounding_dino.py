"""特徴蒸留つき GroundingDINO（exp_028/029/030/031 用・新規ファイル。既存コード無変更）.

design: experiments/exp_028/design.md §4

- 教師は θ_{t-1}（t=1 は θ0）。凍結し `torch.no_grad()` で1回だけ前向きする。
- 蒸留はバッファ由来サンプルのみに適用する。
- 蒸留対象は neck 出力（img）／text_feat_map 出力（txt）／feature enhancer 出力（fus）。

教師の保持:
    `object.__setattr__` で nn.Module の属性登録を回避して保持する。これにより
      - state_dict に含まれない（ckpt が肥大しない。1個 2.0GB のまま）
      - DDP の reducer に載らない / optimizer のパラメータ群に入らない
      - train()/eval() の伝播を受けない（常に eval のまま）
    となる。`init_weights()` の時点で self は既に GPU 上にあるため、deepcopy した
    教師も同じデバイスに載る。

学生の中間特徴の捕捉:
    既存の `grounding_dino.py` を変更しない方針のため、親の `loss()` をそのまま呼び、
    その過程で計算される中間特徴を forward hook と最小限のオーバーライドで捕捉する。
      - text_feat_map への forward hook  -> txt（蒸留B の対象）
      - language_model への forward hook -> プロンプト文字列とトークンマスク
      - extract_feat のオーバーライド     -> img（蒸留A の対象。neck 出力）
      - forward_encoder のオーバーライド  -> fus（蒸留D の対象）
    hook は教師を deepcopy した**後**に登録するため、教師側には付かない。

バッファ由来サンプルの特定:
    `MultiSourceSampler.__iter__` はソース順（source0 → source1 → …）に添字を連結して
    yield するため、1 バッチの並びは [現在×4, 参照×1, 過去×1]（t=1 は [現在×4, 参照×2]）で
    固定される。したがってバッチ末尾の `num_buffer_per_batch` 件がバッファ由来である。
    この前提は実装検証で img_path とバッファ JSON を突き合わせて確認する（design §7 項目2）。

`__init__.py` には登録しない。config の custom_imports でフルパス指定して読み込む。
"""
import copy
import functools
import os.path as osp
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from mmengine.logging import print_log
from mmengine.runner.checkpoint import load_checkpoint
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from .grounding_dino import GroundingDINO

_TARGETS = ('img', 'txt', 'fus')


def _strip_activation_checkpoint(module) -> int:
    """activation checkpoint による forward の差し替えを解除する.

    GroundingDinoTransformerEncoder は `num_cp` により fairscale の
    `checkpoint_wrapper` で encoder の layers / fusion_layers をラップする。これは
    インスタンス属性 `forward` を

        functools.partial(_checkpointed_forward, 元のforward, weakref(元のモジュール), ...)

    に差し替える実装であり、**元のモジュールへの weakref を保持する**。そのため
    `copy.deepcopy` したモデルでも、差し替えられた forward は**複製元**のモジュール
    （＝学生）を実行してしまう。教師をこのまま使うと、

      - 教師の encoder 出力が学生の重みで計算される（蒸留D が意味を成さない）
      - 学生が train モードのとき、教師が eval でも fusion 層の dropout が効いて
        出力が非決定的になる

    という問題が起きる（2026-07-27 の実装検証で検出）。教師は `torch.no_grad()` で
    しか動かさず activation checkpoint は不要なので、インスタンス属性の `forward` を
    削除してクラス本来の forward に戻す。

    Returns:
        int: 解除した箇所数。
    """
    n = 0
    for sub in module.modules():
        f = sub.__dict__.get('forward', None)
        if isinstance(f, functools.partial) and \
                getattr(f.func, '__name__', '') == '_checkpointed_forward':
            del sub.__dict__['forward']
            n += 1
    return n


@MODELS.register_module()
class KDGroundingDINO(GroundingDINO):
    """特徴蒸留つき GroundingDINO.

    Args:
        kd (dict):
            targets (list[str]): {'img','txt','fus'} の部分集合。
                蒸留A=['img'] / B=['txt'] / D=['fus'] / E=['img','txt','fus']。
            loss (dict): FeatureDistillLoss の設定（form='l2' または 'cosine'）。
            loss_weight (float): λ（design §4.5 の校正で決めた値）。
            num_buffer_per_batch (int): 1 バッチ内のバッファ由来サンプル数（既定 2）。
        teacher_ckpt (str): 教師の重み。ドライバが前ドメインの last を渡す。
    """

    def __init__(self,
                 *args,
                 kd: Optional[dict] = None,
                 teacher_ckpt: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert kd is not None, 'kd の設定が必要です'
        targets = list(kd['targets'])
        assert targets and all(t in _TARGETS for t in targets), \
            f'targets は {_TARGETS} の部分集合であること: {targets}'
        self.kd_targets = targets
        self.kd_loss = MODELS.build(kd['loss'])
        self.kd_weight = float(kd['loss_weight'])
        self.num_buffer_per_batch = int(kd.get('num_buffer_per_batch', 2))
        self.teacher_ckpt = teacher_ckpt
        object.__setattr__(self, '_teacher', None)
        object.__setattr__(self, '_kd_capture', False)
        object.__setattr__(self, '_kd_cache', {})

    # -------------------------------------------------------------- 教師の構築
    def init_weights(self) -> None:
        # 早期 return は super() の**前**に置く。GroundingDINO.init_weights は
        # mmengine の _is_init ガードの外で xavier 初期化を無条件に行うため、
        # 2回目の呼び出しで text_feat_map / encoder / decoder が再初期化され、
        # θ^S != θ^T になってしまう（2026-07-27 の検証で検出）。
        if self._teacher is not None:
            return
        super().init_weights()
        assert self.teacher_ckpt is not None, \
            'teacher_ckpt が未指定です（ドライバが前ドメインの last を渡します）'
        teacher = copy.deepcopy(self)  # hook 登録前に複製する
        object.__setattr__(teacher, '_teacher', None)
        object.__setattr__(teacher, '_kd_capture', False)
        n = _strip_activation_checkpoint(teacher)
        if n:
            print_log(
                f'[KDGroundingDINO] 教師から activation checkpoint の '
                f'forward 差し替えを {n} 箇所解除しました', logger='current')
        ckpt = load_checkpoint(
            teacher, self.teacher_ckpt, map_location='cpu', logger='current')
        # load_checkpoint は strict=False。欠落があると学生の初期値が黙って
        # 残るため、教師の全キーが ckpt に存在することを明示的に確認する。
        _sd = ckpt.get('state_dict', ckpt)
        _missing = set(teacher.state_dict()) - set(_sd)
        assert not _missing, (
            '教師の重みに欠落があります（{} 件）: {} ... / ckpt={}'.format(
                len(_missing), sorted(_missing)[:3], self.teacher_ckpt))
        teacher.requires_grad_(False)
        teacher.eval()
        object.__setattr__(self, '_teacher', teacher)
        self._register_kd_hooks()

    def _register_kd_hooks(self) -> None:

        def lm_hook(module, args, output):
            if self._kd_capture:
                self._kd_cache['prompts'] = args[0]
                self._kd_cache['txt_mask'] = output['text_token_mask']

        def tfm_hook(module, args, output):
            if self._kd_capture:
                self._kd_cache['txt'] = output

        self.language_model.register_forward_hook(lm_hook)
        if self.text_feat_map is not None:
            self.text_feat_map.register_forward_hook(tfm_hook)

    def train(self, mode: bool = True):
        out = super().train(mode)
        if self._teacher is not None:
            self._teacher.eval()
        return out

    # ------------------------------------------------------ 学生の中間特徴の捕捉
    def extract_feat(self, batch_inputs: Tensor):
        feats = super().extract_feat(batch_inputs)
        if self._kd_capture:
            self._kd_cache['img'] = feats
        return feats

    def forward_encoder(self, *args, **kwargs) -> Dict:
        out = super().forward_encoder(*args, **kwargs)
        if self._kd_capture:
            self._kd_cache['mem'] = out['memory']
            self._kd_cache['mem_mask'] = out['memory_mask']
            self._kd_cache['mem_txt'] = out['memory_text']
        return out

    # ------------------------------------------------------------ 有効位置マスク
    @staticmethod
    def _mlvl_valid_masks(mlvl_feats: List[Tensor],
                          batch_data_samples: SampleList) -> List[Tensor]:
        """各スケールの有効位置マスク（True が実画像領域。design §4.2 の q_l）。

        `deformable_detr.py` の pre_transformer と同じ手順で作る。
        """
        input_h, input_w = batch_data_samples[0].batch_input_shape
        img_shapes = [s.img_shape for s in batch_data_samples]
        if all(s[0] == input_h and s[1] == input_w for s in img_shapes):
            return [
                feat.new_ones(
                    (feat.shape[0], *feat.shape[-2:]), dtype=torch.bool)
                for feat in mlvl_feats
            ]
        pad = mlvl_feats[0].new_ones((len(img_shapes), input_h, input_w))
        for i, (h, w) in enumerate(img_shapes):
            pad[i, :h, :w] = 0  # 0 が実画像領域
        masks = []
        for feat in mlvl_feats:
            m = F.interpolate(pad[None], size=feat.shape[-2:])[0]
            masks.append(m.to(torch.bool).logical_not())
        return masks

    # ------------------------------------------------------------ 教師の前向き
    @torch.no_grad()
    def _teacher_features(self, batch_inputs: Tensor, prompts: List[str],
                          batch_data_samples: SampleList) -> Dict:
        t = self._teacher
        tg = self.kd_targets
        need_fus = 'fus' in tg
        need_img = need_fus or 'img' in tg      # 融合後は画像特徴を入力に取る
        need_txt = need_fus or 'txt' in tg      # 同上（テキスト側）
        feats: Dict = {}
        text_dict = None
        if need_txt:
            text_dict = t.language_model(prompts)
            if t.text_feat_map is not None:
                text_dict['embedded'] = t.text_feat_map(text_dict['embedded'])
            feats['txt'] = text_dict['embedded']
        visual_features = None
        if need_img:
            visual_features = t.extract_feat(batch_inputs)
            feats['img'] = visual_features
        if need_fus:
            enc_in, _ = t.pre_transformer(visual_features, batch_data_samples)
            enc = t.forward_encoder(**enc_in, text_dict=text_dict)
            feats['mem'] = enc['memory']
            feats['mem_txt'] = enc['memory_text']
        return feats

    # ---------------------------------------------------------------- 蒸留損失
    def _kd_losses(self, s: Dict, t: Dict, img_masks: List[Tensor],
                   buf: slice, diag: Optional[dict] = None) -> Tensor:
        """バッファ由来サンプルで蒸留損失を計算し、対象間で平均する（蒸留E は3項平均）。

        Args:
            diag: 渡すと、対象ごとの重み付け前の値を記録する（学習ログ用の診断値）。
        """

        def f32(x):
            return x.float()

        terms = []
        if 'img' in self.kd_targets:
            v = self.kd_loss.map_loss([f32(x[buf]) for x in s['img']],
                                      [f32(x[buf]) for x in t['img']],
                                      [m[buf] for m in img_masks]).mean()
            terms.append(v)
            if diag is not None:
                diag['kd_img'] = v.detach()
        if 'txt' in self.kd_targets:
            v = self.kd_loss.seq_loss(
                f32(s['txt'][buf]), f32(t['txt'][buf]),
                s['txt_mask'][buf]).mean()
            terms.append(v)
            if diag is not None:
                diag['kd_txt'] = v.detach()
        if 'fus' in self.kd_targets:
            # memory_mask は True がパディング。有効位置はその否定。
            # バッチ内の全画像が同一形状のとき pre_transformer は mask を作らず
            # memory_mask=None を返す（deformable_detr.py:159-160, 208-211）。
            # その場合は全位置が有効。
            mm = s['mem_mask']
            mem_valid = None if mm is None else mm[buf].logical_not()
            l_img = self.kd_loss.seq_loss(
                f32(s['mem'][buf]), f32(t['mem'][buf]), mem_valid)
            l_txt = self.kd_loss.seq_loss(
                f32(s['mem_txt'][buf]), f32(t['mem_txt'][buf]),
                s['txt_mask'][buf])
            v = (0.5 * (l_img + l_txt)).mean()
            terms.append(v)
            if diag is not None:
                diag['kd_fus'] = v.detach()
                # 融合後は画像側とテキスト側で桁が大きく違うため内訳も残す
                diag['kd_fus_img'] = l_img.mean().detach()
                diag['kd_fus_txt'] = l_txt.mean().detach()
        return torch.stack(terms).mean()

    # ------------------------------------------------------ バッファ位置の検査
    @staticmethod
    def _assert_buffer_slice(batch_data_samples: SampleList, bs: int,
                             n: int) -> None:
        """バッチ末尾 n 件が本当にバッファ由来かを毎イテレーション検査する。

        位置ベースの特定は `MultiSourceSampler` がソース順に添字を連結すること
        （`multi_source_sampler.py:123-134`）に依存する。この前提は次の場合に
        **無警告で崩れる**（2026-07-27 の検証で指摘）。

          - `train_dataloader.batch_sampler` に `AspectRatioBatchSampler` が
            入る（継承元の事前学習 config は設定している。現在は `_delete_=True`
            で消えているが、復活すると 1 バッチが複数 sampler バッチの断片で
            構成される）
          - config の `datasets=[現在, 参照, 過去]` の並びを変える

        現在ドメインとバッファは必ず別のディレクトリに置かれる（バッファは
        Objects365 か**過去**ドメインで、現在ドメインとは一致しない）ため、
        画像の親ディレクトリが重なっていないことで検査できる。
        """
        if n <= 0 or bs <= n:
            return
        dirs = [osp.dirname(s.img_path) for s in batch_data_samples]
        cur, bufd = set(dirs[:bs - n]), set(dirs[bs - n:])
        overlap = cur & bufd
        assert not overlap, (
            'バッファ位置の前提が崩れています（末尾 {} 件がバッファ由来では '
            'ありません）。現在ドメインとバッファが同じディレクトリを共有: {}。'
            'train_dataloader.batch_sampler や datasets の並びを確認してください。'
        ).format(n, sorted(overlap))

    # -------------------------------------------------------------------- loss
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        object.__setattr__(self, '_kd_cache', {})
        object.__setattr__(self, '_kd_capture', True)
        try:
            losses = super().loss(batch_inputs, batch_data_samples)
        except BaseException:
            # 例外時もキャッシュを落とす（学習グラフを掴んだまま残さない）
            object.__setattr__(self, '_kd_cache', {})
            raise
        finally:
            object.__setattr__(self, '_kd_capture', False)

        s = self._kd_cache
        for key in ('img', 'txt', 'txt_mask', 'prompts', 'mem', 'mem_mask',
                    'mem_txt'):
            assert key in s, f'学生の中間特徴 {key} を捕捉できていません'

        bs = batch_inputs.shape[0]
        n = min(self.num_buffer_per_batch, bs)
        buf = slice(bs - n, bs)
        self._assert_buffer_slice(batch_data_samples, bs, n)

        t = self._teacher_features(batch_inputs, s['prompts'],
                                   batch_data_samples)
        img_masks = self._mlvl_valid_masks(s['img'], batch_data_samples)
        # diag には重み付け前の対象ごとの値が入る。キーに 'loss' を含めないことで
        # mmengine の parse_losses の合計対象から外れ、ログにのみ残る。
        diag: Dict[str, Tensor] = {}
        kd = self._kd_losses(s, t, img_masks, buf, diag=diag)
        losses['loss_kd'] = self.kd_weight * kd
        losses.update(diag)
        object.__setattr__(self, '_kd_cache', {})
        return losses
