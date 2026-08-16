"""InfLoRA の入力共分散収集の共通処理（inflora_prepare / inflora_update_memory が使う）。

- サンプルは上限枚数まで、超える場合は seed 固定の無作為抽出（EWC の Fisher と同じ規約。
  exp_042 design.md §2.3、2026-08-15 確定）
- 公式実装は学習と同じデータを通常の forward で流して cur_matrix を蓄積する。
  本実装は学習と同じ損失経路（model.loss）を no_grad で流す。dropout は
  loss_mode_dropout_off() で無効化する（下記）。
"""
import random

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.utils import import_modules_from_strings

from mmdet.registry import DATASETS, MODELS


def seed_everything(seed):
    """全 RNG を固定する（再現性）。

    サンプルの選択（random.Random(seed)）だけでなく、データパイプラインの拡張
    （RandomFlip / RandomChoiceResize 等）がグローバル RNG を使うため、これを
    固定しないと同じ引数でも共分散・Fisher が実行ごとに変わる（公開実装として
    再現性を保証するための処置。2026-08-15 監査で追加）。
    CUDA カーネルの原子加算による ~1e-7 相対の非決定性は残る。
    """
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_mode_dropout_off(model):
    """dropout を全て無効にしつつ loss 経路を保つモード。

    `model.train()` のまま dropout モジュールだけ eval にする方式は、
    `F.dropout(..., training=self.training)` を使う層（DecomposedMHA、融合層の
    BiMultiHeadAttention、nn.MultiheadAttention）を取りこぼす（2026-08-15 実測:
    loss が呼び出しごとに 3〜5% 揺れた）。代わりに `model.eval()` で全 dropout
    （モジュール型・関数型とも）を確実に切り、損失経路の構造分岐
    （forward_transformer の学習時専用出力の生成）に使われる**検出器本体の
    training フラグだけ**を立て直す。この状態で loss はビット単位で決定的になる
    ことを実測確認済み。
    """
    model.eval()
    model.training = True


def build_and_collect(config_path, ckpt, max_samples, seed, device,
                      design_path=None):
    """config からモデルを組み、ckpt を読み、共分散を収集して返す。

    Returns:
        (model, covs, n_used): covs は {層名: (d_in, d_in) CPU tensor}
    """
    seed_everything(seed)
    cfg = Config.fromfile(config_path)
    init_default_scope('mmdet')
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg['custom_imports'])
    assert cfg.model['type'] == 'GroundingDINOInfLoRA', \
        f'GroundingDINOInfLoRA の config を渡すこと: {cfg.model["type"]}'
    cfg.model['inflora'] = dict(design_path=design_path)

    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt, map_location='cpu')
    model = model.to(device)
    loss_mode_dropout_off(model)

    dataset = DATASETS.build(cfg.train_dataloader['dataset'])
    n = len(dataset)
    if n > max_samples:
        idx = sorted(random.Random(seed).sample(range(n), max_samples))
    else:
        idx = list(range(n))
    print(f'[InfLoRA] 共分散収集: データ {n} 枚中 {len(idx)} 枚 '
          f'(max={max_samples}, seed={seed}, dropout 無効・決定的)')

    model.set_input_collection(True)
    with torch.no_grad():
        for j, i in enumerate(idx):
            batch = pseudo_collate([dataset[i]])
            data = model.data_preprocessor(batch, True)
            model.loss(data['inputs'], data['data_samples'])
            if (j + 1) % 100 == 0:
                print(f'[InfLoRA] {j + 1}/{len(idx)}')
    model.set_input_collection(False)
    covs = model.pop_covariances()
    return model, covs, len(idx)
