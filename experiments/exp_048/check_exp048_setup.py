"""exp_048 実行前検証（design.md §4.2）。

使い方:
    python experiments/exp_048/check_exp048_setup.py          # config のみ（GPU 不要）
    python experiments/exp_048/check_exp048_setup.py --step   # + 実データ 1 step（GPU 1 枚）

検証項目
  1. 12 本の config が build でき、model 以外（train_dataloader・optimizer・schedule・
     randomness）が継承元（kdEonly: replayfree / kdEall: replay）と一致すること。
     model の差分が「KDGroundingDINO ＋ kd 設定」だけであること。
  2. --cfg-options の到達性（model.teacher_ckpt。誤記の負テスト込み）。
  3. （--step）実データ 1 step で loss・loss_kd が有限であること。
     **蒸留が全サンプルに掛かること**: 先頭サンプル（現在ドメイン）の画像を
     別サンプルの画像に差し替えると loss_kd が変わる（バッファ限定なら変わらない）。
     VRAM 実測。
"""
import copy
import os
import sys

from mmengine.config import Config
from mmengine.registry import init_default_scope

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CFG_DIR = os.path.join(HERE, 'configs')

FULL_ORDER = ['underwater', 'electromagnetic', 'videogames',
              'aerial', 'microscopic', 'documents']
THETA0 = os.path.join(
    ROOT, 'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det'
    '_20231204_095047-b448804b.pth')
results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


def base_for(cond, d):
    front = d in FULL_ORDER[:3]
    if cond == 'kdEonly':
        return (f'experiments/exp_024/configs/fullft_replayfree_{d}.py' if front
                else f'experiments/exp_040/configs/replayfree_{d}.py')
    return (f'experiments/exp_023/configs/fullft_replay_{d}.py' if front
            else f'experiments/exp_040/configs/replay_{d}.py')


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


EXPECT_N = {'kdEonly': 4, 'kdEall': 6}

if __name__ == '__main__':
    do_step = '--step' in sys.argv
    init_default_scope('mmdet')

    # --- 1. build・継承の機械的 diff -----------------------------------------
    bad = []
    n_ok = 0
    for cond in ('kdEonly', 'kdEall'):
        for d in FULL_ORDER:
            p = os.path.join(CFG_DIR, f'{cond}_{d}.py')
            try:
                c = Config.fromfile(p)
                b = Config.fromfile(os.path.join(ROOT, base_for(cond, d)))
            except Exception as e:
                bad.append(f'{cond}_{d}: {type(e).__name__}: {e}')
                continue
            # model 以外の一致
            for key in ('train_dataloader', 'optim_wrapper', 'param_scheduler',
                        'train_cfg', 'randomness', 'default_hooks'):
                if _plain(c.get(key)) != _plain(b.get(key)):
                    bad.append(f'{cond}_{d}: {key} が継承元と不一致')
            # model の差分
            m = c.model
            if m.get('type') != 'KDGroundingDINO':
                bad.append(f'{cond}_{d}: type={m.get("type")}')
            kd = m.get('kd', {})
            if kd.get('num_buffer_per_batch') != EXPECT_N[cond]:
                bad.append(f'{cond}_{d}: num_buffer_per_batch='
                           f'{kd.get("num_buffer_per_batch")}')
            if kd.get('loss_weight') != 10.0 or \
                    sorted(kd.get('targets', [])) != ['fus', 'img', 'txt']:
                bad.append(f'{cond}_{d}: kd 設定不一致 {kd}')
            bs = c.train_dataloader.get('batch_size')
            if bs != EXPECT_N[cond]:
                bad.append(f'{cond}_{d}: batch_size={bs} != {EXPECT_N[cond]}')
            n_ok += 1
    check(1, '12 本 build・継承 diff・kd 設定', not bad and n_ok == 12,
          f'{n_ok}/12 build / 問題 {len(bad)}: {bad[:4]}')

    # --- 2. --cfg-options の到達性 -------------------------------------------
    c = Config.fromfile(os.path.join(CFG_DIR, 'kdEonly_underwater.py'))
    c.merge_from_dict({'model.teacher_ckpt': '/tmp/x.pth'})
    ok2 = c.model.teacher_ckpt == '/tmp/x.pth'
    c2 = Config.fromfile(os.path.join(CFG_DIR, 'kdEonly_underwater.py'))
    try:
        c2.merge_from_dict({'model.teacher_ckp': '/tmp/x.pth'})  # 誤記
        neg = c2.model.get('teacher_ckp') == '/tmp/x.pth' and \
            c2.model.teacher_ckpt is None  # 誤記は別キーに落ち teacher_ckpt に届かない
    except Exception:
        neg = True
    check(2, 'model.teacher_ckpt の到達性（負テスト込み）', ok2 and neg)

    # --- 3. 実データ 1 step（--step のとき） ---------------------------------
    if do_step:
        import torch
        from mmengine.registry import MODELS, DATASETS
        from mmengine.dataset import pseudo_collate
        from mmengine.utils import import_modules_from_strings
        from mmengine.runner.checkpoint import load_checkpoint

        assert os.path.isfile(THETA0), f'θ0 が無い: {THETA0}'
        dev = 'cuda:0'
        for cond in ('kdEonly', 'kdEall'):
            n = EXPECT_N[cond]
            cfg = Config.fromfile(os.path.join(CFG_DIR, f'{cond}_underwater.py'))
            if cfg.get('custom_imports'):
                import_modules_from_strings(**cfg['custom_imports'])
            cfg.model.teacher_ckpt = THETA0
            model = MODELS.build(cfg.model)
            # 実運用（Runner）と同じ順序にする: init_weights → load_from。
            #   - 教師は init_weights 内で self の deepcopy に teacher_ckpt を
            #     読み込んで作られる（deepcopy 前に GPU へ載せる必要がある）
            #   - 学生の θ0 は init_weights の**後**に読む。先に読むと
            #     GroundingDINO.init_weights の無条件 xavier 初期化が学生だけを
            #     上書きし、学生≠教師の人工状態になる（2026-08-20 の検証で
            #     この順序ミスにより loss_kd が 678/882 と過大に出た。正しい順序
            #     では t=1 の学生=教師=θ0 で dropout 差のみが残る）
            model = model.to(dev)
            model.init_weights()
            load_checkpoint(model, THETA0, map_location='cpu')
            model = model.train()

            dcfg = cfg.train_dataloader.dataset
            ds = DATASETS.build(dcfg)
            # kdEall は ConcatDataset [現在, 汎用]。バッチ並び [現在4, 汎用2] を再現する
            if cond == 'kdEall':
                idx = [0, 1, 2, 3, len(ds.datasets[0]), len(ds.datasets[0]) + 1]
            else:
                idx = list(range(n))
            batch = pseudo_collate([ds[j] for j in idx])
            data = model.data_preprocessor(batch, True)

            def kd_of(d):
                with torch.no_grad():
                    pass  # 勾配は要る（loss 経路）ので no_grad にしない
                losses = model.loss(d['inputs'], d['data_samples'])
                return {k: float(v.detach()) for k, v in losses.items()
                        if k in ('loss_kd',)}, losses

            kd1, losses = kd_of(data)
            total = sum(float(v.mean().detach()) if hasattr(v, 'mean') else 0.
                        for k, v in losses.items() if 'loss' in k)
            fin = all(map(lambda x: x == x and abs(x) != float('inf'),
                          [total, kd1['loss_kd']]))
            # 先頭サンプル（現在ドメイン）の画像を最後のサンプルの画像で差し替え
            data2 = copy.deepcopy(data)
            data2['inputs'][0] = data['inputs'][-1].clone()
            data2['data_samples'][0] = copy.deepcopy(data['data_samples'][-1])
            kd2, _ = kd_of(data2)
            changed = abs(kd2['loss_kd'] - kd1['loss_kd']) > 1e-6
            # t=1 相当（学生=教師=θ0）の loss_kd は dropout 差のみ。
            # 参照: exp_035 t=1 の最初のログ（50 iter 更新後）で 8.3。
            # 未更新の本検証では λ=10 込みで 20 を大きく超えないはず。
            sane = kd1['loss_kd'] < 20.0
            mem = torch.cuda.max_memory_allocated() / 2**20
            check(3, f'{cond}: 1 step 有限・全サンプル蒸留・VRAM',
                  fin and changed and sane,
                  f'loss={total:.3f} loss_kd={kd1["loss_kd"]:.3f} -> '
                  f'先頭差替後 {kd2["loss_kd"]:.3f}（変化 {changed}） '
                  f'VRAM {mem:.0f} MiB')
            del model
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
