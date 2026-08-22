"""exp_051 実行前検証（design.md §4）。

使い方:
    python experiments/exp_051/check_exp051_setup.py           # 静的（GPU 不要）
    CUDA_VISIBLE_DEVICES=0 python experiments/exp_051/check_exp051_setup.py --step
        # + 実データ 1 step × 3 条件（GPU 1 枚・要 θ0）

検証項目
  1. 18 本の config が build でき、exp_043 kdE との差分が kd.targets と ckpt 保存
     方針だけであること（機械的 diff）。バッチ構成 [4,2,2]×8（t=1 は [4,4]）の継承確認。
  2. --cfg-options の到達性（model.teacher_ckpt。誤記の負テスト込み）。
  3. （--step）各条件の実データ 1 step（t=1 相当・[現在4, 汎用4] バッチ）で
     loss・loss_kd が有限、診断値が対象と一致（kdA→kd_img のみ / kdB→kd_txt のみ /
     kdD→kd_fus のみ）、蒸留がバッファ 4 枚限定であること（バッファ側サンプルの
     差し替えで loss_kd が変化し、現在ドメイン側の差し替えでは変化しない）。
"""
import copy
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from mmengine.config import Config  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402

ORDER = ['underwater', 'electromagnetic', 'videogames',
         'aerial', 'microscopic', 'documents']
CONDS = {'kdA': ['img'], 'kdB': ['txt'], 'kdD': ['fus']}
DIAG = {'kdA': 'kd_img', 'kdB': 'kd_txt', 'kdD': 'kd_fus'}
THETA0 = os.path.join(
    ROOT, 'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det'
    '_20231204_095047-b448804b.pth')
results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def main():
    do_step = '--step' in sys.argv
    init_default_scope('mmdet')

    # --- 1. build と kdE との機械的 diff -----------------------------------
    bad, n = [], 0
    for cond, targets in CONDS.items():
        for d in ORDER:
            try:
                c = Config.fromfile(f'{HERE}/configs/{cond}_{d}.py')
                e = Config.fromfile(f'experiments/exp_043/configs/kdE_{d}.py')
                for key in ('train_dataloader', 'optim_wrapper', 'param_scheduler',
                            'train_cfg', 'randomness'):
                    if _plain(c.get(key)) != _plain(e.get(key)):
                        bad.append(f'{cond}_{d}: {key} 不一致')
                # model の差分は targets のみ
                cm, em = copy.deepcopy(_plain(c.model)), copy.deepcopy(_plain(e.model))
                assert cm['kd'].pop('targets') == targets
                em['kd'].pop('targets')
                if cm != em:
                    bad.append(f'{cond}_{d}: model に targets 以外の差分')
                ck = c.default_hooks.checkpoint
                assert ck.interval == 20 and ck.save_optimizer is False, 'ckpt 設定'
                bs = c.train_dataloader.batch_size
                sr = c.train_dataloader.sampler.source_ratio
                assert bs == 8 and sr in ([4, 4], [4, 2, 2]), f'バッチ {bs}/{sr}'
                n += 1
            except Exception as ex:
                bad.append(f'{cond}_{d}: {ex}')
    check(1, '18 本 build・kdE との diff（targets と ckpt 方針のみ）・バッチ継承',
          not bad and n == 18, f'{n}/18 / 問題 {bad[:3]}')

    # --- 2. --cfg-options 到達性 --------------------------------------------
    c = Config.fromfile(f'{HERE}/configs/kdA_underwater.py')
    c.merge_from_dict({'model.teacher_ckpt': '/tmp/x.pth'})
    ok2 = c.model.teacher_ckpt == '/tmp/x.pth'
    c2 = Config.fromfile(f'{HERE}/configs/kdA_underwater.py')
    try:
        c2.merge_from_dict({'model.teacher_ckp': '/tmp/x.pth'})
        neg = c2.model.teacher_ckpt is None
    except Exception:
        neg = True
    check(2, 'model.teacher_ckpt の到達性（負テスト込み）', ok2 and neg)

    if not do_step:
        print(f'\n{sum(results)}/{len(results)} OK（--step で 3 を実行）')
        sys.exit(0 if all(results) else 1)

    import torch
    from mmengine.dataset import pseudo_collate
    from mmengine.utils import import_modules_from_strings
    from mmengine.runner.checkpoint import load_checkpoint
    from mmdet.registry import MODELS, DATASETS

    assert os.path.isfile(THETA0)
    for cond in CONDS:
        cfg = Config.fromfile(f'{HERE}/configs/{cond}_underwater.py')
        import_modules_from_strings(**cfg.custom_imports)
        cfg.model.teacher_ckpt = THETA0
        model = MODELS.build(cfg.model)
        # 実運用（Runner）と同じ順序: init_weights（教師構築）→ 学生に θ0 を load。
        # 逆順だと GroundingDINO.init_weights の xavier 再初期化が学生の θ0 を
        # 上書きし、学生≠教師の人工状態で loss_kd が過大に出る（exp_048 と同じ罠）
        model = model.cuda()
        model.init_weights()
        load_checkpoint(model, THETA0, map_location='cpu')
        model.train()
        ds = DATASETS.build(cfg.train_dataloader.dataset)
        # バッチ並び [現在4, 汎用4]（t=1）を再現
        n0 = len(ds.datasets[0])
        idx = [0, 1, 2, 3, n0, n0 + 1, n0 + 2, n0 + 3]
        batch = pseudo_collate([ds[j] for j in idx])
        data = model.data_preprocessor(batch, True)

        # _kd_losses に渡る buf スライスを捕捉して「末尾 4 枚限定」を直接検証する
        # （画像差し替え方式はパディング形状差で比較が汚れるため使わない。
        #   バッファ位置の妥当性は実装内蔵の _assert_buffer_slice が毎 iteration 検査）
        cap = {}
        _orig = model._kd_losses
        def _wrap(s_, t_, m_, buf, diag=None, _o=_orig, _c=cap):
            _c['buf'] = buf
            return _o(s_, t_, m_, buf, diag=diag)
        model._kd_losses = _wrap
        losses = model.loss(data['inputs'], data['data_samples'])
        kd1 = float(losses['loss_kd'].detach())
        diag_keys = [k for k in losses if k.startswith('kd_') and 'fus_' not in k]
        total = sum(float(v.mean().detach()) for k, v in losses.items()
                    if 'loss' in k and hasattr(v, 'mean'))
        ok = (total == total and abs(total) != float('inf')
              and diag_keys == [DIAG[cond]]
              and cap.get('buf') == slice(4, 8))
        check(3, f'{cond}: 1 step（loss 有限・診断 {DIAG[cond]} のみ・buf=slice(4,8)）',
              ok, f'loss={total:.2f} loss_kd={kd1:.3f} buf={cap.get("buf")}')
        del model
        import torch as _t
        _t.cuda.empty_cache()

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
