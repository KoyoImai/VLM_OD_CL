"""exp_040 実行前検証（design.md §6.3）。本環境で実施し、クラスタへは push して持ち込む。

使い方:
    python experiments/exp_040/check_exp040_setup.py          # config のみ（GPU 不要）
    python experiments/exp_040/check_exp040_setup.py --build  # model の build も行う

全項目 OK でなければクラスタへ投入しない。
クラスタのマスターノードには mmdet の実行環境が無いため、本スクリプトは本環境で走らせる。

検証項目
  1. 9本の config が build でき、継承元が前半の base と一致
  2. 学習設定が前半（exp_024 / exp_027 / exp_035）と一致
     （20 epoch / milestones[15] / lr 1e-4 / wd 1e-4 / clip 0.1 / batch / seed 0 / num_classes 256）
  3. データ経路（ODVGDataset・サンプラ・source_ratio・現在ドメイン）
  4. 過去プールが累積し、Objects365 は独立ソースであること（design.md §2.4）
  5. 蒸留設定（kdE: targets / L2 / λ=10 / num_buffer_per_batch=2 / teacher_ckpt）
  6. 初期パラメータ3本と参照するデータ・バッファが実在すること
  7. model dict が前半の同条件 config と一致（意図した差分を除く）
     — 個別項目の列挙では拾えない取りこぼしを機械的に検出する（exp_039 の教訓）
  8. 本環境固有パスの扱い（クラスタでは bind で同一パスに解決される）
  9. （--build 時）model が build でき、学習対象が条件どおり
 10. ドライバの --cfg-options（load_from / model.teacher_ckpt）が狙った場所に届く
"""
import os
import sys

from mmengine.config import Config
from mmengine.registry import init_default_scope

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, 'configs')
ROOT = os.path.join(HERE, '..', '..')

FULL_ORDER = ['underwater', 'electromagnetic', 'videogames',
              'aerial', 'microscopic', 'documents']
LATER = FULL_ORDER[3:]
CONDS = ['replayfree', 'replay', 'kdE']
EXPECT_BATCH = {'replayfree': 4, 'replay': 6, 'kdE': 6}
EXPECT_SAMPLER = {'replayfree': 'DefaultSampler',
                  'replay': 'CurrentEpochMultiSourceSampler',
                  'kdE': 'CurrentEpochMultiSourceSampler'}
# 前半の同条件 config（model dict の突き合わせ相手）
REF = {'replayfree': 'experiments/exp_024/configs/fullft_replayfree_videogames.py',
       'replay': 'experiments/exp_023/configs/fullft_replay_videogames.py',
       'kdE': 'experiments/exp_035/configs/kdE_condA_l2w100_videogames.py'}
INIT = {'replayfree': 'experiments/exp_024/fullft_replayfree_videogames_work_dir/epoch_20.pth',
        'replay': 'experiments/exp_027/fullft_replay_videogames_work_dir/epoch_20.pth',
        'kdE': 'experiments/exp_035/kdE_condA_l2w100_videogames_work_dir/epoch_20.pth'}

results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


def all_cfgs():
    for c in CONDS:
        for d in LATER:
            yield c, d, os.path.join(CFG_DIR, f'{c}_{d}.py')


def _flat(d, pre=''):
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else []):
        key = f'{pre}.{k}' if pre else k
        if isinstance(v, dict):
            out.update(_flat(v, key))
        else:
            out[key] = v
    return out


if __name__ == '__main__':
    do_build = '--build' in sys.argv
    init_default_scope('mmdet')

    # --- 1. build と継承元 --------------------------------------------------
    cfgs, bad = {}, []
    for c, d, p in all_cfgs():
        if not os.path.exists(p):
            bad.append(f'{c}_{d}: なし'); continue
        try:
            cfgs[(c, d)] = Config.fromfile(p)
        except Exception as e:
            bad.append(f'{c}_{d}: {type(e).__name__}'); continue
        src = open(p).read()
        want = ('fullft_replayfree_base' if c == 'replayfree'
                else 'fullft_replay_base')
        if want not in src:
            bad.append(f'{c}_{d}: 継承元が {want} でない')
    check(1, '9本の config が build でき継承元が前半の base',
          len(cfgs) == 9 and not bad,
          f'{len(cfgs)}/9' + (f' / 問題 {bad[:3]}' if bad else ''))

    # --- 2. 学習設定 --------------------------------------------------------
    bad = []
    for (c, d), cf in cfgs.items():
        tag = f'{c}_{d}'
        for label, got, want in (
                ('max_epochs', cf.train_cfg.get('max_epochs'), 20),
                ('loop', cf.train_cfg.get('type'), 'EpochBasedTrainLoop'),
                ('milestones', list(cf.param_scheduler[0]['milestones']), [15]),
                ('lr', cf.optim_wrapper['optimizer']['lr'], 0.0001),
                ('wd', cf.optim_wrapper['optimizer']['weight_decay'], 0.0001),
                ('clip', cf.optim_wrapper['clip_grad']['max_norm'], 0.1),
                ('batch', cf.train_dataloader['batch_size'], EXPECT_BATCH[c]),
                ('seed', cf.randomness['seed'], 0),
                ('num_classes', cf.model['bbox_head']['num_classes'], 256)):
            if got != want:
                bad.append(f'{tag}: {label}={got}(期待 {want})')
    check(2, '学習設定が前半と一致', not bad,
          f'9本 × 9項目 / 不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 3. データ経路 ------------------------------------------------------
    bad = []
    for (c, d), cf in cfgs.items():
        tag = f'{c}_{d}'
        tl = cf.train_dataloader
        if tl['sampler']['type'] != EXPECT_SAMPLER[c]:
            bad.append(f'{tag}: sampler={tl["sampler"]["type"]}')
        ds = tl['dataset']
        cur = ds if c == 'replayfree' else ds['datasets'][0]
        if cur['type'] != 'ODVGDataset':
            bad.append(f'{tag}: current type={cur["type"]}')
        if cur['ann_file'] != f'{d}_train_od.json':
            bad.append(f'{tag}: ann_file={cur["ann_file"]}')
        if not cur['data_root'].endswith(f'/{d}/'):
            bad.append(f'{tag}: data_root={cur["data_root"]}')
        if c != 'replayfree':
            if list(tl['sampler'].get('source_ratio', [])) != [4, 1, 1]:
                bad.append(f'{tag}: source_ratio={tl["sampler"].get("source_ratio")}')
            if len(ds['datasets']) != 3:
                bad.append(f'{tag}: ソース数={len(ds["datasets"])}(期待 3)')
    check(3, 'データ経路が期待どおり', not bad,
          f'不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 4. 過去プールの累積と Objects365 の独立性 --------------------------
    bad, info = [], []
    for (c, d), cf in cfgs.items():
        if c == 'replayfree':
            continue
        tag = f'{c}_{d}'
        t = 4 + LATER.index(d)
        want_past = FULL_ORDER[:t - 1]
        ds = cf.train_dataloader['dataset']['datasets']
        ref, pool = ds[1], ds[2]
        # 参照 = Objects365 のみ、単一 ODVGDataset
        if ref['type'] != 'ODVGDataset' or 'o365' not in ref['ann_file']:
            bad.append(f'{tag}: 参照が Objects365 でない')
        # 過去プール = 学習済みドメインの累積、Objects365 を含まない
        if pool['type'] != 'ConcatDataset':
            bad.append(f'{tag}: 過去プールが ConcatDataset でない')
        else:
            got = [os.path.basename(x['ann_file']).replace('_500.odvg.json', '')
                   for x in pool['datasets']]
            if got != want_past:
                bad.append(f'{tag}: 過去プール={got}(期待 {want_past})')
            if any('o365' in x['ann_file'] for x in pool['datasets']):
                bad.append(f'{tag}: 過去プールに Objects365 が混入')
            info.append(f'{tag}:{len(got)}')
    check(4, '過去プールが累積し Objects365 は独立ソース', not bad,
          f'{" ".join(info)}' + (f' / 不一致 {bad[:3]}' if bad else ''))

    # --- 5. 蒸留設定 --------------------------------------------------------
    bad = []
    for (c, d), cf in cfgs.items():
        tag = f'{c}_{d}'
        if c == 'kdE':
            if cf.model['type'] != 'KDGroundingDINO':
                bad.append(f'{tag}: type={cf.model["type"]}')
            kd = cf.model.get('kd', {})
            if list(kd.get('targets', [])) != ['img', 'txt', 'fus']:
                bad.append(f'{tag}: targets={kd.get("targets")}')
            if kd.get('loss', {}).get('form') != 'l2':
                bad.append(f'{tag}: loss={kd.get("loss")}')
            if kd.get('loss_weight') != 10.0:
                bad.append(f'{tag}: loss_weight={kd.get("loss_weight")}(期待 10.0)')
            if kd.get('num_buffer_per_batch') != 2:
                bad.append(f'{tag}: num_buffer_per_batch={kd.get("num_buffer_per_batch")}')
            if 'teacher_ckpt' not in cf.model:
                bad.append(f'{tag}: teacher_ckpt が無い')
        else:
            if cf.model['type'] != 'GroundingDINO':
                bad.append(f'{tag}: type={cf.model["type"]}(期待 GroundingDINO)')
    check(5, '蒸留設定（kdE）と検出器の型', not bad,
          f'不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 6. 実在確認 --------------------------------------------------------
    bad = []
    for c, p in INIT.items():
        if not os.path.exists(os.path.join(ROOT, p)):
            bad.append(f'初期パラメータ {c}: {p}')
    for (c, d), cf in cfgs.items():
        ds = cf.train_dataloader['dataset']
        cur = ds if c == 'replayfree' else ds['datasets'][0]
        for f in (cur['data_root'] + cur['ann_file'],
                  cur['data_root'] + cur['label_map_file']):
            if not os.path.exists(f):
                bad.append(f'{c}_{d}: {f}')
        if c == 'replayfree':
            continue
        for x in ds['datasets'][2]['datasets'] + [ds['datasets'][1]]:
            if not os.path.exists(x['ann_file']):
                bad.append(f'{c}_{d}: {x["ann_file"]}')
    check(6, '初期パラメータ・データ・バッファが実在', not bad,
          f'欠落 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 7. 前半 config との model dict 突き合わせ --------------------------
    # 継承元が同じなので model は完全一致するはず。データ以外の差分が出たら
    # overlay の書き起こしミスである（exp_039 で 2 件続けて取りこぼしたため追加）。
    bad = []
    for (c, d), cf in cfgs.items():
        ref = Config.fromfile(os.path.join(ROOT, REF[c]))
        fa, fb = _flat(ref.model), _flat(cf.model)
        for k in sorted(set(fa) | set(fb)):
            if fa.get(k) != fb.get(k):
                bad.append(f'{c}_{d}: {k} 前半={fa.get(k)!r} exp040={fb.get(k)!r}')
    check(7, 'model dict が前半の同条件 config と一致', not bad,
          f'9本 / 想定外の差分 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 8. パスの扱い ------------------------------------------------------
    # クラスタでは Singularity の bind で /workspace/kouyou/... に解決されるため、
    # このパス表記は前半（exp_023 等）と同じで正しい。他の環境固有パスが無いことを見る。
    bad = []
    for c, d, p in all_cfgs():
        if not os.path.exists(p):
            continue
        for tok in ('/root/', '/home/kouyou/', '/local_cache/'):
            if tok in open(p).read():
                bad.append(f'{c}_{d}: {tok}')
    check(8, 'クラスタで解決できないパスを含まない', not bad,
          f'検出 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 10. ドライバの --cfg-options が実際に効くか ------------------------
    # `teacher_ckpt=` のようにキー名を誤ると mmengine はトップレベルに新しいキーを
    # 作るだけでエラーにならず、学習開始まで気づけない（2026-08-14 に実際に踏んだ）。
    # ドライバが渡す文字列を config に適用して、狙った場所に入るかを検査する。
    import re as _re
    drv = open(os.path.join(HERE, 'run_sequential.sh')).read()
    bad = []
    # ドライバが load_from / model.teacher_ckpt をどう書いているか
    if 'load_from=$prev' not in drv and 'load_from=\$prev' not in drv:
        bad.append('load_from の受け渡しが見当たらない')
    if 'model.teacher_ckpt=' not in drv:
        bad.append('teacher_ckpt が model. 付きで渡されていない')
    # 実際に適用して確かめる
    cf = Config.fromfile(os.path.join(CFG_DIR, 'kdE_aerial.py'))
    cf.merge_from_dict({'load_from': 'X.pth', 'model.teacher_ckpt': 'Y.pth'})
    if cf.get('load_from') != 'X.pth':
        bad.append(f'load_from={cf.get("load_from")}')
    if cf.model.get('teacher_ckpt') != 'Y.pth':
        bad.append(f'model.teacher_ckpt={cf.model.get("teacher_ckpt")}')
    # 誤った書式ではモデルに届かないことも確認（検査が意味を持つことの裏取り）
    cf2 = Config.fromfile(os.path.join(CFG_DIR, 'kdE_aerial.py'))
    cf2.merge_from_dict({'teacher_ckpt': 'Z.pth'})
    if cf2.model.get('teacher_ckpt') is not None:
        bad.append('model. 無しでも届いてしまう（検査が無意味）')
    check(10, 'ドライバの --cfg-options が狙った場所に届く', not bad,
          'load_from / model.teacher_ckpt とも OK' if not bad else f'{bad}')

    # --- 9. model の build --------------------------------------------------
    if do_build:
        import copy
        import torch
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        bad, info = [], []
        for c in CONDS:
            cf = cfgs[(c, 'aerial')]     # 代表 1 ドメイン（データ以外は同型）
            if cf.get('custom_imports'):
                import_modules_from_strings(**cf['custom_imports'])
            try:
                m = MODELS.build(copy.deepcopy(cf.model))
                n = sum(p.numel() for p in m.parameters() if p.requires_grad)
                info.append(f'{c}: {type(m).__name__} trainable {n/1e6:.1f}M')
                del m
                torch.cuda.empty_cache()
            except Exception as e:
                bad.append(f'{c}: {type(e).__name__}: {str(e)[:60]}')
        check(9, 'model が build でき条件どおり', not bad,
              ' / '.join(info) + (f' / {bad}' if bad else ''))

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
