"""exp_041 実行前検証（design.md §6.3）。本環境で実施し、クラスタへは push して持ち込む。

使い方:
    python experiments/exp_041/check_exp041_setup.py              # config のみ（GPU 不要）
    python experiments/exp_041/check_exp041_setup.py --build      # + model build（項目1後半）
    python experiments/exp_041/check_exp041_setup.py --step       # + 初期ckpt読込と実データ1 step（項目5。GPU）
    python experiments/exp_041/check_exp041_setup.py --evalcheck  # + 260/152クラス評価の等価確認（項目7。GPU）

全項目 OK でなければクラスタへ投入しない。

検証項目（design.md §6.3）
  1. 学習12本・評価10本の config が build でき、継承元が正しい。
     （--build 時）model が build でき、学習対象が手法どおり（ZiRa は RDB、DitHub は LoRA）
  2. 学習設定が exp_039 / exp_040 と同一（20 epoch / milestones[15] / lr 1e-4 / wd 1e-4 /
     clip 0.1 / batch 6 or 4 / seed 0 / num_classes 256）
  3. 手法固有値が design.md §2.3 と一致（ZiRa: zil=0.1・llrb 0.2 / DitHub: warmup_epochs=10・
     trained_classes・クラス数・encoder.num_cp=0・encoder_cp=6・replay の検出器型）
  4. 機械的 diff: model dict が exp_039 の同条件 config と一致（dithub_classes を除く）、
     train_dataloader が exp_040 の同ドメイン config と一致（exp_039 item7 / exp_040 item7 の方式）
  5. （--step 時）初期パラメータ（exp_039 の t=3 融合済み ckpt）を読み、想定どおりの
     missing/unexpected keys になり、実データ 1 step で loss が有限になること。
     DitHub は warmup と specialization の両フェーズで 1 step ずつ流す
     （replay は DitHubReplayLinear のライブラリ非登録クラス経路を通す）
  6. リプレイのバッチ構成が [現在4, 参照1, 過去プール1] で、過去プールが累積構成
     （t=4: 3ドメイン、t=5: 4、t=6: 5）であること
  7. （--evalcheck 時）260 クラス和集合の評価 config が、既存の 152 クラス版と同一の
     予測を返すこと（t=3 ライブラリ・underwater valid で実測。design.md §4 の前提の確認）
  8. 生成 config に本環境固有の絶対パスを埋め込んでいないこと
"""
import copy
import os
import sys

from mmengine.config import Config
from mmengine.registry import init_default_scope

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CFG_DIR = os.path.join(HERE, 'configs')
DOMAINS = ['aerial', 'microscopic', 'documents']
FULL_ORDER = ['underwater', 'electromagnetic', 'videogames',
              'aerial', 'microscopic', 'documents']
N_CLASSES = {'aerial': 22, 'microscopic': 28, 'documents': 59}
# 式3 の対象は canonical_key（dithub_layers.py）で突き合わせる（2026-08-15 実測）。
# aerial の 'Fish' は underwater の 'fish' と同一キー（class_fish）。
TRAINED = {'aerial': ['Fish'], 'microscopic': [], 'documents': ['object']}
# canonical キーの個数（aerial は 'orange-sphero'/'orange_sphero' が衝突し 22→21）と
# t=3 ライブラリ（152 キー）との重複キー数。--step の期待値に使う。
N_SLOTS = {'aerial': 21, 'microscopic': 28, 'documents': 59}
N_OVERLAP = {'aerial': 1, 'microscopic': 0, 'documents': 1}
EXPECT_BATCH = {'replay': 6, 'replayfree': 4}
EXPECT_SAMPLER = {'replay': 'CurrentEpochMultiSourceSampler',
                  'replayfree': 'DefaultSampler'}
N_UNION = 260          # 6ドメイン和集合（gen_configs.py で算出）
N_LIB_T3 = 152         # exp_039 t=3 ライブラリのクラス数（前半3ドメイン和集合）

results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


def all_cfgs():
    for m in ('zira', 'dithub'):
        for r in ('replay', 'replayfree'):
            for d in DOMAINS:
                yield m, r, d, os.path.join(CFG_DIR, f'{m}_{r}_{d}.py')


def _flat(d, pre=''):
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else []):
        key = f'{pre}.{k}' if pre else k
        if isinstance(v, dict):
            out.update(_flat(v, key))
        elif k == 'dithub_classes':
            out[key] = f'<{len(v)}>'
        else:
            out[key] = v
    return out


def _to_plain(x):
    """ConfigDict/リストを素の dict/list に落として == 比較できるようにする。"""
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_plain(v) for v in x]
    return x


if __name__ == '__main__':
    do_build = '--build' in sys.argv
    do_step = '--step' in sys.argv
    do_evalcheck = '--evalcheck' in sys.argv
    init_default_scope('mmdet')

    # --- 1. build と継承元 --------------------------------------------------
    cfgs, bad = {}, []
    for m, r, d, p in all_cfgs():
        if not os.path.exists(p):
            bad.append(f'{m}_{r}_{d}: なし')
            continue
        try:
            cfgs[(m, r, d)] = Config.fromfile(p)
        except Exception as e:
            bad.append(f'{m}_{r}_{d}: {type(e).__name__}')
            continue
        src = open(p).read()
        want = f'../../exp_040/configs/{r}_{d}.py'
        if want not in src:
            bad.append(f'{m}_{r}_{d}: 継承元が exp_040 でない')
    eval_cfgs, n_eval = {}, 0
    for name, want_type in (
            [(f'zira_eval_{d}', 'ZiRaGroundingDINO') for d in DOMAINS]
            + [(f'dithub_eval_{t}', 'DitHubGroundingDINO')
               for t in FULL_ORDER + ['zcoco']]):
        p = os.path.join(CFG_DIR, f'{name}.py')
        try:
            c = Config.fromfile(p)
            eval_cfgs[name] = c
        except Exception as e:
            bad.append(f'{name}: {type(e).__name__}')
            continue
        n_eval += 1
        if c.model['type'] != want_type:
            bad.append(f'{name}: type={c.model["type"]}')
        if want_type == 'DitHubGroundingDINO':
            if len(c.model['dithub_classes']) != N_UNION:
                bad.append(f'{name}: {len(c.model["dithub_classes"])} クラス'
                           f'(期待 {N_UNION})')
            if c.model['encoder'].get('num_cp') != 0 or c.model.get('encoder_cp') != 0:
                bad.append(f'{name}: 評価に checkpointing が残っている')
    check(1, '学習12本・評価10本の config が build でき継承元が正しい',
          len(cfgs) == 12 and n_eval == 10 and not bad,
          f'学習 {len(cfgs)}/12 評価 {n_eval}/10'
          + (f' / 問題 {bad[:3]}' if bad else ''))

    # --- 2. 学習設定 --------------------------------------------------------
    bad = []
    for (m, r, d), c in cfgs.items():
        tag = f'{m}_{r}_{d}'
        for label, got, want in (
                ('max_epochs', c.train_cfg.get('max_epochs'), 20),
                ('loop', c.train_cfg.get('type'), 'EpochBasedTrainLoop'),
                ('milestones', list(c.param_scheduler[0]['milestones']), [15]),
                ('lr', c.optim_wrapper['optimizer']['lr'], 0.0001),
                ('wd', c.optim_wrapper['optimizer']['weight_decay'], 0.0001),
                ('clip', c.optim_wrapper['clip_grad']['max_norm'], 0.1),
                ('batch', c.train_dataloader['batch_size'], EXPECT_BATCH[r]),
                ('seed', c.randomness['seed'], 0),
                ('num_classes', c.model['bbox_head']['num_classes'], 256)):
            if got != want:
                bad.append(f'{tag}: {label}={got}(期待 {want})')
    check(2, '学習設定が exp_039 / exp_040 と同一', not bad,
          f'12 本 × 9 項目 / 不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 3. 手法固有値 ------------------------------------------------------
    bad = []
    for (m, r, d), c in cfgs.items():
        tag = f'{m}_{r}_{d}'
        if m == 'zira':
            if c.model['type'] != 'ZiRaGroundingDINO':
                bad.append(f'{tag}: type={c.model["type"]}')
            if c.model.get('zil_loss_weight') != 0.1:
                bad.append(f'{tag}: zil={c.model.get("zil_loss_weight")}')
            if c.model['neck']['type'] != 'ZiRaChannelMapper':
                bad.append(f'{tag}: neck={c.model["neck"]["type"]}')
            ck = c.optim_wrapper['paramwise_cfg']['custom_keys']
            if ck.get('llrb', {}).get('lr_mult') != 0.2:
                bad.append(f'{tag}: llrb lr_mult={ck.get("llrb")}')
        else:
            want_type = ('DitHubReplayGroundingDINO' if r == 'replay'
                         else 'DitHubODVGGroundingDINO')
            if c.model['type'] != want_type:
                bad.append(f'{tag}: type={c.model["type"]}(期待 {want_type})')
            h = [x for x in (c.get('custom_hooks') or [])
                 if x['type'] == 'DitHubSeqPhaseHook']
            if len(h) != 1 or h[0].get('warmup_epochs') != 10:
                bad.append(f'{tag}: hook={h}')
            elif list(h[0].get('trained_classes') or []) != TRAINED[d]:
                bad.append(f'{tag}: trained={h[0].get("trained_classes")}')
            n = len(c.model.get('dithub_classes') or ())
            if n != N_CLASSES[d]:
                bad.append(f'{tag}: dithub_classes={n}(期待 {N_CLASSES[d]})')
            if c.model['encoder'].get('num_cp') != 0:
                bad.append(f'{tag}: encoder.num_cp='
                           f'{c.model["encoder"].get("num_cp")}(期待 0)')
            if c.model.get('encoder_cp') != 6:
                bad.append(f'{tag}: encoder_cp={c.model.get("encoder_cp")}')
            if not c.get('find_unused_parameters'):
                bad.append(f'{tag}: find_unused_parameters が無い')
    check(3, '手法固有値が design.md §2.3 と一致', not bad,
          f'不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 4. 機械的 diff（exp_039 の model / exp_040 のデータ経路）-----------
    # 個別項目の列挙では拾えない取りこぼしを機械的に検出する
    # （exp_039 item7 / exp_040 item7 で確立した方式）。
    bad = []
    for (m, r, d), c in cfgs.items():
        # model: exp_039 の同 method/replay（underwater を代表に取る。model は
        # dithub_classes 以外にドメイン依存が無い）
        ref = Config.fromfile(os.path.join(
            ROOT, 'experiments', 'exp_039', 'configs', f'{m}_{r}_underwater.py'))
        fa, fb = _flat(ref.model), _flat(c.model)
        for k in sorted(set(fa) | set(fb)):
            if k == 'dithub_classes':          # クラス数の差は意図どおり（項目3で検査）
                continue
            if fa.get(k) != fb.get(k):
                bad.append(f'{m}_{r}_{d}: model.{k} '
                           f'exp039={fa.get(k)!r} exp041={fb.get(k)!r}')
        # データ経路: exp_040 の同 replay/domain と完全一致
        ref40 = Config.fromfile(os.path.join(
            ROOT, 'experiments', 'exp_040', 'configs', f'{r}_{d}.py'))
        if _to_plain(ref40.train_dataloader) != _to_plain(c.train_dataloader):
            bad.append(f'{m}_{r}_{d}: train_dataloader が exp_040 と不一致')
    check(4, 'model が exp_039、train_dataloader が exp_040 と一致', not bad,
          f'12 本 / 想定外の差分 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 6. リプレイのバッチ構成（累積過去プール）---------------------------
    bad = []
    for (m, r, d), c in cfgs.items():
        tag = f'{m}_{r}_{d}'
        t = 4 + DOMAINS.index(d)
        s = c.train_dataloader['sampler']
        if s['type'] != EXPECT_SAMPLER[r]:
            bad.append(f'{tag}: sampler={s["type"]}')
            continue
        ds = c.train_dataloader['dataset']
        if r == 'replayfree':
            if ds['type'] != 'ODVGDataset' or d not in ds['ann_file']:
                bad.append(f'{tag}: dataset={ds["type"]}')
            continue
        if list(s.get('source_ratio', [])) != [4, 1, 1]:
            bad.append(f'{tag}: source_ratio={s.get("source_ratio")}')
        subs = ds['datasets']
        if len(subs) != 3:
            bad.append(f'{tag}: ソース数={len(subs)}')
            continue
        if d not in subs[0]['ann_file']:
            bad.append(f'{tag}: ソース0 が現在ドメインでない')
        if 'reference_o365v1_1000' not in subs[1]['ann_file']:
            bad.append(f'{tag}: ソース1 が参照バッファでない')
        pool = subs[2]
        want_past = FULL_ORDER[:t - 1]
        got = [p['ann_file'].split('/')[-1] for p in pool.get('datasets', [])]
        want = [f'{p}_500.odvg.json' for p in want_past]
        if pool.get('type') != 'ConcatDataset' or got != want:
            bad.append(f'{tag}: 過去プール={got}(期待 {want})')
    check(6, 'リプレイが [現在4, 参照1, 過去1] で過去プールが累積構成', not bad,
          f'不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 8. 本環境固有パスの埋め込み ----------------------------------------
    bad = []
    for f in sorted(os.listdir(CFG_DIR)):
        src = open(os.path.join(CFG_DIR, f)).read()
        for token in ('/workspace/', '/root/', '/home/kouyou/'):
            if token in src:
                bad.append(f'{f}: {token}')
    check(8, '生成 config に環境固有の絶対パスを埋め込んでいない', not bad,
          f'22 本 / 検出 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 1 後半. model build と学習対象（--build）---------------------------
    if do_build:
        import torch
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        bad, info = [], []
        for (m, r, d), c in sorted(cfgs.items()):
            if d != 'aerial':
                continue          # 代表 1 ドメインで足りる（クラス数以外は同型）
            import_modules_from_strings(**c['custom_imports'])
            model = MODELS.build(copy.deepcopy(c.model))
            tr = [n for n, p in model.named_parameters() if p.requires_grad]
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if m == 'zira':
                ok = all(('llrb' in n or 'hlrb' in n or 'scaling' in n)
                         for n in tr)
                key = 'llrb/hlrb/scaling'
            else:
                ok = all(('lora' in n or 'warmup' in n) for n in tr)
                key = 'lora/warmup'
            if not ok:
                bad.append(f'{m}_{r}_{d}: 想定外の学習対象')
            info.append(f'{m}_{r}: {len(tr)} param ({n_tr/1e6:.2f}M, {key})')
            del model
            torch.cuda.empty_cache()
        check('1b', 'model が build でき学習対象が手法どおり', not bad,
              ' / '.join(info))

    # --- 5. 初期 ckpt の読込と実データ 1 step（--step）----------------------
    if do_step:
        import torch
        from mmengine.runner import Runner
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        bad, info = [], []
        for m in ('zira', 'dithub'):
            for r in ('replayfree', 'replay'):
                tag, d = f'{m}_{r}', 'aerial'
                c = cfgs[(m, r, d)]
                init = os.path.join(ROOT, 'experiments', 'exp_039',
                                    f'{tag}_merged',
                                    'merged_after_t3_videogames.pth')
                import_modules_from_strings(**c['custom_imports'])
                model = MODELS.build(copy.deepcopy(c.model))
                sd = torch.load(init, map_location='cpu')['state_dict']
                miss, unexp = model.load_state_dict(sd, strict=False)
                if m == 'zira':
                    # 同一アーキテクチャなので過不足なし
                    if miss or unexp:
                        bad.append(f'{tag}: missing={len(miss)} '
                                   f'unexpected={len(unexp)}(期待 0/0)')
                else:
                    # missing = 現ドメインの per_class A のうちライブラリに無いもの
                    #           = (canonical スロット数 − 重複キー数)×109。
                    #           aerial は 21 スロット・class_fish がライブラリと一致
                    #           するため (21−1)×109 = 2180。
                    # unexpected = ライブラリのうち現ドメインのスロットに無いもの
                    #           = (152 − 重複キー数)×109。学習 config は現ドメインの
                    #           スロットしか持たず、これらはタスク終了時の
                    #           merge_dithub.py（A 和集合）で保全される（design.md §2.1）
                    want_miss = (N_SLOTS[d] - N_OVERLAP[d]) * 109
                    want_unexp = (N_LIB_T3 - N_OVERLAP[d]) * 109
                    ok = (all('per_class_lora_A' in k for k in miss)
                          and len(miss) == want_miss
                          and all('per_class_lora_A' in k for k in unexp)
                          and len(unexp) == want_unexp)
                    if not ok:
                        bad.append(f'{tag}: missing={len(miss)} '
                                   f'unexpected={len(unexp)}'
                                   f'(期待 {want_miss}/{want_unexp})')
                model = model.cuda().train()
                loader = Runner.build_dataloader(copy.deepcopy(c.train_dataloader))
                batch = next(iter(loader))
                phases = [('', None)]
                if m == 'dithub':
                    # 実学習では DitHubPhaseHook が train 開始時に warmup へ初期化する
                    # （ライブラリ ckpt の per_class_enabled=1 を持ち越さないため）
                    model.set_phase(False)
                    model.reinit_warmup_a()
                    phases = [('warmup', None), ('spec', TRAINED[d])]
                for ph, trained in phases:
                    if ph == 'spec':
                        model.enable_per_class(trained_keys=trained)
                    data = model.data_preprocessor(batch, True)
                    losses = model.loss(data['inputs'], data['data_samples'])
                    total = sum(v.sum() for v in losses.values()
                                if isinstance(v, torch.Tensor))
                    if not torch.isfinite(total):
                        bad.append(f'{tag}[{ph}]: loss が有限でない')
                    info.append(f'{tag}{"/" + ph if ph else ""}: '
                                f'loss={float(total):.2f}')
                del model, loader
                torch.cuda.empty_cache()
        check(5, '初期 ckpt を読み実データ 1 step で loss が有限', not bad,
              (f'{bad[:2]} / ' if bad else '') + ' / '.join(info))

    # --- 7. 260/152 クラス評価の等価確認（--evalcheck）----------------------
    if do_evalcheck:
        import torch
        from mmengine.runner import Runner
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        N_BATCH = 8
        lib = os.path.join(ROOT, 'experiments', 'exp_039',
                           'dithub_replayfree_merged',
                           'merged_after_t3_videogames.pth')
        sd = torch.load(lib, map_location='cpu')['state_dict']
        preds = {}
        for name, path in (
                ('152', os.path.join(ROOT, 'experiments', 'exp_023', 'configs',
                                     'dithub_eval_underwater.py')),
                ('260', os.path.join(CFG_DIR, 'dithub_eval_underwater.py'))):
            c = Config.fromfile(path)
            import_modules_from_strings(**c['custom_imports'])
            model = MODELS.build(copy.deepcopy(c.model))
            model.load_state_dict(sd, strict=False)
            model = model.cuda().eval()
            loader = Runner.build_dataloader(copy.deepcopy(c.test_dataloader))
            out = []
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= N_BATCH:
                        break
                    out.extend(model.test_step(batch))
            preds[name] = out
            del model, loader
            torch.cuda.empty_cache()
        bad = []
        for i, (a, b) in enumerate(zip(preds['152'], preds['260'])):
            pa, pb = a.pred_instances, b.pred_instances
            if not (torch.equal(pa.bboxes, pb.bboxes)
                    and torch.equal(pa.scores, pb.scores)
                    and torch.equal(pa.labels, pb.labels)):
                bad.append(f'画像 {i}: 予測が不一致')
        check(7, '260 クラス評価 config が 152 クラス版と同一の予測を返す', not bad,
              f'{N_BATCH} 画像で bboxes/scores/labels を完全一致比較'
              + (f' / {bad[:2]}' if bad else ''))

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
