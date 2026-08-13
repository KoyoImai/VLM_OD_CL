"""exp_039 実行前検証（design.md §6.3）。本環境で実施し、クラスタへは push して持ち込む。

使い方:
    python experiments/exp_039/check_exp039_setup.py          # config のみ（GPU 不要）
    python experiments/exp_039/check_exp039_setup.py --build  # model の build も行う

全項目 OK でなければクラスタへ投入しない。

検証項目
  1. 12 本の config が build でき、継承元がリプレイ・蒸留側になっている
  2. 学習設定がリプレイ・蒸留と同一（20 epoch / milestones[15] / lr 1e-4 / wd 1e-4 /
     clip 0.1 / batch 6 or 4 / seed 0 / dn 有効 / num_classes 256）
  3. データ経路がリプレイ・蒸留と同一（ODVGDataset・サンプラ・source_ratio）
  4. 手法固有値が design.md §2.2 と一致
       ZiRa   : zil_loss_weight=0.1 / neck=ZiRaChannelMapper / llrb lr_mult=0.2
       DitHub : warmup_epochs=10 / trained_classes / dithub_classes のクラス数 /
                encoder.num_cp=0（fairscale 再入版が DDP と非互換）/ encoder_cp>0 /
                検出器の型（replay は DitHubReplayGroundingDINO）
  7. model dict が exp_023 と一致（意図した差分を除く）— 個別項目の列挙では拾えない
     取りこぼしを機械的に検出する
  5. 本環境固有のパスを config に埋め込んでいないこと（クラスタで解決できること）
  6. （--build 時）model が build でき、学習対象が手法どおり
"""
import os
import sys
from collections import Counter

from mmengine.config import Config
from mmengine.registry import init_default_scope

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, 'configs')
DOMAINS = ['underwater', 'electromagnetic', 'videogames']
N_CLASSES = {'underwater': 28, 'electromagnetic': 39, 'videogames': 87}
TRAINED = {'underwater': [], 'electromagnetic': [], 'videogames': ['person', 'car']}
EXPECT_BATCH = {'replay': 6, 'replayfree': 4}
EXPECT_SAMPLER = {'replay': 'CurrentEpochMultiSourceSampler',
                  'replayfree': 'DefaultSampler'}

results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


def all_cfgs():
    for m in ('zira', 'dithub'):
        for r in ('replay', 'replayfree'):
            for d in DOMAINS:
                yield m, r, d, os.path.join(CFG_DIR, f'{m}_{r}_{d}.py')


if __name__ == '__main__':
    do_build = '--build' in sys.argv
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
        want = ('fullft_replay_' if r == 'replay' else 'fullft_replayfree_')
        if want not in src:
            bad.append(f'{m}_{r}_{d}: 継承元が {want}* でない')
    check(1, '12 本の config が build でき継承元がリプレイ・蒸留側',
          len(cfgs) == 12 and not bad,
          f'{len(cfgs)}/12' + (f' / 問題 {bad[:3]}' if bad else ''))

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
    check(2, '学習設定がリプレイ・蒸留と同一', not bad,
          f'12 本 × 9 項目 / 不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 3. データ経路 ------------------------------------------------------
    bad = []
    for (m, r, d), c in cfgs.items():
        tag = f'{m}_{r}_{d}'
        s = c.train_dataloader['sampler']['type']
        if s != EXPECT_SAMPLER[r]:
            bad.append(f'{tag}: sampler={s}')
        ds = c.train_dataloader['dataset']
        if r == 'replay':
            if ds['type'] != 'ConcatDataset':
                bad.append(f'{tag}: dataset={ds["type"]}')
            ratio = list(c.train_dataloader['sampler'].get('source_ratio', []))
            want = [2, 1] if d == 'underwater' else [4, 1, 1]
            if ratio != want:
                bad.append(f'{tag}: source_ratio={ratio}(期待 {want})')
        else:
            if ds['type'] != 'ODVGDataset':
                bad.append(f'{tag}: dataset={ds["type"]}')
    check(3, 'データ経路がリプレイ・蒸留と同一', not bad,
          f'不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 4. 手法固有値 ------------------------------------------------------
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
            # リプレイ有りは DitHubReplayGroundingDINO でなければならない
            # （exp_023 と同一）。参照(Objects365)・過去ドメインの画像は現ドメインの
            # dithub_classes に無いクラスを持つため、素の _delta_per_sample は
            # specialization 中に AttributeError で落ちる（2026-08-13 にクラスタで発生）。
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
            if h and h[0].get('warmup_iters') is not None:
                bad.append(f'{tag}: warmup_iters が残っている')
            n = len(c.model.get('dithub_classes') or ())
            if n != N_CLASSES[d]:
                bad.append(f'{tag}: dithub_classes={n}(期待 {N_CLASSES[d]})')
            # fairscale の再入版 checkpointing は DDP と非互換
            # （2026-08-12 にクラスタで "marked as ready twice"）。
            # DitHub は非再入版を encoder_cp で自前に掛けるので num_cp は 0。
            if c.model['encoder'].get('num_cp') != 0:
                bad.append(f'{tag}: encoder.num_cp='
                           f'{c.model["encoder"].get("num_cp")}(期待 0)')
            if not c.model.get('encoder_cp'):
                bad.append(f'{tag}: encoder_cp={c.model.get("encoder_cp")}')
    check(4, '手法固有値が design.md §2.2 と一致', not bad,
          f'不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 5. 本環境固有パスの埋め込み ----------------------------------------
    bad = []
    for m, r, d, p in all_cfgs():
        if not os.path.exists(p):
            continue
        src = open(p).read()
        for token in ('/workspace/', '/root/', '/home/kouyou/'):
            if token in src:
                bad.append(f'{m}_{r}_{d}: {token}')
    check(5, '生成 config に環境固有パスを埋め込んでいない', not bad,
          f'検出 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 7. exp_023 の model dict との突き合わせ ----------------------------
    # 継承元をリプレイ・蒸留側へ張り替えたぶん overlay を手で書き起こしているため、
    # 取りこぼしを個別項目ではなく機械的な差分で検出する（2026-08-13 追加。
    # 個別項目の列挙では encoder.num_cp と検出器 type の 2 件を続けて取りこぼした）。
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

    # 継承元をリプレイ・蒸留側へ変えたことで生じる差分（design.md §2.1 の意図どおり）。
    # exp_024 の replayfree base が backbone.init_cfg=None を指定しているため。
    ALLOWED = {'backbone.init_cfg.type', 'backbone.init_cfg.checkpoint'}
    bad = []
    for (m, r, d), c in cfgs.items():
        ref = Config.fromfile(os.path.join(
            CFG_DIR, '..', '..', 'exp_023', 'configs', f'{m}_{r}_{d}.py'))
        fa, fb = _flat(ref.model), _flat(c.model)
        for k in sorted(set(fa) | set(fb)):
            if k in ALLOWED or fa.get(k) == fb.get(k):
                continue
            bad.append(f'{m}_{r}_{d}: {k} exp023={fa.get(k)!r} exp039={fb.get(k)!r}')
    check(7, 'model dict が exp_023 と一致（意図した差分を除く）', not bad,
          f'12 本 / 想定外の差分 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 6. model の build --------------------------------------------------
    if do_build:
        import copy
        import torch
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        bad, info = [], []
        for (m, r, d), c in sorted(cfgs.items()):
            if d != 'underwater':
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
        check(6, 'model が build でき学習対象が手法どおり', not bad,
              ' / '.join(info))

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
