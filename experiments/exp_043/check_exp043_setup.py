"""exp_043 実行前検証（design.md §4.3）。本環境で実施し、クラスタへは push して持ち込む。

使い方:
    python experiments/exp_043/check_exp043_setup.py          # config のみ（GPU 不要）
    python experiments/exp_043/check_exp043_setup.py --data   # + 実データのバッチ構成確認
    python experiments/exp_043/check_exp043_setup.py --step   # + 実データ 1 step（GPU 1 枚）

検証項目
  1. 12 本の config が build でき、学習設定が既存 RF100 系列と一致
     （20 epoch / milestones[15] / lr 1e-4 / wd 1e-4 / clip 0.1 / batch 8 / seed 0 /
     num_classes 256）。バッチ構成が §2.2（t=1: [4,4] 2 ソース / t>=2: [4,2,2] 3 ソース、
     過去プール累積）であること
  2. 機械的 diff: model dict が既存系列と一致（replay は exp_040 の replay と同一、
     kdE は exp_040 の kdE と num_buffer_per_batch=2→4 だけの差）
  3. --cfg-options の到達性: model.teacher_ckpt が model に届き、model. なしの誤記は
     届かないこと（exp_040 項目 10 と同じ）
  4. （--data）実データでバッチを引き、順序・枚数が [現在4, 汎用4]（t=1）/
     [現在4, 汎用2, 過去2]（t>=2）になっていること（img_path で分類）
  5. （--step）実データ 1 step で loss が有限（replay と kdE、t=1 / t>=2 の両構成）。
     kdE はバッファ末尾 4 枚の assert を通過し loss_kd が有限であること。VRAM も実測
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


def _flat(d, pre=''):
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else []):
        key = f'{pre}.{k}' if pre else k
        if isinstance(v, dict):
            out.update(_flat(v, key))
        else:
            out[key] = v
    return out


def classify(img_path, domain):
    if f'/rf100_domain/{domain}/' in img_path:
        return 'current'
    if '/o365v1_stage/' in img_path:
        return 'generic'
    if '/rf100_domain/' in img_path:
        return 'past'
    return '?'


if __name__ == '__main__':
    do_data = '--data' in sys.argv
    do_step = '--step' in sys.argv
    init_default_scope('mmdet')

    # --- 1. build・学習設定・バッチ構成 --------------------------------------
    cfgs, bad = {}, []
    for cond in ('replay', 'kdE'):
        for i, d in enumerate(FULL_ORDER):
            t = i + 1
            p = os.path.join(CFG_DIR, f'{cond}_{d}.py')
            try:
                c = Config.fromfile(p)
                cfgs[(cond, d)] = c
            except Exception as e:
                bad.append(f'{cond}_{d}: {type(e).__name__}')
                continue
            tag = f'{cond}_{d}'
            for label, got, want in (
                    ('max_epochs', c.train_cfg.get('max_epochs'), 20),
                    ('milestones', list(c.param_scheduler[0]['milestones']), [15]),
                    ('lr', c.optim_wrapper['optimizer']['lr'], 0.0001),
                    ('wd', c.optim_wrapper['optimizer']['weight_decay'], 0.0001),
                    ('clip', c.optim_wrapper['clip_grad']['max_norm'], 0.1),
                    ('batch', c.train_dataloader['batch_size'], 8),
                    ('seed', c.randomness['seed'], 0),
                    ('num_classes', c.model['bbox_head']['num_classes'], 256)):
                if got != want:
                    bad.append(f'{tag}: {label}={got}(期待 {want})')
            s = c.train_dataloader['sampler']
            ds = c.train_dataloader['dataset']
            if s['type'] != 'CurrentEpochMultiSourceSampler':
                bad.append(f'{tag}: sampler={s["type"]}')
            want_ratio = [4, 4] if t == 1 else [4, 2, 2]
            if list(s['source_ratio']) != want_ratio:
                bad.append(f'{tag}: ratio={s["source_ratio"]}(期待 {want_ratio})')
            subs = ds['datasets']
            if len(subs) != (2 if t == 1 else 3):
                bad.append(f'{tag}: ソース数={len(subs)}')
                continue
            if d not in subs[0]['ann_file']:
                bad.append(f'{tag}: ソース0 が現在ドメインでない')
            if 'reference_o365v1_1000' not in subs[1]['ann_file']:
                bad.append(f'{tag}: ソース1 が汎用バッファでない')
            if t > 1:
                got_p = [q['ann_file'].split('/')[-1]
                         for q in subs[2].get('datasets', [])]
                want_p = [f'{q}_500.odvg.json' for q in FULL_ORDER[:i]]
                if got_p != want_p:
                    bad.append(f'{tag}: 過去プール={got_p}')
    check(1, '12 本 build・学習設定・バッチ構成（§2.1-2.2）',
          len(cfgs) == 12 and not bad,
          f'{len(cfgs)}/12 / 不一致 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 2. 機械的 diff（exp_040 と）-----------------------------------------
    bad = []
    ref_r = Config.fromfile(os.path.join(
        ROOT, 'experiments', 'exp_040', 'configs', 'replay_aerial.py'))
    ref_k = Config.fromfile(os.path.join(
        ROOT, 'experiments', 'exp_040', 'configs', 'kdE_aerial.py'))
    for (cond, d), c in cfgs.items():
        ref = ref_r if cond == 'replay' else ref_k
        fa, fb = _flat(ref.model), _flat(c.model)
        for k in sorted(set(fa) | set(fb)):
            if fa.get(k) == fb.get(k):
                continue
            if cond == 'kdE' and k == 'kd.num_buffer_per_batch':
                if (fa.get(k), fb.get(k)) != (2, 4):
                    bad.append(f'{cond}_{d}: {k}={fb.get(k)}(期待 4)')
                continue
            bad.append(f'{cond}_{d}: model.{k} 040={fa.get(k)!r} 043={fb.get(k)!r}')
    check(2, 'model が既存系列と一致（kdE は num_buffer_per_batch=4 のみ差）',
          not bad, f'12 本 / 想定外の差分 {len(bad)}' + (f' {bad[:3]}' if bad else ''))

    # --- 3. --cfg-options の到達性 -------------------------------------------
    from mmengine.config import DictAction
    bad = []
    act = DictAction(option_strings=['--cfg-options'], dest='o')

    def parse(args):
        # DictAction は namespace の既存辞書へマージするため、毎回新しい namespace を使う
        ns = type('NS', (), {})()
        act(None, ns, args)
        return ns.o

    c = copy.deepcopy(cfgs[('kdE', 'underwater')])
    c.merge_from_dict(parse(['model.teacher_ckpt=/tmp/t.pth']))
    if c.model['teacher_ckpt'] != '/tmp/t.pth':
        bad.append('model.teacher_ckpt が届かない')
    c2 = copy.deepcopy(cfgs[('kdE', 'underwater')])
    c2.merge_from_dict(parse(['teacher_ckpt=/tmp/x.pth']))
    if c2.model.get('teacher_ckpt') is not None:
        bad.append('model. なしの誤記が model に届いてしまう')
    check(3, 'model.teacher_ckpt が届く（誤記は届かない）', not bad,
          f'{bad[:2]}' if bad else 'OK')

    # --- 4. 実データのバッチ構成（--data）-----------------------------------
    if do_data:
        from mmengine.runner import Runner
        from mmengine.utils import import_modules_from_strings
        bad, info = [], []
        for cond, d, want in (('replay', 'underwater', ['current'] * 4 + ['generic'] * 4),
                              ('kdE', 'electromagnetic',
                               ['current'] * 4 + ['generic'] * 2 + ['past'] * 2)):
            c = cfgs[(cond, d)]
            if c.get('custom_imports'):
                import_modules_from_strings(**c['custom_imports'])
            loader = Runner.build_dataloader(copy.deepcopy(c.train_dataloader))
            it = iter(loader)
            ok_all = True
            for _ in range(3):          # 3 バッチ確認
                batch = next(it)
                kinds = [classify(s.img_path, d) for s in batch['data_samples']]
                if kinds != want:
                    ok_all = False
                    bad.append(f'{cond}_{d}: {kinds}')
                    break
            info.append(f'{cond}_{d}: {"/".join(want)} ×3 バッチ一致={ok_all}')
            del loader, it
        check(4, '実バッチが [現在4,汎用4]（t=1）/ [現在4,汎用2,過去2]（t>=2）',
              not bad, ' / '.join(info) + (f' / {bad[:1]}' if bad else ''))

    # --- 5. 実データ 1 step（--step）-----------------------------------------
    if do_step:
        import torch
        from mmengine.runner import Runner
        from mmengine.runner.checkpoint import load_checkpoint
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        bad, info = [], []
        for cond, d in (('replay', 'underwater'), ('replay', 'electromagnetic'),
                        ('kdE', 'underwater'), ('kdE', 'electromagnetic')):
            c = cfgs[(cond, d)]
            if c.get('custom_imports'):
                import_modules_from_strings(**c['custom_imports'])
            m = copy.deepcopy(c.model)
            if cond == 'kdE':
                m['teacher_ckpt'] = THETA0
            # Runner と同じ順序を踏む: build -> デバイスへ移動 -> init_weights
            # -> load_from。教師の構築と捕捉 hook の登録は init_weights() で行われ、
            # 教師は学生の deepcopy なので移動後に呼ばないと CPU に残る。
            model = MODELS.build(m).cuda()
            if cond == 'kdE':
                model.init_weights()
            load_checkpoint(model, THETA0, map_location='cpu')
            model = model.train()
            torch.cuda.reset_peak_memory_stats()
            loader = Runner.build_dataloader(copy.deepcopy(c.train_dataloader))
            batch = next(iter(loader))
            data = model.data_preprocessor(batch, True)
            losses = model.loss(data['inputs'], data['data_samples'])
            total = sum(v.sum() for v in losses.values()
                        if isinstance(v, torch.Tensor) and v.requires_grad)
            total.backward()
            kd = losses.get('loss_kd')
            if not torch.isfinite(total) or (cond == 'kdE' and
                                             (kd is None or not torch.isfinite(kd))):
                bad.append(f'{cond}_{d}: loss 異常')
            vram = torch.cuda.max_memory_allocated() / 2**30
            info.append(
                f'{cond}_{d}(t{"1" if d == "underwater" else "2"}): '
                f'loss={float(total):.2f}'
                + (f' kd={float(kd):.4f}' if kd is not None else '')
                + f' VRAM={vram:.1f}GB')
            del model, loader
            torch.cuda.empty_cache()
        check(5, '実データ 1 step で loss（kdE は loss_kd 込み）が有限', not bad,
              ' / '.join(info))

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
