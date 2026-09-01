"""exp_054 実行前検証（design.md §2）。

使い方:
    python experiments/exp_054/check_exp054_setup.py          # 静的検査（GPU 不要）
    python experiments/exp_054/check_exp054_setup.py --step   # + 実データ 1 step（GPU 1 枚）

検証項目
  1. ドライバが参照する全 config（学習 6 手法 × 6 ドメイン・評価）が存在し build できる
  2. --cfg-options の到達性: randomness.seed / ckpt 保存方針の上書きが merge 後に効く
     （既存 config は変更しないため、seed 以外の全設定は既存実験と定義上同一）
  3. 補助スクリプト（Fisher 推定・InfLoRA prepare/update）が --seed 引数を受け付ける
  4. （--step）Ours（kdE）と EWC で seed=1 の実データ 1 step が有限
     （seed 上書きが学習経路を壊さないことの確認）
"""
import copy
import os
import sys

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DOMS = ['underwater', 'electromagnetic', 'videogames', 'aerial', 'microscopic', 'documents']
FIRST3 = DOMS[:3]
THETA0 = os.path.join(
    ROOT, 'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det'
    '_20231204_095047-b448804b.pth')
results = []


def check(no, name, ok, detail=''):
    results.append(ok)
    print(f'[{"OK" if ok else "NG"}] {no}. {name}' + (f'  {detail}' if detail else ''))


def train_cfgs():
    out = {}
    for d in DOMS:
        first = d in FIRST3
        out[('ours', d)] = (f'experiments/exp_035/configs/kdE_condA_l2w100_{d}.py' if first
                            else f'experiments/exp_040/configs/kdE_{d}.py')
        out[('er', d)] = (f'experiments/exp_027/configs/fullft_replay_{d}.py' if first
                          else f'experiments/exp_040/configs/replay_{d}.py')
        out[('ft', d)] = (f'experiments/exp_024/configs/fullft_replayfree_{d}.py' if first
                          else f'experiments/exp_040/configs/replayfree_{d}.py')
        out[('ewc', d)] = f'experiments/exp_045/configs/ewc_{d}.py'
        out[('inflora', d)] = f'experiments/exp_045/configs/inflora_{d}.py'
        out[('zira', d)] = (f'experiments/exp_039/configs/zira_replayfree_{d}.py' if first
                            else f'experiments/exp_041/configs/zira_replayfree_{d}.py')
        out[('dithub', d)] = (f'experiments/exp_039/configs/dithub_replayfree_{d}.py' if first
                              else f'experiments/exp_041/configs/dithub_replayfree_{d}.py')
    return out


def eval_cfgs():
    out = []
    for d in DOMS:
        plain = (f'experiments/exp_023/configs/eval_{d}.py' if d in FIRST3
                 else f'experiments/exp_026/configs/eval_{d}.py')
        out.append(plain)
        out.append(f'experiments/exp_041/configs/dithub_eval_{d}.py')
        out.append(f'experiments/exp_023/configs/zira_eval_{d}.py' if d in FIRST3
                   else f'experiments/exp_041/configs/zira_eval_{d}.py')
    out += ['configs/mm_grounding_dino/eval_base_coco.py',
            'experiments/exp_023/configs/zira_eval_zcoco.py',
            'experiments/exp_023/configs/dithub_eval_zcoco.py',
            'experiments/exp_041/configs/dithub_eval_zcoco.py']
    return out


if __name__ == '__main__':
    do_step = '--step' in sys.argv
    init_default_scope('mmdet')
    os.chdir(ROOT)

    # --- 1. 参照 config の存在と build ---------------------------------------
    bad, cfgs = [], {}
    for key, p in train_cfgs().items():
        if not os.path.exists(p):
            bad.append(f'{key}: 無い {p}')
            continue
        try:
            cfgs[key] = Config.fromfile(p)
        except Exception as e:
            bad.append(f'{key}: {type(e).__name__}')
    for p in eval_cfgs():
        if not os.path.exists(p):
            bad.append(f'eval: 無い {p}')
    check(1, '学習 42 本・評価 config の存在と build', not bad,
          f'学習 {len(cfgs)}/42' + (f' / {bad[:3]}' if bad else ''))

    # --- 2. --cfg-options の到達性 -------------------------------------------
    from mmengine.config import DictAction
    act = DictAction(option_strings=['--cfg-options'], dest='o')

    def parse(args):
        ns = type('NS', (), {})()
        act(None, ns, args)
        return ns.o

    bad = []
    for key in [('ours', 'underwater'), ('er', 'aerial'), ('ft', 'documents'),
                ('ewc', 'videogames'),
                ('inflora', 'documents'), ('zira', 'underwater'), ('dithub', 'aerial')]:
        c = copy.deepcopy(cfgs[key])
        c.merge_from_dict(parse([
            'randomness.seed=1', 'default_hooks.checkpoint.interval=20',
            'default_hooks.checkpoint.save_optimizer=False',
            'default_hooks.checkpoint.save_param_scheduler=False']))
        if c.randomness['seed'] != 1:
            bad.append(f'{key}: seed が届かない')
        h = c.default_hooks['checkpoint']
        if h['interval'] != 20 or h.get('save_optimizer') is not False:
            bad.append(f'{key}: checkpoint 上書きが届かない')
    check(2, 'randomness.seed=1 と ckpt 方針が全手法の config に届く', not bad,
          f'{bad[:2]}' if bad else '7 手法で確認')

    # --- 3. 補助スクリプトの --seed ------------------------------------------
    bad = []
    for p in ['projects/ewc_cl/estimate_fisher.py',
              'projects/lora_cl/inflora_prepare.py',
              'projects/lora_cl/inflora_update_memory.py']:
        s = open(p).read()
        if "'--seed'" not in s and '"--seed"' not in s:
            bad.append(p)
    check(3, 'Fisher / InfLoRA 補助スクリプトが --seed を受け付ける', not bad,
          f'{bad}' if bad else '3 本で確認')

    # --- 4. 実データ 1 step（--step）-----------------------------------------
    if do_step:
        from mmengine.runner import Runner
        from mmengine.runner.checkpoint import load_checkpoint
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        bad, info = [], []
        for key, needs_teacher in [(('ours', 'underwater'), True),
                                   (('ewc', 'underwater'), False)]:
            c = copy.deepcopy(cfgs[key])
            c.merge_from_dict(parse(['randomness.seed=1']))
            if 'custom_imports' in c:
                import_modules_from_strings(**c['custom_imports'])
            m = copy.deepcopy(c.model)
            if needs_teacher:
                m['teacher_ckpt'] = THETA0
            model = MODELS.build(m).cuda()
            model.init_weights()
            load_checkpoint(model, THETA0, map_location='cpu')
            loader = Runner.build_dataloader(copy.deepcopy(c.train_dataloader), seed=1)
            batch = next(iter(loader))
            model.train()
            data = model.data_preprocessor(batch, True)
            losses = model.loss(data['inputs'], data['data_samples'])
            total = sum(v.sum() for v in losses.values()
                        if isinstance(v, torch.Tensor) and v.requires_grad)
            total.backward()
            if not torch.isfinite(total):
                bad.append(f'{key}: loss={float(total)}')
            info.append(f'{key[0]}: loss={float(total):.1f}')
            del model, loader
            torch.cuda.empty_cache()
        check(4, 'seed=1 の実データ 1 step（Ours・EWC）で有限', not bad,
              ' / '.join(info) + (f' / {bad[:1]}' if bad else ''))

    print(f'\n{sum(results)}/{len(results)} OK')
    sys.exit(0 if all(results) else 1)
