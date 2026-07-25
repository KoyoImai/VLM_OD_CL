"""exp_027 デバッグ実行: リプレイ経路（特に Objects365）がクラスタで動くかの確認。

本実験がクラスタで新しく必要とするのは「Objects365 のバッファ画像が読めること」だけである。
40 時間級のジョブを投げてから転送漏れが発覚すると損失が大きいため、学習を伴わない短時間の
確認をここに集約する（design.md §6.1。exp_023.5 の Tier1 と同じ位置づけ）。

確認項目:
  1. bind      Objects365 のステージングディレクトリが見えるか
  2. buffer    バッファ JSON と label_map が読めるか（件数も確認）
  3. exists    バッファが指す画像ファイルが実在するか（転送漏れの検出）
  4. decode    実際に数枚デコードできるか（転送途中で切れたファイルの検出）
  5. mixing    各 t の dataloader で 1 バッチの内訳が設計どおりか
  6. step      θ0 をロードして model.loss() まで通り、凍結（条件B）が効いているか

使い方:
    python experiments/exp_027/debug_o365.py            # 全項目
    python experiments/exp_027/debug_o365.py --quick    # 1〜4 のみ（GPU 不要）

失敗した項目を明示して非ゼロ終了する。
"""
import argparse
import json
import os
import os.path as osp
import sys

REPO = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

O365_ROOT = '/workspace/kouyou/datasets/o365v1_stage/Objects365_v1/2019-08-02/'
RF100_ROOT = '/workspace/kouyou/datasets/rf100_domain/'
BUFFER = osp.join(REPO, 'experiments/exp_023/buffer/')
_TH0_NAME = ('grounding_dino_swin-t_pretrain_obj365_goldg_grit9m'
             '_v3det_20231204_095047-b448804b.pth')
# クラスタでは /workspace/kouyou/ckpt（sbatch.sh の bind 先）、本環境では torch hub
# キャッシュに置かれている。THETA0 env があればそれを最優先する。
THETA0 = os.environ.get('THETA0') or next(
    (p for p in (
        osp.join('/workspace/kouyou/ckpt', _TH0_NAME),
        osp.expanduser(osp.join('~/.cache/torch/hub/checkpoints', _TH0_NAME)),
    ) if osp.isfile(p)), osp.join('/workspace/kouyou/ckpt', _TH0_NAME))

# 各 t の config と、1 バッチに期待する内訳（design.md §3）
CASES = [
    ('underwater', 'experiments/exp_027/configs/fullft_replay_underwater.py',
     {'current': 4, 'reference': 2}),
    ('electromagnetic',
     'experiments/exp_027/configs/fullft_replay_electromagnetic.py',
     {'current': 4, 'reference': 1, 'past': 1}),
    ('videogames', 'experiments/exp_027/configs/fullft_replay_videogames.py',
     {'current': 4, 'reference': 1, 'past': 1}),
]

results = []


def check(name, fn):
    print(f'\n{"=" * 70}\n[{name}]\n{"=" * 70}')
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — 失敗を集約して最後にまとめて報告する
        import traceback
        traceback.print_exc()
        results.append((name, False, f'{type(e).__name__}: {e}'))
    else:
        results.append((name, True, ''))


def _load_odvg(path):
    """ODVG は 1 行 1 JSON（jsonl）。全行を読んで list で返す。"""
    with open(path) as f:
        head = f.read(1)
        f.seek(0)
        if head == '[':
            return json.load(f)
        return [json.loads(ln) for ln in f if ln.strip()]


# ---------------------------------------------------------------- 1. bind
def check_bind():
    train_dir = osp.join(O365_ROOT, 'train')
    print(f'Objects365 train ディレクトリ: {train_dir}')
    if not osp.isdir(train_dir):
        raise FileNotFoundError(
            f'{train_dir} が無い。sbatch.sh の --bind と、クラスタ側 '
            f'/home/kouyou/datasets/o365v1_stage の転送状況を確認すること。')
    n = 0
    for _ in os.scandir(train_dir):
        n += 1
        if n >= 5:
            break
    print(f'  読み取り可。エントリを {n} 件以上確認')
    lm = osp.join(O365_ROOT, 'o365v1_label_map.json')
    if not osp.isfile(lm):
        raise FileNotFoundError(f'label_map が無い: {lm}')
    print(f'  label_map OK: {lm} ({len(json.load(open(lm)))} クラス)')


# -------------------------------------------------------------- 2. buffer
def check_buffer():
    files = {
        'reference_o365v1_1000.odvg.json': 1000,
        'underwater_500.odvg.json': 500,
        'electromagnetic_500.odvg.json': 500,
        'videogames_500.odvg.json': 500,
    }
    for fn, expect in files.items():
        p = osp.join(BUFFER, fn)
        if not osp.isfile(p):
            raise FileNotFoundError(f'バッファ JSON が無い: {p}（git pull 済みか確認）')
        recs = _load_odvg(p)
        print(f'  {fn}: {len(recs)} 件 (期待 {expect})')
        if len(recs) != expect:
            raise ValueError(f'{fn} の件数が {len(recs)}、期待 {expect}')


# -------------------------------------------------------------- 3. exists
def _check_images(json_name, img_root, limit=None):
    recs = _load_odvg(osp.join(BUFFER, json_name))
    if limit:
        recs = recs[:limit]
    missing = []
    for r in recs:
        p = osp.join(img_root, r['filename'])
        if not osp.isfile(p):
            missing.append(p)
    print(f'  {json_name}: {len(recs) - len(missing)}/{len(recs)} 枚が実在')
    if missing:
        for p in missing[:5]:
            print(f'    欠損例: {p}')
        raise FileNotFoundError(
            f'{json_name} が指す画像が {len(missing)} 枚欠損（転送漏れの可能性）')


def check_exists():
    _check_images('reference_o365v1_1000.odvg.json', osp.join(O365_ROOT, 'train'))
    for dom in ('underwater', 'electromagnetic', 'videogames'):
        _check_images(f'{dom}_500.odvg.json', osp.join(RF100_ROOT, dom, 'train'))


# -------------------------------------------------------------- 4. decode
def check_decode():
    import mmcv
    recs = _load_odvg(osp.join(BUFFER, 'reference_o365v1_1000.odvg.json'))
    idxs = [0, len(recs) // 2, len(recs) - 1]
    for i in idxs:
        p = osp.join(O365_ROOT, 'train', recs[i]['filename'])
        img = mmcv.imread(p)
        if img is None:
            raise ValueError(f'デコードできない: {p}')
        print(f'  [{i}] {osp.basename(p)}  shape={img.shape}')


# -------------------------------------------------------------- 5. mixing
def check_mixing():
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    init_default_scope('mmdet')

    for dom, cfg_path, expect in CASES:
        cfg = Config.fromfile(osp.join(REPO, cfg_path))
        cfg.train_dataloader.num_workers = 0
        cfg.train_dataloader.persistent_workers = False
        loader = Runner.build_dataloader(cfg.train_dataloader)

        ds = loader.dataset            # トップの ConcatDataset
        bounds = list(ds.cumulative_sizes)   # ソース境界
        names = ['current', 'reference', 'past'][:len(bounds)]

        def source_of(gidx):
            for k, b in enumerate(bounds):
                if gidx < b:
                    return names[k]
            return 'unknown'

        counts = {}
        for gidx in next(iter(loader.batch_sampler)):
            counts[source_of(gidx)] = counts.get(source_of(gidx), 0) + 1
        print(f'  [{dom}] 1バッチの内訳: {counts}  (期待 {expect})')
        if counts != expect:
            raise ValueError(f'{dom} の混合比が {counts}、期待 {expect}')


# ---------------------------------------------------------------- 6. step
def check_step():
    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmengine.runner.checkpoint import load_checkpoint
    init_default_scope('mmdet')

    if not osp.isfile(THETA0):
        raise FileNotFoundError(f'θ0 が無い: {THETA0}（THETA0 env で指定可）')

    for tag, cfg_path in (
        ('条件A', 'experiments/exp_027/configs/fullft_replay_underwater.py'),
        ('条件B', 'experiments/exp_027/configs/condB_replay_underwater.py'),
    ):
        cfg = Config.fromfile(osp.join(REPO, cfg_path))
        cfg.model.backbone.init_cfg = None       # ドライバと同じ扱い
        cfg.train_dataloader.num_workers = 0
        cfg.train_dataloader.persistent_workers = False

        from mmdet.registry import MODELS
        model = MODELS.build(cfg.model)   # Runner は作らず、モデルのみ組む
        load_checkpoint(model, THETA0, map_location='cpu', strict=False, logger='current')
        model = model.cuda().train()

        loader = Runner.build_dataloader(cfg.train_dataloader)
        batch = next(iter(loader))
        data = model.data_preprocessor(batch, training=True)
        # GroundingDINO.loss(batch_inputs, batch_data_samples) の位置引数で渡す
        losses = model.loss(data['inputs'], data['data_samples'])
        total = sum(v.sum() for v in losses.values() if isinstance(v, torch.Tensor))
        print(f'  [{tag}] loss 合計 = {float(total):.4f}  (有限: {torch.isfinite(total).item()})')
        if not torch.isfinite(total):
            raise ValueError(f'{tag} の loss が非有限')

        # 実効 lr（凍結の確認）。Runner.build_optim_wrapper はインスタンスメソッドなので
        # mmengine.optim.build_optim_wrapper を直接使う。
        from mmengine.optim import build_optim_wrapper
        optim = build_optim_wrapper(model, cfg.optim_wrapper)
        lrs = {}
        for g in optim.optimizer.param_groups:
            lrs.setdefault(round(g['lr'], 10), 0)
            lrs[round(g['lr'], 10)] += len(g['params'])
        print(f'  [{tag}] 実効 lr ごとのパラメータ数: {lrs}')
        if tag == '条件B' and 0.0 not in lrs:
            raise ValueError('条件B なのに実効 lr = 0 のパラメータ群が無い')
        if tag == '条件A' and 0.0 in lrs:
            raise ValueError('条件A なのに実効 lr = 0 のパラメータ群がある')

        del model, loader, optim
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='1〜4 のみ（GPU 不要）')
    args = ap.parse_args()

    check('1. bind: Objects365 が見えるか', check_bind)
    check('2. buffer: バッファ JSON が読めるか', check_buffer)
    check('3. exists: バッファの画像が実在するか', check_exists)
    check('4. decode: 画像がデコードできるか', check_decode)
    if not args.quick:
        check('5. mixing: 1バッチの内訳が設計どおりか', check_mixing)
        check('6. step: θ0 で loss() が通り凍結が効くか', check_step)

    print(f'\n{"=" * 70}\n判定\n{"=" * 70}')
    ng = 0
    for name, ok, msg in results:
        print(f'  {"OK  " if ok else "NG  "}{name}' + (f'  <- {msg}' if msg else ''))
        ng += 0 if ok else 1
    print()
    if ng:
        print(f'{ng} 項目が NG。上記を解消してから本実行を投入すること。')
        sys.exit(1)
    print('全項目 OK。exp_027 の本実行を投入してよい。')


if __name__ == '__main__':
    main()
