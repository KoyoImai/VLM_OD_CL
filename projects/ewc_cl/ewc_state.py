"""EWC の状態（2 バッファ）と学習対象選択のヘルパ。

exp_042 design.md §2.2 の実装:

- 原論文のタスク別二次項 Σ_k (λ/2)·F_k·(θ−θ*_k)² を、二次式の和の等価変形
  （論文 §2「the sum of two quadratic penalties is itself a quadratic penalty」）で
  2 バッファに畳んで保持する:
      A = Σ_k F_k              （重要度の和）
      B = Σ_k F_k·θ*_k         （重要度で重み付けしたアンカーの和）
      const = Σ_k Σ_i F_k,i·θ*²_k,i   （スカラー。ペナルティ値を非負に保つ定数項）
  ペナルティ: (λ/2)·[Σ_i (A_i·θ_i² − 2·B_i·θ_i) + const] = Σ_k (λ/2)·F_k·(θ−θ*_k)²
  勾配: λ·(A·θ − B)。タスク数によらずメモリ一定で、原論文と数学的に等価。

- 学習対象（= EWC 対象）は構成要素（トップレベルモジュール名）単位で選択する。
  'all' なら全パラメータ。対象外は requires_grad=False で凍結する（design.md §2.2）。

状態ファイル形式（torch.save）:
    {'A': {name: fp32 CPU tensor}, 'B': {...}, 'const': float, 'meta': {...}}
"""
import torch

# 状態ファイルのフォーマット版数（後方互換の判定用）
STATE_FORMAT = 1


def target_param_names(model, target_components):
    """学習対象（= EWC 対象）のパラメータ名リストを返す。

    Args:
        target_components: 'all' または構成要素名（トップレベルモジュール名）のリスト。
            例: ['backbone', 'language_model', 'encoder', 'text_feat_map']
    """
    names = []
    for name, _ in model.named_parameters():
        top = name.split('.')[0]
        if target_components == 'all' or top in target_components:
            names.append(name)
    return names


def apply_target_selection(model, target_components):
    """対象は requires_grad=True、対象外は False にする。対象名リストを返す。

    構成要素名の誤記が黙って空選択（全凍結）になるのを防ぐため、存在しない名前と
    空の選択はエラーにする（2026-08-15 監査で追加）。
    """
    if target_components != 'all':
        tops = {n.split('.')[0] for n, _ in model.named_parameters()}
        unknown = sorted(set(target_components) - tops)
        assert not unknown, (
            f'存在しない構成要素: {unknown}。指定可能: {sorted(tops)}')
    targets = set(target_param_names(model, target_components))
    assert targets, 'EWC の学習対象が空です'
    for name, p in model.named_parameters():
        p.requires_grad_(name in targets)
    return sorted(targets)


def new_empty_state():
    return {'A': {}, 'B': {}, 'const': 0.0,
            'meta': {'format': STATE_FORMAT, 'num_tasks': 0, 'history': []}}


def load_state(path):
    state = torch.load(path, map_location='cpu')
    assert state.get('meta', {}).get('format') == STATE_FORMAT, \
        f'未知の状態フォーマット: {path}'
    return state


def save_state(state, path):
    torch.save(state, path)


@torch.no_grad()
def update_state(state, fisher, theta_star, task_meta=None):
    """タスク終了時の 2 バッファ更新: A += F、B += F·θ*、const += Σ F·θ*²。

    Args:
        state: 既存の状態（初回は new_empty_state()）
        fisher: {name: tensor}（対角経験 Fisher。非負）
        theta_star: {name: tensor}（タスク終了時のパラメータ値）
    """
    for name, f in fisher.items():
        f = f.detach().float().cpu()
        t = theta_star[name].detach().float().cpu()
        if name in state['A']:
            state['A'][name] += f
            state['B'][name] += f * t
        else:
            state['A'][name] = f.clone()
            state['B'][name] = f * t
        state['const'] += float((f * t * t).sum())
    state['meta']['num_tasks'] += 1
    state['meta']['history'].append(task_meta or {})
    return state


def penalty(state_gpu, named_params, lam):
    """2 バッファからペナルティ (λ/2)·Σ_k F_k(θ−θ*_k)² を計算する。

    Args:
        state_gpu: {'A': {name: tensor(デバイス上)}, 'B': {...}, 'const': float}
        named_params: model.named_parameters() のイテラブル（対象のみ渡すこと）
        lam: λ
    Returns:
        スカラー tensor（θ に対して微分可能。勾配は λ·(A·θ−B)）
    """
    acc = None
    for name, p in named_params:
        if name not in state_gpu['A']:
            continue
        a = state_gpu['A'][name]
        b = state_gpu['B'][name]
        term = (a * p * p).sum() - 2.0 * (b * p).sum()
        acc = term if acc is None else acc + term
    if acc is None:
        return None
    return 0.5 * lam * (acc + state_gpu['const'])
