"""DualGPM（InfLoRA の勾配部分空間メモリ）と次元削減行列 A の設計。

公式実装 liangyanshuo/InfLoRA `methods/inflora.py` の移植（exp_042 design.md §2.3）:
    - update_DualGPM: 層ごとの入力共分散（活性の行列）を SVD し、累積寄与率の閾値
      （threshold = lamb + (lame−lamb)·t/T の線形増加）で基底を拡張する。
      メモリは 'remove'（過去空間そのものを保持し、設計時に射影で除く）と
      'retain'（直交補空間を保持し、設計時に射影で残す）の双対表現で、
      層次元の半分以下に保つ。
    - A の設計: 学習前に共分散をメモリで射影（remove: M M^T を引く / retain: M M^T を
      掛ける）してから SVD し、上位 r 主成分の転置を 1/√3 倍して A に固定する。

公式との対応を優先し、ロジックは numpy で公式コードの形をほぼ保って移植している。
メモリファイル形式（torch.save）:
    {'feature': {layer_name: np.ndarray (d, k)},
     'ptype': {layer_name: 'remove' | 'retain'},
     'meta': {...}}
"""
import math

import numpy as np
import torch

MEMORY_FORMAT = 1


def new_empty_memory():
    return {'feature': {}, 'ptype': {},
            'meta': {'format': MEMORY_FORMAT, 'num_tasks': 0, 'history': []}}


def load_memory(path):
    mem = torch.load(path, map_location='cpu')
    assert mem.get('meta', {}).get('format') == MEMORY_FORMAT, \
        f'未知のメモリフォーマット: {path}'
    return mem


def save_memory(mem, path):
    torch.save(mem, path)


def threshold_at(task_index, total, lamb, lame):
    """公式: threshold = (lame − lamb)·t/T + lamb（t は 0 始まりのタスク番号）。"""
    return (lame - lamb) * task_index / total + lamb


def design_A(cov, r, feature=None, ptype=None):
    """学習前の A の設計（公式 _train の SVD 部）。

    Args:
        cov: (d, d) 入力共分散（torch or np）
        r: LoRA ランク
        feature: メモリの基底 (d, k)。無ければ生の共分散を使う（t=1）
        ptype: 'remove' | 'retain'
    Returns:
        (r, d) numpy 配列。lora_A にコピーして固定する（公式は U[:, :r].T / sqrt(3)）
    """
    mat = cov.detach().cpu().numpy() if torch.is_tensor(cov) else np.asarray(cov)
    mat = mat.astype(np.float32)   # 公式は float32 の numpy SVD
    if feature is not None:
        M = np.asarray(feature, dtype=np.float32)
        if ptype == 'remove':
            mat = mat - M @ (M.T @ mat)
        else:
            assert ptype == 'retain', ptype
            mat = M @ (M.T @ mat)
    # 縮退ガード: retain 型のメモリは新タスクの部分空間に食われて列が減っていき、
    # ゼロ列（干渉しない方向が残っていない）まで行き着き得る（公式コードも同挙動）。
    # その場合の射影は零行列で、SVD から得る A は無意味な方向になるため明示的に止める。
    assert np.linalg.norm(mat) > 0, (
        'A の設計元行列が零行列（DualGPM メモリが全方向を覆い、干渉しない方向が'
        '残っていない）。threshold（lamb/lame）を下げるか設定を見直すこと。')
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    A = (U[:, :r].T / math.sqrt(3)).astype(np.float32)
    # 【公式コードとの意図的な差分】実効ランク（S > 1e-6·S[0]）が r に満たない場合、
    # 不足分の U の列は零特異値側の「任意の直交補完」で、メモリと直交する保証も
    # タスクの勾配空間に入る保証も無い方向になる（2026-08-15 実測: 1 クラスタスクでは
    # プロンプトが全画像で同一のため BERT 層の共分散ランクが 3 程度しかなく、
    # 該当行がメモリ方向に整列した）。ViT 分類の公式実装では入力の多様性から
    # ランク不足が起きず、この領域は未定義。干渉フリーの意図に従い、実効ランクを
    # 超える行は 0 にする（その成分の ΔW=0。B の対応列は不活性になる）。
    # 閾値は相対 1e-3。fp32 のランク落ち行列では数値ノイズの特異値が相対 1e-5 前後に
    # 出るため（実測: BERT 層で S = [60.78, 10.86, 0.24, ~4e-3(ノイズ), ...]）、
    # 1e-6 ではノイズ方向を有効と誤判定する。1e-3 は実信号（この例で相対 4e-3）と
    # ノイズを両側に余裕を持って分離し、切り捨てる方向の分散は最上位の 0.1% 未満。
    erank = int((S > S[0] * 1e-3).sum()) if S[0] > 0 else 0
    if erank < r:
        A[erank:] = 0.0
    return A


def update_dual_gpm(covs, mem, threshold):
    """タスク終了時のメモリ更新（公式 update_DualGPM の移植。層はリストでなく名前で持つ）。

    Args:
        covs: {layer_name: (d, d) 共分散}（torch or np）
        mem: new_empty_memory() 形式
        threshold: threshold_at() の値
    """
    feature, ptype = mem['feature'], mem['ptype']
    for name, cov in covs.items():
        activation = (cov.detach().cpu().numpy() if torch.is_tensor(cov)
                      else np.asarray(cov)).astype(np.float32)
        if name not in feature:
            # 最初のタスク（公式: After First Task）
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = int(np.sum(np.cumsum(sval_ratio) < threshold))
            if r < (activation.shape[0] / 2):
                feature[name] = U[:, 0:max(r, 1)]
                ptype[name] = 'remove'
            else:
                # 【公式コードとの意図的な差分】公式はここでも U[:, :r] を保存するが、
                # 'retain' の意味（過去空間の**直交補**を保持し、設計時に M M^T を掛けて
                # 残す）と矛盾し、直後の整合 assert（retain は層次元の半分以下）とも
                # 両立しない。ViT では活性の集中により r < d/2 に収まりこの分岐は実行
                # されないと考えられる（未実行の潜在バグ）。双対表現の定義に従い
                # 直交補を保存する。threshold < 1 なら r <= d-1 で直交補は空にならない。
                feature[name] = U[:, r:]
                ptype[name] = 'retain'
        elif ptype[name] == 'remove':
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1 ** 2).sum()
            M = feature[name]
            act_hat = activation - M @ (M.T @ activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
            sval_hat = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total
            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                continue          # 公式: Skip Updating DualGPM for layer
            Ui = np.hstack((M, U[:, 0:r]))
            if Ui.shape[1] > Ui.shape[0]:
                feature[name] = Ui[:, 0:Ui.shape[0]]
            else:
                feature[name] = Ui
        else:
            assert ptype[name] == 'retain'
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1 ** 2).sum()
            M = feature[name]
            act_hat = M @ (M.T @ activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
            sval_hat = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            accumulated_sval = sval_hat / sval_total
            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval >= (1 - threshold):
                    accumulated_sval -= sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                continue
            act_feature = M - U[:, 0:r] @ (U[:, 0:r].T @ M)
            Ui, Si, Vi = np.linalg.svd(act_feature)
            feature[name] = Ui[:, :M.shape[1] - r]

    # 公式: Gradient Constraints Summary の調整
    # remove 型が層次元の半分を超えたら直交補空間に切り替えて retain 型へ
    for name in list(feature):
        M = feature[name]
        if ptype[name] == 'remove' and (M.shape[1] > (M.shape[0] / 2)):
            U, S, V = np.linalg.svd(M)
            feature[name] = U[:, M.shape[1]:]
            ptype[name] = 'retain'
        elif ptype[name] == 'retain':
            assert M.shape[1] <= (M.shape[0] / 2), \
                f'{name}: retain 型の基底が層次元の半分を超えている'
    return mem
