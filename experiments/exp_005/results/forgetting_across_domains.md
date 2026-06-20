# exp_005 結果：他ドメイン凍結 fine-tune による COCO 忘却（普遍性確認）

- 設定: backbone・言語エンコーダ 凍結、lr=1e-4、実効バッチ64（per-GPU8×4GPU×累積2）、
  20 epochs、seed=0/deterministic=False、4GPU 分散（exp_004 と同一設定）。
- 評価: (800,1333) プロトコル。各ドメイン自身（適応）＋ COCO（忘却）。
- underwater は exp_004 の値を再掲（同一設定）。

## 適応 ＆ COCO 忘却（6ドメイン）

| ドメイン | 適応 zero-shot | 適応 凍結FT | COCO zero-shot | COCO 凍結FT後 | 忘却量(絶対) | 忘却率 |
|---|---:|---:|---:|---:|---:|---:|
| underwater (exp_004) | 0.051 | 0.316 | 0.504 | 0.414 | −0.090 | **−17.9%** |
| electromagnetic | 0.024 | 0.454 | 0.504 | 0.331 | −0.173 | −34.3% |
| microscopic | 0.002 | 0.499 | 0.504 | 0.327 | −0.177 | −35.1% |
| videogames | 0.016 | 0.719 | 0.504 | 0.324 | −0.180 | −35.7% |
| aerial | 0.034 | 0.468 | 0.504 | 0.318 | −0.186 | −36.9% |
| documents | 0.006 | 0.478 | 0.504 | 0.275 | −0.229 | **−45.4%** |

→ **全6ドメインで COCO mAP が低下**（−17.9% 〜 −45.4%）。忘却はドメインに依らず普遍的に発生。

## 各ドメインの copypaste（COCO 忘却評価, bbox）
- underwater(exp_004): `0.414 0.570 0.448 0.252 0.443 0.568`
- aerial:          `0.318 0.447 0.347 0.194 0.381 0.452`
- microscopic:     `0.327 0.503 0.352 0.193 0.358 0.457`
- videogames:      `0.324 0.460 0.352 0.179 0.352 0.464`
- documents:       `0.275 0.445 0.287 0.171 0.315 0.387`
- electromagnetic: `0.331 0.464 0.358 0.212 0.359 0.459`

## 各ドメインの適応 copypaste（自ドメイン valid, bbox）
- underwater(exp_004): `0.316 0.487 0.331 0.159 0.323 0.423`
- aerial:          `0.468 0.752 0.520 0.460 0.433 0.305`
- microscopic:     `0.499 0.745 0.546 0.281 0.420 0.581`
- videogames:      `0.719 0.858 0.799 0.297 0.719 0.571`
- documents:       `0.478 0.689 0.512 0.258 0.405 0.572`
- electromagnetic: `0.454 0.739 0.490 0.242 0.447 0.598`

## best epoch
underwater ep20 / aerial ep17 / microscopic ep20 / videogames ep20 / documents ep19 / electromagnetic ep18
