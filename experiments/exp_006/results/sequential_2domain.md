# exp_006 結果：2ドメイン逐次学習（D1=aerial → D2）

- 設定: backbone・言語凍結 / lr=1e-4 / 実効64(per-GPU8×4×累積2) / 20ep / seed=0 / 4GPU。
  D2 学習は aerial(D1) チェックポイント（exp_005 ep17）から継続。
- 評価: (800,1333) プロトコル。各 D2 学習後に aerial(D1) / D2 / COCO を評価。

## 基準値
- aerial 単独FT後: **aerial=0.468**, COCO=0.318（exp_005）。COCO zero-shot=0.504（exp_001）。
- 各 D2 単独FT適応: underwater0.316 / microscopic0.499 / videogames0.719 / documents0.478 / electromagnetic0.454。

## 逐次学習後の結果（aerial→D2）

| D2 | aerial(D1) | D1忘却 vs 0.468 | D2適応 | (D2単独FT) | COCO | COCO vs aerial後0.318 |
|---|---:|---:|---:|---:|---:|---:|
| videogames | 0.241 | −0.227（−48.5%） | 0.698 | (0.719) | 0.214 | −0.104 |
| microscopic | 0.127 | −0.341（−72.9%） | 0.503 | (0.499) | 0.163 | −0.155 |
| underwater | 0.113 | −0.355（−75.9%） | 0.313 | (0.316) | **0.355** | **+0.037**↑ |
| documents | 0.099 | −0.369（−78.8%） | 0.481 | (0.478) | 0.128 | −0.190 |
| electromagnetic | 0.147 | −0.321（−68.6%） | 0.466 | (0.454) | 0.216 | −0.102 |

### 「D2として学習」vs「D1(単独)として学習」の同一ドメイン精度差（forward transfer）
各ドメインを D2 として学習（aerial→D2）した時の自ドメイン精度と、D1 として単独学習
（事前学習起点の単独FT, exp_005/exp_004）した時の自ドメイン精度の差。
「aerial を先に学習しておくことが、後続ドメインの適応に与える影響」を表す。

| ドメイン | D2として学習(aerial→D2) | D1(単独)として学習 | 精度差(D2−D1) |
|---|---:|---:|---:|
| underwater | 0.313 | 0.316 | −0.003 |
| microscopic | 0.503 | 0.499 | +0.004 |
| videogames | 0.698 | 0.719 | −0.021 |
| documents | 0.481 | 0.478 | +0.003 |
| electromagnetic | 0.466 | 0.454 | +0.012 |

- 差はいずれも極めて小さい（−0.021〜+0.012, ほぼ ±0.01）。
- → **同じドメインは、D1 として単独で学んでも D2 として（aerial の後に）学んでも、適応精度はほぼ同じ**。
  aerial の事前学習による forward transfer（後続ドメインへの正/負の転移）はほぼ無い。
- これは「忘却は過去方向(D1)にのみ強く起き、新ドメイン方向(D2)の学習は阻害されない」ことを示す。
- 注: underwater の「D1単独」基準は exp_004（seed 未固定）、他は exp_005（seed=0）。差が微小なため結論に影響なし。

## 主要な観察
1. **D1(aerial) の忘却は全ケースで発生**: 0.468 → 0.099〜0.241（−48.5%〜−78.8%）。
   過去に学習したドメインの破滅的忘却を実証。
2. **D2 適応は単独FTとほぼ同等**: aerial 起点でも D2 への適応性能は劣化しない（差±0.02 程度）。
   → 新ドメインは問題なく学習できる。失われるのは過去ドメイン(D1)と汎用(COCO)。
3. **COCO 累積忘却は概ね深化**（aerial後0.318 → 0.13〜0.22）。ただし **underwater のみ COCO 回復**
   （0.318→0.355）。underwater は COCO に近いドメインで、D2学習が共有検出経路を COCO 寄りに
   戻すため（exp_005 でも underwater は最小忘却だった知見と整合）。
4. **D2 が COCO から遠いほど D1・COCO とも忘却が大きい傾向**: documents（最遠）が D1忘却最大・COCO最低。

## copypaste（参考, COCO累積忘却評価 bbox）
- aerial→videogames:      `coco/bbox_mAP 0.214`
- aerial→microscopic:     `coco/bbox_mAP 0.163`
- aerial→underwater:      `coco/bbox_mAP 0.355`
- aerial→documents:       `coco/bbox_mAP 0.128`
- aerial→electromagnetic: `coco/bbox_mAP 0.216`
