# 実験10

## 位置付け
ZiRaとDitHubのRoboflow100の学習と評価

## 目的
ZiRaとDitHubの結果取得

## 実験種別
確認実験

## 実験条件
- ZiRa：従来手法 ZiRa
- DitHub：従来手法 DitHub
- ZiRa + replay：リプレイを導入した ZiRa
- DitHub + replay：リプレイを導入した DitHub


## 結果

## 結果

---
**t=1: underwater 学習後**
|                    | underwater |  zcoco |
|--------------------|------------|--------|
| ZiRa               |   0.206    |  0.493 |
| ZiRa + replay      |   0.201    |  0.500 |
| DitHub             |   0.215    |  0.504 |
| DitHub + replay    |   0.215    |  0.504 |

---

**t=2: electromagnetic 学習後**
|                    | electromagnetic | underwater |  zcoco |
|--------------------|-----------------|------------|--------|
| ZiRa               |      0.221      |   0.146    |  0.463 |
| ZiRa + replay      |      0.195      |   0.196    |  0.490 |
| DitHub             |      0.215      |   0.142    |  0.435 |
| DitHub + replay    |      0.179      |   0.106    |  0.489 |

---

**t=3: videogames 学習後**
|                    | videogames | underwater | electromagnetic |  zcoco |
|--------------------|------------|------------|-----------------|--------|
| ZiRa               |   0.115    |   0.111    |      0.139      |  0.418 |
| ZiRa + replay      |   0.100    |   0.192    |      0.172      |  0.486 |
| DitHub             |   0.111    |   0.063    |      0.211      |  0.422 |
| DitHub + replay    |   0.089    |   0.037    |      0.194      |  0.484 |

---



## 備考
### 実験結果ディレクトリ
exp_039