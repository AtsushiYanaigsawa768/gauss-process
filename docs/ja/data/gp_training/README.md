# gp_training/ -- .dat ファイル形式

[English version](../../../../data/gp_training/)

## 概要

`data/gp_training/` にはMATLABで事前計算された周波数応答データが
.dat 形式で4ファイル格納されている。
GP回帰に直接入力可能な Bode プロット形式のデータである。

## ファイル一覧

| ファイル名 | 計測日 |
|-----------|--------|
| `SKE2024_data16-Apr-2025_1819.dat` | 2025-04-16 |
| `SKE2024_data18-Apr-2025_1205.dat` | 2025-04-18 |
| `SKE2024_data07-May-2025_1825.dat` | 2025-05-07 |
| `SKE2024_data09-May-2025_1421.dat` | 2025-05-09 |

## データ形式

カンマ区切りの3行テキストファイル。各行の意味は以下の通り。

| 行 | 内容 | 単位 |
|----|------|------|
| 1行目 | 角周波数 omega | rad/s |
| 2行目 | ゲイン |G(jw)| | -- (無次元) |
| 3行目 | 位相角 | rad |

### 読み込み例

```python
import numpy as np

data = np.loadtxt("data/gp_training/SKE2024_data07-May-2025_1825.dat", delimiter=",")
omega = data[0]   # 角周波数 [rad/s]
mag   = data[1]   # ゲイン |G(jw)|
phase = data[2]   # 位相 [rad]
```

また、`src/utils/data_io.py` のヘルパー関数も利用可能:

```python
from src.utils.data_io import load_bode_data
from pathlib import Path

omega, mag, phase = load_bode_data(Path("data/gp_training/SKE2024_data07-May-2025_1825.dat"))
```

## 使用方法

`--use-existing` オプションで .mat ファイルの処理をスキップし、
直接GP回帰に入力する。

```bash
python main.py --use-existing data/gp_training/SKE2024_data07-May-2025_1825.dat \
    --kernel rbf --normalize --log-frequency --out-dir output
```

## 注意事項

- 行方向にデータが格納されている (列方向ではない)
- MATLABの位相符号規約: `atan2(-ImG, -ReG)` で計算されている
- 4ファイルは異なる日時の実験データであり、条件の再現性検証に使用可能
