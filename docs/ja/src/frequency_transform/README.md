# frequency_transform/ -- 周波数応答推定

[English version](../../../../src/README_PIPELINE.md)

## 概要

`src/frequency_transform/` は時系列 .mat ファイルから周波数応答関数 (FRF) を推定する。
2つの推定手法と、ディスクベースのキャッシュ機構を提供する。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `frf_estimator.py` | 同期復調法による FRF 推定 (対数周波数グリッド) |
| `fourier_estimator.py` | FFT による伝達関数推定 (線形周波数グリッド) |
| `cache.py` | SHA-256ハッシュベースのディスクキャッシュ |
| `transform.py` | 統合インターフェース (手法切替・キャッシュ制御) |

## 手法比較: FRF vs FFT

| 項目 | FRF (`frf`) | FFT (`fourier`) |
|------|-------------|-----------------|
| 周波数グリッド | 対数スケール (MATLAB互換) | 線形スケール |
| 推定方法 | 台形積分による同期復調 | numpy FFT |
| 集約方法 | クロスパワー平均 `G = sum(Y*conj(U)) / sum(|U|^2)` | 複数推定値の平均 |
| MATLAB互換 | 高 (位相符号規約を含む) | 標準的 |
| 適用場面 | 低周波の分解能が必要な場合 | 均一な周波数分解能が必要な場合 |

## 統合インターフェース

`transform.py` の `run_frequency_analysis()` が手法を切り替える。

```python
from src.frequency_transform.transform import run_frequency_analysis

# FRF手法 (デフォルト)
omega, G = run_frequency_analysis(mat_files, method='frf', nd=100)

# FFT手法
omega, G = run_frequency_analysis(mat_files, method='fourier', nd=100)
```

### デフォルト定数

| 定数 | 値 | CLI フラグ |
|------|-----|-----------|
| `DEFAULT_ND` | 100 | `--nd` |
| `DEFAULT_F_LOW_LOG10` | -1.0 | -- |
| `DEFAULT_F_UP_LOG10` | 2.3 | -- |

## キャッシュ機構

`cache.py` の `FrequencyDataCache` は計算済み FRF を CSV として保存し、
同一設定での再計算を回避する。

```
.cache/frequency_data/
└── <sha256_hash>.csv    # 設定のハッシュ値でファイル名を決定
```

キャッシュキーは、入力ファイルリスト・周波数パラメータ・推定手法の
組み合わせから決定論的に生成される。

## 入力データ形式

.mat ファイルは以下のいずれかの形式に対応:

1. 変数 `t`, `u`, `y` が個別に格納されている場合
2. `[time, output, input]` の3列 (または3行) 配列が格納されている場合
