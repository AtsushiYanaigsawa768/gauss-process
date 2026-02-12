# sample_data/ -- .mat ファイル形式

[English version](../../../../data/sample_data/)

## 概要

`data/sample_data/` にはフレキシブルリンク機構の計測データが
MATLAB .mat 形式で10ファイル格納されている。
各ファイルは約1時間の連続計測データを含む。

## ファイル一覧

| ファイル名 | 計測開始時刻 |
|-----------|-------------|
| `input_test_20250913_010037.mat` | 2025-09-13 01:00 |
| `input_test_20250913_030050.mat` | 2025-09-13 03:00 |
| `input_test_20250913_050103.mat` | 2025-09-13 05:00 |
| `input_test_20250913_070119.mat` | 2025-09-13 07:00 |
| `input_test_20250913_090135.mat` | 2025-09-13 09:00 |
| `input_test_20250913_110148.mat` | 2025-09-13 11:00 |
| `input_test_20250913_130201.mat` | 2025-09-13 13:00 |
| `input_test_20250913_150214.mat` | 2025-09-13 15:00 |
| `input_test_20250913_170227.mat` | 2025-09-13 17:00 |
| `input_test_20250913_190241.mat` | 2025-09-13 19:00 |

## データ形式

各 .mat ファイルは以下のいずれかの形式で時系列データを格納する。

### 形式A: 個別変数

| 変数名 | 説明 | 単位 |
|--------|------|------|
| `t` | 時間ベクトル | 秒 |
| `u` | 入力信号 (制御入力) | -- |
| `y` | 出力信号 (応答) | -- |

### 形式B: 3列配列

`output` または任意の3xN (またはNx3) 数値配列:

| 列 | 説明 |
|----|------|
| 1列目 | 時間 [s] |
| 2列目 | 出力信号 |
| 3列目 | 入力信号 |

## サンプリング情報

- サンプリング間隔: 非均一 (dtは一定でない場合がある)
- 計測時間: 各ファイル約1時間
- 全10ファイルで約10時間分のデータ

## 使用方法

```bash
# 1ファイルで周波数応答推定
python main.py data/sample_data/input_test_20250913_010037.mat --kernel rbf --out-dir output

# 全ファイルを使用
python main.py data/sample_data/*.mat --n-files 10 --kernel rbf --out-dir output

# FIR検証の参照データとして指定
python main.py data/sample_data/*.mat --kernel rbf \
    --extract-fir --fir-validation-mat data/sample_data/input_test_20250913_010037.mat
```

## 注意事項

- FIR検証に使用するファイルは、GP学習に使用したファイルと異なるものを推奨
- `--time-duration` オプションで各ファイルの使用時間を制限可能
