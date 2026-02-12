# pipeline/ -- パイプライン使用方法

[English version](../../../../src/README_PIPELINE.md)

## 概要

`src/pipeline/` はシステム同定の全工程をオーケストレーションするモジュールである。
CLIエントリポイントから設定の解析、データ読み込み、GP回帰、FIR抽出までを統合する。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `config.py` | Dataclassベースの設定 (`GPConfig`, `FIRConfig`, `FrequencyConfig`) |
| `data_loader.py` | データ読み込み (FRF/FFT切替、.matファイル処理) |
| `gp_pipeline.py` | メインパイプラインオーケストレータ |
| `unified_pipeline.py` | CLIエントリポイント (argparse) |
| `comprehensive_test.py` | バッチテスト (全カーネル x 全設定) |
| `gp_fir_legacy.py` | レガシーパイプライン (後方互換) |

## 設定体系

3つのdataclassで設定を管理する。

### GPConfig

| パラメータ | デフォルト | CLI フラグ |
|-----------|----------|-----------|
| `kernel_type` | `'rbf'` | `--kernel` |
| `noise_variance` | `1e-6` | `--noise-variance` |
| `optimize` | `True` | `--no-optimize` |
| `n_restarts` | `3` | `--n-restarts` |
| `normalize_inputs` | `True` | `--normalize` |
| `log_frequency` | `True` | `--log-frequency` |
| `gp_mode` | `'separate'` | `--gp-mode` |

### FIRConfig

| パラメータ | デフォルト | CLI フラグ |
|-----------|----------|-----------|
| `extract_fir` | `False` | `--extract-fir` |
| `fir_length` | `1024` | `--fir-length` |
| `validation_mat` | `None` | `--fir-validation-mat` |

### FrequencyConfig

| パラメータ | デフォルト | CLI フラグ |
|-----------|----------|-----------|
| `n_files` | `1` | `--n-files` |
| `time_duration` | `None` | `--time-duration` |
| `nd` | `100` | `--nd` |
| `freq_method` | `'frf'` | `--freq-method` |

## 実行例

```bash
# 基本: RBFカーネルでGP回帰
python main.py data/sample_data/*.mat --kernel rbf --normalize --log-frequency --out-dir output

# FIR抽出付き
python main.py data/sample_data/*.mat --kernel rbf --normalize \
    --extract-fir --fir-length 1024 \
    --fir-validation-mat data/sample_data/input_test_20250913_010037.mat \
    --out-dir output

# FFT手法を使用
python main.py data/sample_data/*.mat --freq-method fourier --nd 100 \
    --kernel rbf --normalize --out-dir output_fourier

# グリッドサーチ
python main.py data/sample_data/*.mat --kernel rbf --normalize --grid-search \
    --extract-fir --fir-length 1024 --out-dir output_grid
```

## 出力構造

```
output/
├── *_frf.csv, *_frf.mat        周波数応答データ
├── *_bode_*.png, *_nyquist.png Bode/Nyquistプロット
├── gp_*.png                    GP可視化
├── gp_smoothed_frf.csv         GP平滑化済みFRF
├── gp_results.json             GPパラメータ・評価指標
└── fir_gp/                     FIR結果
    ├── fir_coefficients_gp.npz
    ├── fir_gp_results.json
    └── gp_fir_results.png
```

## バッチテスト

`comprehensive_test.py` は全カーネルと設定の組み合わせを自動テストする。

```bash
python -m src.pipeline.comprehensive_test
```
