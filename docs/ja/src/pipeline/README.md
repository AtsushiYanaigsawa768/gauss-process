# pipeline/ -- パイプライン使用方法

[English version](../../../../src/pipeline/README.md)

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

## デフォルトパラメータでの実行結果

### 論文基準条件のコマンド

以下のコマンドで論文の基準条件（Matern-5/2、N_d = 50、T = 1時間）を再現できる:

```bash
python main.py data/sample_data/*.mat --kernel matern --nu 2.5 \
    --normalize --log-frequency --nd 50 --n-files 1 \
    --extract-fir --fir-length 1024 \
    --fir-validation-mat data/sample_data/input_test_20250913_010037.mat \
    --out-dir output
```

### エンドツーエンドの結果

| ステージ | 出力 | 主要指標 |
|:---|:---|:---|
| FRF推定 | 50周波数点、[0.1, 250] Hz | 同期復調法、対数間隔グリッド |
| GP回帰 | 平滑化されたG(jw) +/-2シグマ帯付き | 最良カーネル: Matern-5/2 |
| FIR検証 | 1024タップFIR係数 | RMSE = 0.0290 rad（マルチサイン）、0.0589 rad（矩形波） |

<p align="center">
<img src="../../../../docs/images/control_block_diagram.jpg" alt="制御ブロック図" width="450"><br>
<em>閉ループフィードバック制御系（P制御器、K_p = 1.65）</em>
</p>

### 包括テストの概要

`comprehensive_test.py` は全11カーネル + LS/NLS を複数の N_d（10, 30, 50, 100）と T（10分、30分、60分、600分）の組み合わせで評価する。結果は論文の表I--IIIと一致している:

- **最良GPRカーネル**: Matern-5/2（RMSE = 0.0290、N_d = 50、T = 60分）
- **最良古典的手法**: NLS（RMSE = 0.0275、モデル次数 n_b = 2, n_a = 4 が必要）
- **最もロバスト**: RBF と SS1（全観測時間に対して安定）

詳細なカーネル比較表は [gpr/README.md](../gpr/README.md) を参照。
