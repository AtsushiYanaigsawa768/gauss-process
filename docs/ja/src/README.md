# src/ -- アーキテクチャ概要

[English version](../../../src/README_PIPELINE.md)

## 概要

`src/` はフレキシブルリンク機構のシステム同定パイプラインを構成するPythonパッケージである。
周波数応答データの推定からガウス過程回帰 (GPR)、FIRモデル同定までをエンドツーエンドで実行する。

## モジュールマップ

```
src/
├── utils/                 共有ユーティリティ (ハンペルフィルタ、データI/O)
├── visualization/         プロット補助 (スタイル設定、時系列プロット)
├── frequency_transform/   周波数応答推定 (FRF同期復調法、FFT、キャッシュ)
├── gpr/                   ガウス過程回帰 (14カーネル、ITGP、t分布GP 等)
├── fir_model/             FIRモデル同定 (GP→FIR変換、LMS、RLS 等)
├── classical_methods/     古典的手法 (NLS、LS、IWLS、TLS、ML、RF、GBR、SVM)
├── pipeline/              パイプラインオーケストレータ (設定、CLI、バッチテスト)
├── examples/              サンプルスクリプト
└── tests/                 テストスクリプト
```

## データフロー

```
.matファイル (時系列)
    │
    ▼
frequency_transform/    周波数応答推定 (FRF or FFT)
    │
    ▼
gpr/                    GP回帰で平滑化
    │
    ▼
fir_model/              GP予測 → IFFT → FIR係数
    │
    ▼
検証                    時間領域シミュレーションで精度評価 (RMSE, R^2, FIT%)
```

## クイックスタート

```bash
# 環境構築
conda create --name GaussProcess python=3.11
pip install -r requirements.txt

# 基本実行 (RBFカーネル、FIR抽出あり)
python main.py data/sample_data/*.mat --kernel rbf --normalize --log-frequency \
    --extract-fir --fir-length 1024 --out-dir output

# 既存のFRFデータを使用
python main.py --use-existing output/matched_frf.csv --kernel matern --nu 2.5

# グリッドサーチでカーネル最適化
python main.py data/sample_data/*.mat --kernel rbf --normalize --grid-search \
    --extract-fir --fir-length 1024 --out-dir output_grid
```

## エントリポイント

`main.py` が `src.pipeline.unified_pipeline.main()` を呼び出す。
CLI引数は `src/pipeline/config.py` の3つのdataclass (`GPConfig`, `FIRConfig`, `FrequencyConfig`) にマッピングされる。

## 主要な依存ライブラリ

| ライブラリ | 用途 |
|-----------|------|
| numpy / scipy | 数値計算、FFT、最適化 |
| scikit-learn | StandardScaler、RF/GBR/SVM |
| matplotlib | 可視化 |
| robustgp | ITGP (Iteratively-Trimmed GP) |
| gpflow | t分布GP |

## 各モジュールの詳細

- [visualization/](./visualization/) -- プロット機能
- [frequency_transform/](./frequency_transform/) -- 周波数応答推定
- [gpr/](./gpr/) -- ガウス過程回帰
- [fir_model/](./fir_model/) -- FIRモデル同定
- [classical_methods/](./classical_methods/) -- 古典的システム同定手法
- [pipeline/](./pipeline/) -- パイプライン使用方法
