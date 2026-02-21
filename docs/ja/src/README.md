# src/ -- アーキテクチャ概要

[English version](../../../src/README.md)

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

## デフォルトパラメータ（論文基準条件）

| パラメータ | 値 |
|:---|:---|
| 対象プラント | Quanser Rotary Flexible Link |
| 制御器 | P制御器 (K_p = 1.65) |
| サンプリングレート | 500 Hz (dt = 0.002 s) |
| 周波数範囲 | [0.1, 250] Hz（対数間隔） |
| N_d（周波数点数） | 50 |
| T（観測時間） | 1時間 |
| 最良GPRカーネル | Matern-5/2 |
| FIR長 | 1024タップ |
| 学習信号 | マルチサイン |
| 検証信号 | ランダム矩形波 |

<p align="center">
<img src="../../images/control_block_diagram.jpg" alt="制御ブロック図" width="450"><br>
<em>閉ループフィードバック制御系（P制御器、K_p = 1.65）</em>
</p>

## 結果概要

| | 結果 |
|:---|:---|
| **最良GPRカーネル** | Matern-5/2: RMSE = 0.0290（マルチサイン）、0.0589（矩形波） |
| **最良古典的手法** | NLS: RMSE = 0.0275（マルチサイン）、0.0577（矩形波） |
| **主要な利点** | GPRはパラメトリックモデル構造**なし**でNLSに匹敵する精度を達成 |
| **疎データでの最良** | DIカーネルがN_d <= 30で最良 |
| **最もロバスト** | RBFとSS1が全観測時間で安定した精度を維持 |

<p align="center">
<img src="../../images/flexlink.jpg" alt="Quanser Rotary Flexible Link" width="400"><br>
<em>Quanser Rotary Flexible Link -- 実験装置</em>
</p>

各モジュールの詳細結果:
- [gpr/](./gpr/) -- カーネル比較表、N_dおよびTの影響分析
- [fir_model/](./fir_model/) -- 時間領域FIR検証
- [frequency_transform/](./frequency_transform/) -- FRF推定出力（ボード/ナイキスト線図）
- [classical_methods/](./classical_methods/) -- LS/NLS比較
- [visualization/](./visualization/) -- 入出力信号例
- [pipeline/](./pipeline/) -- エンドツーエンドのパイプライン結果

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
