# gpr/ -- ガウス過程回帰

[English version](../../../../src/README_PIPELINE.md)

## 概要

`src/gpr/` はガウス過程回帰 (GPR) による周波数応答の平滑化を実装する。
14種類のカーネル、複数の外れ値ロバスト手法、グリッドサーチを提供する。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `kernels.py` | 14カーネル定義 + `create_kernel()` ファクトリ |
| `gpr_fitting.py` | `GaussianProcessRegressor` 本体 |
| `grid_search.py` | カーネル・ハイパーパラメータのグリッドサーチ |
| `visualization.py` | GP結果の可視化 (Bode/Nyquistプロット) |
| `itgp.py` | Iteratively-Trimmed GP (ITGP) |
| `t_distribution.py` | t分布尤度を用いたGP (gpflow) |
| `linear_interpolation.py` | 線形補間 (GPではないがベースライン比較用) |
| `pure_gp_kernels.py` | スクラッチ実装カーネル |
| `pure_gp_fitting.py` | スクラッチGP回帰 |
| `knn_noise_filter.py` | k-NN によるノイズフィルタリング |
| `least_squares.py` | 最小二乗法によるフィッティング |
| `descriptive_stats.py` | 記述統計 (平均、分散等) |

## カーネル一覧

| カーネル | クラス名 | 特徴 |
|---------|---------|------|
| RBF | `RBFKernel` | 汎用、滑らかな関数に適合 |
| Matern | `MaternKernel` | 滑らかさパラメータ nu で柔軟性を調整 |
| Rational Quadratic | `RationalQuadraticKernel` | 複数のスケールを持つ関数に対応 |
| Exponential | `ExponentialKernel` | Matern (nu=0.5) と等価 |
| TC | `TCKernel` | Tuned-Correlated カーネル |
| DC | `DCKernel` | Diagonal-Correlated カーネル |
| DI | `DIKernel` | Diagonal-Independent カーネル |
| Stable Spline (1st) | `FirstOrderStableSplineKernel` | 1次安定スプライン |
| Stable Spline (2nd) | `SecondOrderStableSplineKernel` | 2次安定スプライン |
| HF Stable Spline | `HighFrequencyStableSplineKernel` | 高周波安定スプライン |
| Stable Spline | `StableSplineKernel` | 統合スプラインカーネル |

## GP動作モード

| モード | 説明 | 設定 |
|-------|------|------|
| `separate` | 実部・虚部を独立にGPフィッティング | `--gp-mode separate` |
| `polar` | 対数振幅・位相を独立にGPフィッティング | `--gp-mode polar` |

## 使用例

```python
from src.gpr.kernels import create_kernel
from src.gpr.gpr_fitting import GaussianProcessRegressor

kernel = create_kernel('rbf', length_scale=1.0, variance=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, noise_variance=1e-6)
gpr.fit(X_train, y_train)
y_pred, y_var = gpr.predict(X_test, return_var=True)
```

## 外れ値ロバスト手法

- **ITGP** (`itgp.py`): Hampelフィルタで外れ値を反復的に除去しながらGPを再学習
- **t分布GP** (`t_distribution.py`): gpflowのStudentT尤度で重い裾を持つノイズに対応
- **k-NN フィルタ** (`knn_noise_filter.py`): k近傍法でノイズを事前除去

## グリッドサーチ

```bash
python main.py data/sample_data/*.mat --kernel rbf --grid-search --out-dir output
```

カーネル種類 x ハイパーパラメータの全組み合わせを評価し、最良モデルを選択する。
