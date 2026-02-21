# gpr/ -- ガウス過程回帰

[English version](../../../../src/gpr/README.md)

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

## デフォルトパラメータでの実行結果


### デフォルト設定

| パラメータ | 値 |
|:---|:---|
| カーネル | Matern-5/2 |
| noise_variance | 1e-6 |
| optimize | True |
| normalize_inputs | True |
| log_frequency | True |
| gp_mode | separate（実部・虚部を独立にフィッティング） |
| N_d（周波数点数） | 50 |
| T（観測時間） | 1時間 |
| サンプリングレート | 500 Hz |
| 周波数範囲 | [0.1, 250] Hz（対数間隔） |

### カーネル比較（N_d = 50、T = 1時間）

| 手法 | マルチサイン RMSE (x10^-2 rad) | 矩形波 RMSE (x10^-2 rad) |
|:---|:---:|:---:|
| **GPRカーネル** | | |
| DC | 7.54 | 15.8 |
| DI | 6.92 | 15.1 |
| Exponential | 16.7 | 36.2 |
| Matern-1/2 | 3.01 | 5.97 |
| Matern-3/2 | 2.94 | 6.04 |
| **Matern-5/2** | **2.90** | **5.89** |
| RBF | 3.05 | 5.96 |
| SS1（1次安定スプライン） | 3.01 | 5.97 |
| SS2（2次安定スプライン） | 5.59 | 8.22 |
| SSHF（高周波スプライン） | 3.44 | 6.31 |
| Stable Spline | 6.05 | 9.93 |
| **古典的手法** | | |
| LS（最小二乗法） | 9.79 | 26.9 |
| NLS（非線形最小二乗法） | **2.75** | **5.77** |

**Matern-5/2** がGPRカーネル中で最高精度（RMSE = 0.0290）を達成し、パラメトリックモデル構造を**必要とせず**にNLS（0.0275）に匹敵する。

### GP回帰出力

<table>
<tr>
<td align="center" width="33%">
<img src="../../../images/gp_real.png" alt="GP実部" width="280"><br>
<em>GP予測 -- G(jw)の実部</em>
</td>
<td align="center" width="33%">
<img src="../../../images/gp_imag.png" alt="GP虚部" width="280"><br>
<em>GP予測 -- G(jw)の虚部</em>
</td>
<td align="center" width="33%">
<img src="../../../images/gp_nyquist.png" alt="GPナイキスト" width="280"><br>
<em>GP予測 -- ナイキスト線図</em>
</td>
</tr>
</table>

<p align="center">
<img src="../../../images/gpr_nyquist_interpolation.png" alt="GPRナイキスト補間と不確かさ" width="500"><br>
<em>Matern-5/2 GPR補間と+/-2シグマ信頼帯（N_d = 50）</em>
</p>

### 周波数点数 N_d の影響（T = 60分）

| 手法 | N_d = 10 | N_d = 30 | N_d = 50 | N_d = 100 |
|:---|:---:|:---:|:---:|:---:|
| DI | **7.93** | **9.92** | 6.92 | 6.75 |
| Matern-3/2 | 8.57 | 18.1 | 2.94 | 4.29 |
| Matern-5/2 | 9.79 | 18.0 | **2.90** | 2.46 |
| RBF | 9.16 | 18.0 | 3.05 | 2.47 |
| SS1 | 9.13 | 22.4 | 3.01 | **2.45** |
| SSHF | 11.1 | 24.3 | 3.44 | 2.73 |
| NLS | 9.40 | 14.5 | **2.75** | **2.35** |

<sub>RMSE x 10^-2 [rad]（マルチサイン入力）。**太字** = 列内最良。</sub>

- **疎データ（N_d <= 30）**: **DIカーネル**が最良 -- 対角構造により遠い周波数点間の過外挿を回避
- **密データ（N_d >= 50）**: **Matern-5/2**がGPR中で最良; C2サンプルパスが物理的FRFの滑らかさに適合

### 観測時間 T の影響（N_d = 50）

| 手法 | 10分 | 30分 | 60分 | 600分 |
|:---|:---:|:---:|:---:|:---:|
| DI | 6.87 | 6.93 | 6.92 | 7.03 |
| Matern-3/2 | 3.23 | 2.98 | 2.94 | 3.98 |
| Matern-5/2 | 4.12 | **3.00** | **2.90** | 3.78 |
| RBF | **3.05** | 3.02 | 3.05 | **3.03** |
| SS1 | **3.00** | 3.01 | 3.01 | 3.04 |
| SSHF | 3.40 | 3.43 | 3.44 | 3.43 |
| NLS | **2.74** | **2.90** | **2.75** | **2.90** |

<sub>RMSE x 10^-2 [rad]（マルチサイン入力）。**太字** = 列内最良。</sub>

- **RBF**: 平滑化効果で高周波ノイズを抑制 -- 観測時間に対して安定
- **SS1**: 安定性事前分布が指数減衰を強制 -- 観測時間に対してロバスト
- **Matern-5/2**: 30--60分で最高精度を達成するが、観測時間に対する感度がやや高い
