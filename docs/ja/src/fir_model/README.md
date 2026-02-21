# fir_model/ -- FIRモデル同定

[English version](../../../../src/fir_model/README.md)

## 概要

`src/fir_model/` はGPで平滑化された周波数応答からFIR (Finite Impulse Response) フィルタ係数を抽出し、時間領域での検証を行う。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `fir_fitting.py` | GP → FIR変換パイプライン (メイン) |
| `fir_helpers.py` | DFTユーティリティ (均一グリッド生成、Hermitian対称化、IDFT) |
| `fir_validation.py` | 時間領域検証 (RMSE, R^2, FIT%) |
| `fir_legacy.py` | レガシー IRFFT ベースの変換 |
| `kernel_regularized.py` | カーネル正則化FIR推定 |
| `lms_filter.py` | LMS (Least Mean Squares) 適応フィルタ |
| `rls_filter.py` | RLS (Recursive Least Squares) 適応フィルタ |
| `partial_update_lms.py` | 部分更新LMSフィルタ |

## FIR抽出プロセス

```
GP予測 (omega, G_complex)
    │
    ▼
1. 均一周波数グリッドへの補間
    │
    ▼
2. Hermitian対称化: G(-omega) = conj(G(omega))
    │
    ▼
3. IFFT (逆フーリエ変換)
    │
    ▼
FIR係数 (デフォルト: 1024タップ)
```

### 2つのモード

| モード | 説明 |
|-------|------|
| Paper mode | 均一omega + 両側Hermitian + IDFT (論文準拠) |
| Legacy mode | numpy `irfft` ベース |

## 適応フィルタ

| 手法 | ファイル | 特徴 |
|------|---------|------|
| LMS | `lms_filter.py` | 低計算コスト、収束が遅い |
| RLS | `rls_filter.py` | 高速収束、計算コスト高 |
| 部分更新LMS | `partial_update_lms.py` | LMSの計算量を削減 |

## カーネル正則化FIR

`kernel_regularized.py` はGPカーネルによる正則化を用いて、
少数のデータ点からでもFIR係数を安定に推定する。

## 使用例

```bash
# パイプライン経由でFIR抽出
python main.py data/sample_data/*.mat --kernel rbf --normalize \
    --extract-fir --fir-length 1024 \
    --fir-validation-mat data/sample_data/input_test_20250913_010037.mat \
    --out-dir output
```

## 検証指標

| 指標 | 説明 |
|-----|------|
| RMSE | 二乗平均平方根誤差 |
| R^2 | 決定係数 |
| FIT% | `100 * (1 - NRMSE)` |

検証は `fir_validation.py` が .mat ファイルの時系列データを用いて実施する。

## デフォルトパラメータでの実行結果


### デフォルト設定

| パラメータ | 値 |
|:---|:---|
| FIR長 | 1024タップ |
| 上流GPカーネル | Matern-5/2 |
| N_d（周波数点数） | 50 |
| T（観測時間） | 1時間 |
| サンプリングレート | 500 Hz (dt = 0.002 s) |
| 周波数範囲 | [0.1, 250] Hz |
| 抽出モード | Paper mode（均一omega + Hermitian IDFT） |

### 時間領域検証

<table>
<tr>
<td align="center" width="50%">
<img src="../../../images/fir_validation_multisine.png" alt="FIR検証 - マルチサイン" width="400"><br>
<em>マルチサイン入力での検証（RMSE = 0.0290）</em>
</td>
<td align="center" width="50%">
<img src="../../../images/fir_validation_square_wave.png" alt="FIR検証 - 矩形波" width="400"><br>
<em>矩形波入力での検証（RMSE = 0.0589）</em>
</td>
</tr>
</table>

| 検証信号 | RMSE (x10^-2 rad) | 説明 |
|:---|:---:|:---|
| マルチサイン | 2.90 | 学習時と同種の信号 |
| 矩形波 | 5.89 | 未知の検証信号 |

GPRから再構成されたFIRモデルは、**学習信号**（マルチサイン）と全く**異なる検証信号**（矩形波）の両方を正確に追従し、汎化性能を実証している。

### FIR予測の詳細

<table>
<tr>
<td align="center" width="50%">
<img src="../../../images/gp_fir_wave_output_vs_predicted.png" alt="FIR出力 vs 予測" width="400"><br>
<em>実出力 y(t) vs FIR予測 y_hat(t)</em>
</td>
<td align="center" width="50%">
<img src="../../../images/gp_fir_wave_error.png" alt="FIR予測誤差" width="400"><br>
<em>予測誤差 e(t) = y(t) - y_hat(t)</em>
</td>
</tr>
</table>

1024タップのFIR畳み込み `y_hat(t) = sum(h_k * u(t - k*dt))` により、共振挙動を含むフレキシブルリンクのダイナミクスを捕捉している。
