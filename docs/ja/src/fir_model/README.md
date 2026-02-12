# fir_model/ -- FIRモデル同定

[English version](../../../../src/README_PIPELINE.md)

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
