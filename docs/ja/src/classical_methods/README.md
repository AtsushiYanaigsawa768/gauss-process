# classical_methods/ -- 古典的システム同定手法

[English version](../../../../src/README_PIPELINE.md)

## 概要

`src/classical_methods/` は周波数領域のパラメトリック/ノンパラメトリック手法と、
機械学習ベースの推定手法を提供する。GPR との比較ベースラインとして使用する。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `frequency_domain.py` | 古典的周波数領域推定 (6手法) |
| `ml_methods.py` | ML手法 (3手法) + ローカル多項式/有理法 (2手法) + ファクトリ |

## 周波数領域推定手法

`frequency_domain.py` は有理多項式モデル `H(jw) = N(jw) / D(jw)` を推定する。
全手法が `FrequencyDomainEstimator` 基底クラスを継承する。

| 手法 | クラス名 | 説明 |
|------|---------|------|
| NLS | `NLSEstimator` | 非線形最小二乗法 |
| LS | `LSEstimator` | 線形最小二乗法 (Levy法) |
| IWLS | `IWLSEstimator` | 反復重み付き最小二乗法 (Sanathanan-Koerner) |
| TLS | `TLSEstimator` | 全最小二乗法 (ユークリッドノルム制約) |
| ML | `MLEstimator` | 最尤推定法 (複素ガウスノイズ) |
| LOG | `LOGEstimator` | 対数最小二乗法 |

### 使用例

```python
from src.classical_methods.frequency_domain import NLSEstimator

estimator = NLSEstimator(n_numerator=2, n_denominator=2)
estimator.fit(omega, H_measured)
H_pred = estimator.predict(omega_test)
```

## 機械学習手法

`ml_methods.py` は以下の手法を提供する。

| 手法 | クラス名 | 説明 |
|------|---------|------|
| LPM | `LocalPolynomialMethod` | ローカル多項式法 (ノンパラメトリック FRF 推定) |
| LRMP | `LRMPEstimator` | 事前極付きローカル有理法 |
| RF | `RFEstimator` | Random Forest 回帰 |
| GBR | `GBREstimator` | Gradient Boosting 回帰 |
| SVM | `SVMEstimator` | Support Vector Machine 回帰 |

## 統合ファクトリ

`create_estimator()` は全手法 (古典的 + ML) を統一的に生成する。

```python
from src.classical_methods.ml_methods import create_estimator

# 古典的手法
estimator = create_estimator('nls', n_numerator=3, n_denominator=3)

# ML手法
estimator = create_estimator('rf', n_estimators=100)
```

## 手法選択の指針

| 要件 | 推奨手法 |
|------|---------|
| 物理モデルの次数が既知 | NLS, ML |
| 初期推定が必要 | LS → IWLS (2段階) |
| ノイズが大きい | TLS, ML |
| モデル次数が不明 | LPM (ノンパラメトリック) |
| 大量データでの汎化 | RF, GBR |
