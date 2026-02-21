# visualization/ -- プロット機能

[English version](../../../../src/visualization/README.md)

## 概要

`src/visualization/` は論文・発表用の一貫したプロットスタイルと、
時間領域信号の可視化機能を提供する。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `plot_styles.py` | matplotlib rcParams の一括設定 (論文品質) |
| `plot_io.py` | .matファイルの入出力時系列プロット |

## plot_styles.py

`configure_plot_style()` を呼び出すと、以下のパラメータが設定される。

| パラメータ | 値 |
|-----------|-----|
| 基本フォント | 22 pt |
| 軸ラベル | 30 pt |
| タイトル | 32 pt |
| 凡例 | 24 pt |
| 線幅 | 3.5 |
| マーカーサイズ | 12 |

```python
from src.visualization.plot_styles import configure_plot_style
configure_plot_style()
```

## plot_io.py

`.mat` ファイルから時間・入力・出力信号を読み込み、プロットする。

主な関数:

- `parse_time_window(time_window)` -- `"5s"`, `"1min"`, `"30min"` 等の文字列を秒に変換
- 時系列データの読み込みには `frequency_transform.frf_estimator.load_time_u_y()` を内部で使用

### 使用例

```python
from src.visualization.plot_io import parse_time_window

seconds = parse_time_window("30min")  # -> 1800.0
```

## 出力形式

全プロットは matplotlib の `Agg` バックエンドを使用し、PNGファイルとして保存される。
対話的表示は行わない (サーバー環境対応)。

## 出力例

### 入出力信号プロット

<table>
<tr>
<td align="center" width="50%">
<img src="../../../images/figure_input_data.png" alt="マルチサイン入出力" width="400"><br>
<em>マルチサイン入力 u(t) と出力 y(t)</em>
</td>
<td align="center" width="50%">
<img src="../../../images/figure_wave_data.png" alt="矩形波入出力" width="400"><br>
<em>矩形波入力 u(t) と出力 y(t)</em>
</td>
</tr>
</table>

Quanser Rotary Flexible Linkのシステム同定に使用された時間領域信号。マルチサイン信号は学習用（FRF推定 + GP回帰）、矩形波は未知の検証信号として使用される。

### プロットスタイルの適用例

本プロジェクトの全図表（ボード線図、ナイキスト線図、GP予測、FIR検証）は `configure_plot_style()` により上記のデフォルト設定で描画される。これにより、全出力で一貫した論文品質のフォーマットが保証される。
