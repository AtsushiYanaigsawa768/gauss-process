# visualization/ -- Plot Helpers

[日本語版はこちら](../../docs/ja/src/visualization/README.md)

## Purpose

Provides consistent, publication-quality plotting utilities shared across the
system identification pipeline.

## Files

| File | Description |
|---|---|
| `plot_styles.py` | `configure_plot_style()` -- sets matplotlib rcParams for large, paper-ready figures (22-34 pt fonts, thick lines, large ticks) |
| `plot_io.py` | `plot_io_signals()` -- plots input u(t) and output y(t) from a `.mat` file with configurable time windows |

## Usage

### Apply consistent plot style

```python
from src.visualization.plot_styles import configure_plot_style
configure_plot_style()  # Call once before any plotting
```

### Plot I/O time-domain signals

```python
from src.visualization.plot_io import plot_io_signals
path = plot_io_signals("data/sample_data/input_test_20250913_010037.mat",
                       time_window="5s", output_path="io_plot.png")
```

### CLI usage

```bash
python -m src.visualization.plot_io data/sample_data/input_test_20250913_010037.mat \
    --time-window 30s --output io_30s.png
```

## Time Window Format

The `time_window` parameter accepts duration strings: `"5s"`, `"1min"`, `"30min"`, `"1h"`.

## Plot Style Defaults

| Parameter | Value |
|---|---|
| Base font | 22 pt |
| Axis labels | 30 pt |
| Title | 32 pt |
| Legend | 24 pt |
| Line width | 3.5 |
| Marker size | 12 |
