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

## Example Outputs

### I/O Signal Plots

<table>
<tr>
<td align="center" width="50%">
<img src="../../docs/images/figure_input_data.png" alt="Multisine input/output" width="400"><br>
<em>Multisine input u(t) and output y(t)</em>
</td>
<td align="center" width="50%">
<img src="../../docs/images/figure_wave_data.png" alt="Square wave input/output" width="400"><br>
<em>Square wave input u(t) and output y(t)</em>
</td>
</tr>
</table>

Time-domain signals used for system identification of the Quanser Rotary Flexible Link. The multisine signal is used for training (FRF estimation + GP regression), while the square wave serves as an unseen validation signal.

### Plot Style in Context

All figures in this project (Bode plots, Nyquist plots, GP predictions, FIR validations) are rendered using `configure_plot_style()` with the defaults listed above. This ensures consistent, publication-quality formatting across all outputs.
