# sample_data/ -- Time-Domain Recordings

[日本語版はこちら](../../docs/ja/data/sample_data/README.md)

## Overview

Contains 10 MATLAB `.mat` files, each a 1-hour recording of the flexible link
mechanism experiment conducted on 2025-09-13 at 2-hour intervals.

## File List

| File | Recording Start |
|---|---|
| `input_test_20250913_010037.mat` | 01:00:37 |
| `input_test_20250913_030050.mat` | 03:00:50 |
| `input_test_20250913_050103.mat` | 05:01:03 |
| `input_test_20250913_070119.mat` | 07:01:19 |
| `input_test_20250913_090135.mat` | 09:01:35 |
| `input_test_20250913_110148.mat` | 11:01:48 |
| `input_test_20250913_130201.mat` | 13:02:01 |
| `input_test_20250913_150214.mat` | 15:02:14 |
| `input_test_20250913_170227.mat` | 17:02:27 |
| `input_test_20250913_190241.mat` | 19:02:41 |

## Data Format

Each `.mat` file contains a numeric array (typically named `output`) with
shape 3xN or Nx3, where the rows/columns are ordered as:

| Index | Signal | Description |
|---|---|---|
| 0 | `t` | Time vector [s] |
| 1 | `y` | Output signal (link tip displacement) |
| 2 | `u` | Input signal (motor command) |

Alternatively, some files may store separate variables `t`, `u`, `y`.

## Loading

```python
from src.utils.data_io import load_mat_time_series
t, u, y = load_mat_time_series("data/sample_data/input_test_20250913_010037.mat")
```

## Usage

- **FRF estimation**: Pass to `frequency_transform/` for synchronous demodulation or FFT
- **FIR validation**: Use as time-domain validation data for FIR models
- **Visualization**: Plot with `src.visualization.plot_io.plot_io_signals()`
