# data/ -- Dataset Directory

[日本語版はこちら](../docs/ja/data/README.md)

## Overview

Contains experimental data for system identification of a flexible link
mechanism. Data is organized into two subdirectories.

## Directory Structure

```
data/
└── sample_data/    10 .mat files -- time-domain recordings (input/output)
```

## Subdirectories

### sample_data/

Ten 1-hour recordings of the flexible link mechanism sampled at high rate.
Each `.mat` file contains time, input, and output vectors.
Used for: FRF estimation, FIR model validation.

See [sample_data/README.md](sample_data/README.md) for format details.