#!/usr/bin/env python3
"""
Run all 13 methods (11 GP kernels + 2 classical) with standard settings
and output the FIR RMSE results table.

Settings match run_comprehensive_test with nd=50, time_duration=None (full data).

Usage:
    python -m src.run_baseline
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.pipeline.gp_pipeline import run_gp_pipeline

# ── Method definitions ──

GP_METHODS = {
    'dc', 'di', 'exp', 'matern12', 'matern32', 'matern52',
    'rbf', 'ss1', 'ss2', 'sshf', 'stable_spline',
}

METHOD_ORDER: List[str] = [
    'dc', 'di', 'exp', 'matern12', 'matern32', 'matern52',
    'rbf', 'ss1', 'ss2', 'sshf', 'stable_spline',
    'ls', 'nls',
]

DISPLAY_NAMES: Dict[str, str] = {
    'dc': 'DC', 'di': 'DI', 'exp': 'Exponential',
    'matern12': 'Matern-1/2', 'matern32': 'Matern-3/2', 'matern52': 'Matern-5/2',
    'rbf': 'RBF', 'ss1': 'SS1', 'ss2': 'SS2', 'sshf': 'SSHF',
    'stable_spline': 'Stable Spline', 'ls': 'LS', 'nls': 'NLS',
}


def build_config(
    method: str,
    training_files: List[str],
    validation_mat: str,
    output_dir: str,
    use_existing: Optional[str] = None,
) -> argparse.Namespace:
    """Build a pipeline config matching run_comprehensive_test (nd=50, full data)."""
    is_gp = method in GP_METHODS
    return argparse.Namespace(
        mat_files=training_files[:1],
        use_existing=use_existing,
        n_files=1,
        time_duration=None,
        nd=50,
        freq_method='frf',
        kernel=method if is_gp else 'rbf',
        nu=None,
        gp_mode='separate',
        noise_variance=0.0,
        normalize=True,
        log_frequency=True,
        optimize=True,
        n_restarts=3,
        out_dir=output_dir,
        extract_fir=True,
        fir_length=1024,
        fir_validation_mat=validation_mat,
        method='gp' if is_gp else method,
        is_gp=is_gp,
        use_grid_search=True,
        grid_search_max_combinations=500000,
        validation_mat=validation_mat,
        n_numerator=2,
        n_denominator=4,
    )


def find_frf_csv(output_dir: str) -> Optional[str]:
    frf_path = Path(output_dir) / 'unified_frf.csv'
    if frf_path.exists():
        return str(frf_path)
    candidates = list(Path(output_dir).glob('**/unified_frf.csv'))
    return str(candidates[0]) if candidates else None


def main():
    mat_files = sorted(glob.glob('input/*.mat'))
    if len(mat_files) < 2:
        print(f"ERROR: Need at least 2 MAT files. Found {len(mat_files)}")
        sys.exit(1)

    training_files = [mat_files[0]]
    validation_mat = mat_files[-1]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path('baseline_output') / timestamp
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BASELINE (N_d=50, T=full, max_comb=500000)")
    print("=" * 70)
    print(f"Training:   {training_files[0]}")
    print(f"Validation: {validation_mat}")
    print(f"Output:     {output_base}")
    print("=" * 70)

    results: List[Dict] = []
    frf_csv_path: Optional[str] = None

    for i, method in enumerate(METHOD_ORDER):
        display = DISPLAY_NAMES[method]
        method_output = str(output_base / method)

        print(f"\n[{i+1}/{len(METHOD_ORDER)}] {display} ({method})")
        print("-" * 50)

        try:
            np.random.seed(42)
            config = build_config(
                method, training_files, validation_mat,
                method_output, use_existing=frf_csv_path,
            )
            pipeline_results = run_gp_pipeline(config)

            if frf_csv_path is None and pipeline_results:
                found = find_frf_csv(method_output)
                if found:
                    frf_csv_path = found

            fir_rmse = None
            if pipeline_results:
                fir_rmse = (pipeline_results.get('fir_extraction', {})
                            .get('rmse', None))

            if fir_rmse is not None:
                print(f"  FIR RMSE: {fir_rmse:.4e}")
            else:
                print(f"  FIR RMSE: N/A")

            results.append({
                'method': method,
                'display_name': display,
                'fir_rmse': fir_rmse,
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback; traceback.print_exc()
            results.append({
                'method': method,
                'display_name': display,
                'fir_rmse': None,
            })

        plt.close('all')
        gc.collect()

    # ── Summary table ──
    print()
    print("=" * 50)
    print("RESULTS (N_d=50, T=full, max_comb=500000)")
    print("=" * 50)
    print(f"| {'Method':<16}| {'RMSE (x1e-2)':>13} |")
    print("|:" + "-" * 15 + "|" + "-" * 14 + ":|")
    for row in results:
        rmse_str = f"{row['fir_rmse'] * 100:.2f}" if row['fir_rmse'] else "N/A"
        print(f"| {row['display_name']:<16}| {rmse_str:>13} |")
    print("=" * 50)

    # ── Save CSV ──
    csv_path = output_base / 'baseline_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'display_name', 'fir_rmse'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {csv_path}")


if __name__ == '__main__':
    main()
