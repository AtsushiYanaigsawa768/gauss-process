#!/usr/bin/env python3
"""
comprehensive_test.py

Comprehensive testing harness that sweeps across GP kernels, classical methods,
file counts, time durations, and nd values.  Results are saved incrementally
to CSV files after each test.

Public entry points:
- save_results_to_csv(result_entry, output_base_dir, timestamp)
- run_comprehensive_test(mat_files, ...) -- main test loop
- __main__ block for CLI invocation

All numerical behaviour is identical to the original
``unified_pipeline.py`` test-mode implementation.
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


# =====================
# CSV result writer
# =====================

def save_results_to_csv(
    result_entry: Dict,
    output_base_dir: Path,
    timestamp: str,
) -> None:
    """Save a single test result to multiple CSV files incrementally.

    Three CSVs are maintained:
    1. ``overall_results.csv``          -- all results
    2. ``results_by_method_<m>.csv``    -- per-method results
    3. ``results_by_nd_<n>.csv``        -- per-nd results

    Args:
        result_entry:    Dictionary with the test result.
        output_base_dir: Base output directory.
        timestamp:       Timestamp string for the test run.
    """
    base_path = Path(output_base_dir) / timestamp
    base_path.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'method', 'n_files', 'time_duration', 'nd', 'freq_method',
        'fir_rmse', 'fir_wave_rmse',
    ]

    # 1. Overall results
    overall_csv = base_path / 'overall_results.csv'
    file_exists = overall_csv.exists()
    with open(overall_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_entry)

    # 2. Method-specific
    method_name = result_entry.get('method', '')
    if method_name:
        method_csv = base_path / f'results_by_method_{method_name}.csv'
        file_exists = method_csv.exists()
        with open(method_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_entry)

    # 3. nd-specific
    nd_value = result_entry.get('nd')
    if nd_value is not None:
        nd_csv = base_path / f'results_by_nd_{nd_value}.csv'
        file_exists = nd_csv.exists()
        with open(nd_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_entry)

    print(f"  Results saved to CSVs (overall, method: {method_name}, nd: {nd_value})")


# =====================
# Metric extraction
# =====================

def _extract_metrics(results: Dict, is_gp: bool):
    """Return (rmse_real, rmse_imag, fir_rmse, fir_fit, fir_wave_rmse, fir_wave_fit)."""
    if is_gp:
        rmse_real = results.get('real', {}).get('rmse', None)
        rmse_imag = results.get('imag', {}).get('rmse', None)
    else:
        rmse_real = results.get('rmse_real', None)
        rmse_imag = results.get('rmse_imag', None)

    fir_rmse = fir_fit = fir_wave_rmse = fir_wave_fit = None
    if 'fir_extraction' in results:
        fir_rmse = results['fir_extraction'].get('rmse', None)
        fir_fit = results['fir_extraction'].get('fit_percent', None)
    if 'fir_extraction_wave' in results:
        fir_wave_rmse = results['fir_extraction_wave'].get('rmse', None)
        fir_wave_fit = results['fir_extraction_wave'].get('fit_percent', None)

    return rmse_real, rmse_imag, fir_rmse, fir_fit, fir_wave_rmse, fir_wave_fit


def _make_result_entry(method, n_files, time_duration, nd, freq_method, fir_rmse, fir_wave_rmse):
    return {
        'method': method,
        'n_files': n_files,
        'time_duration': time_duration,
        'nd': nd,
        'freq_method': freq_method,
        'fir_rmse': fir_rmse,
        'fir_wave_rmse': fir_wave_rmse,
    }


# =====================
# Single test runner
# =====================

def _run_single_test(
    *,
    method, is_gp, kernel, training_files, n_files, time_duration,
    nd, freq_method, fir_validation_mat,
    use_grid_search, grid_search_max_combinations,
    output_dir, all_results, output_base_dir, timestamp,
):
    """Execute a single configuration and save results."""
    test_name_parts = [method, f"nd{nd}", f"{n_files}file"]
    if n_files == 1:
        test_name_parts.append(f"{time_duration}s" if time_duration else "full")
    else:
        test_name_parts[-1] += "s"

    print(f"\nTest: {'_'.join(test_name_parts)}")
    print(f"  Method: {method}")
    print(f"  nd: {nd}")

    try:
        config = argparse.Namespace(
            mat_files=training_files[:n_files] if n_files else training_files,
            use_existing=None,
            n_files=n_files if n_files else len(training_files),
            time_duration=time_duration,
            kernel=kernel if is_gp else 'rbf',
            nu=2.5 if kernel == 'matern' else None,
            gp_mode='separate',
            noise_variance=0.0,
            normalize=True,
            log_frequency=True,
            optimize=use_grid_search,
            n_restarts=3,
            out_dir=str(output_dir),
            extract_fir=True,
            fir_length=1024,
            fir_validation_mat=fir_validation_mat,
            method=method,
            is_gp=is_gp,
            nd=nd,
            freq_method=freq_method,
            use_grid_search=use_grid_search,
            grid_search_max_combinations=grid_search_max_combinations,
            validation_mat=fir_validation_mat,
        )

        results = run_gp_pipeline(config)

        if results:
            (rmse_real, rmse_imag,
             fir_rmse, fir_fit,
             fir_wave_rmse, fir_wave_fit) = _extract_metrics(results, is_gp)

            entry = _make_result_entry(
                method, n_files, time_duration, nd, freq_method, fir_rmse, fir_wave_rmse,
            )
            all_results.append(entry)
            save_results_to_csv(entry, Path(output_base_dir), timestamp)

            print(f"  Success - GP RMSE Real: {rmse_real:.3e}, Imag: {rmse_imag:.3e}")
            if fir_rmse:
                print(f"            FIR RMSE: {fir_rmse:.3e}, FIT: {fir_fit:.1f}%")
            if fir_wave_rmse:
                print(f"            FIR Wave RMSE: {fir_wave_rmse:.3e}, FIT: {fir_wave_fit:.1f}%")
        else:
            print(f"  No results returned")
            entry = _make_result_entry(
                method, n_files, time_duration, nd, freq_method, None, None,
            )
            all_results.append(entry)
            save_results_to_csv(entry, Path(output_base_dir), timestamp)

    except Exception as e:
        print(f"  Error: {str(e)}")
        entry = _make_result_entry(
            method, n_files, time_duration, nd, freq_method, None, None,
        )
        all_results.append(entry)
        save_results_to_csv(entry, Path(output_base_dir), timestamp)

    plt.close('all')
    gc.collect()


# =====================
# Main test driver
# =====================

def run_comprehensive_test(
    mat_files: List[str],
    output_base_dir: str = 'test_output',
    fir_validation_mat: Optional[str] = None,
    nd_values: Optional[List[int]] = None,
    freq_method: str = 'frf',
    use_grid_search: bool = False,
    grid_search_max_combinations: int = 5000,
) -> Path:
    """Run comprehensive tests across kernels, time intervals, file counts, and nd values.

    Results are saved incrementally after each test.

    Args:
        mat_files:                   List of MAT files.
        output_base_dir:             Base directory for output.
        fir_validation_mat:          MAT file for FIR validation.
        nd_values:                   List of nd values (default: [10, 30, 50, 100]).
        freq_method:                 'frf' or 'fourier'.
        use_grid_search:             Enable grid search.
        grid_search_max_combinations: Max grid combinations.

    Returns:
        Path to the overall_results.csv file.
    """
    # Test configurations
    kernels = [
        'rbf', 'matern', 'matern12', 'matern32', 'matern52', 'rq',
        'exp', 'tc', 'dc', 'di', 'ss1', 'ss2', 'sshf', 'stable_spline',
    ]
    classical_methods = ['nls', 'ls']
    all_methods = ['gp_' + k for k in kernels] + classical_methods

    if nd_values is None:
        nd_values = [10, 30, 50, 100]

    time_durations = [600.0, 1800.0, None]
    n_files_list = [1, 10]

    mat_files = sorted(mat_files)

    # Exclude validation file from training
    training_files = mat_files.copy()
    if fir_validation_mat is not None:
        validation_mat_path = Path(fir_validation_mat).resolve()
        training_files = [
            f for f in training_files if Path(f).resolve() != validation_mat_path
        ]
        print("=" * 80)
        print("TRAIN/TEST DATA SEPARATION")
        print("=" * 80)
        print(f"Total MAT files available: {len(mat_files)}")
        print(f"Test file (EXCLUDED from training): {fir_validation_mat}")
        print(f"Training files available: {len(training_files)}")
        print("=" * 80)
        print()

        if len(training_files) == 0:
            print("ERROR: No training files remaining after excluding test file!")
            print("Need at least 2 MAT files for proper train/test split.")
            return Path(output_base_dir)

    all_results: List[Dict] = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    total_tests = 0

    print("=" * 80)
    print("Starting Comprehensive System Identification Testing Suite")
    print(f"Timestamp: {timestamp}")
    print(f"Training files (sorted):")
    for i, f in enumerate(training_files, 1):
        print(f"  [{i}] {f}")
    print(f"GP Kernels: {', '.join(kernels)}")
    print(f"Classical methods: {', '.join(classical_methods)}")
    print(f"Time durations: {time_durations}")
    print(f"File counts: {n_files_list}")
    print(f"Frequency points (nd): {nd_values}")
    print("=" * 80)

    for method in all_methods:
        print(f"\n{'='*60}")
        print(f"Testing method: {method}")
        print(f"{'='*60}")

        is_gp = method.startswith('gp_')
        kernel = method[3:] if is_gp else None

        for nd in nd_values:
            for n_files in n_files_list:
                if n_files is not None and n_files > len(training_files):
                    continue

                actual_n_files = n_files if n_files is not None else len(training_files)

                if n_files == 1:
                    for time_duration in time_durations:
                        test_name = (
                            f"{method}_nd{nd}_1file_"
                            f"{'full' if time_duration is None else f'{time_duration}s'}"
                        )
                        output_dir = Path(output_base_dir) / timestamp / test_name

                        _run_single_test(
                            method=method, is_gp=is_gp, kernel=kernel,
                            training_files=training_files, n_files=1,
                            time_duration=time_duration, nd=nd,
                            freq_method=freq_method,
                            fir_validation_mat=fir_validation_mat,
                            use_grid_search=use_grid_search,
                            grid_search_max_combinations=grid_search_max_combinations,
                            output_dir=output_dir,
                            all_results=all_results,
                            output_base_dir=output_base_dir,
                            timestamp=timestamp,
                        )
                        total_tests += 1
                else:
                    test_name = f"{method}_nd{nd}_{actual_n_files}files"
                    output_dir = Path(output_base_dir) / timestamp / test_name

                    _run_single_test(
                        method=method, is_gp=is_gp, kernel=kernel,
                        training_files=training_files, n_files=actual_n_files,
                        time_duration=None, nd=nd,
                        freq_method=freq_method,
                        fir_validation_mat=fir_validation_mat,
                        use_grid_search=use_grid_search,
                        grid_search_max_combinations=grid_search_max_combinations,
                        output_dir=output_dir,
                        all_results=all_results,
                        output_base_dir=output_base_dir,
                        timestamp=timestamp,
                    )
                    total_tests += 1

    csv_file = Path(output_base_dir) / timestamp / 'overall_results.csv'

    print("\n" + "=" * 80)
    print(f"Comprehensive testing complete!")
    print(f"Total tests run: {total_tests}")
    print(f"Successful: {sum(1 for r in all_results if r.get('fir_rmse') is not None)}")
    print(f"Failed: {sum(1 for r in all_results if r.get('fir_rmse') is None)}")
    print(f"\nResults saved incrementally to:")
    print(f"  - Overall: {csv_file}")
    print(f"  - By method: {csv_file.parent}/results_by_method_*.csv")
    print(f"  - By nd: {csv_file.parent}/results_by_nd_*.csv")
    print("=" * 80)

    # Summary report
    summary_file = Path(output_base_dir) / timestamp / 'summary_report.txt'
    _write_summary(summary_file, timestamp, total_tests, all_results)
    print(f"Summary report saved to: {summary_file}")

    return csv_file


def _write_summary(summary_file: Path, timestamp: str, total_tests: int, all_results: List[Dict]):
    """Write a human-readable summary report."""
    with open(summary_file, 'w') as f:
        f.write("GP and FIR Model Testing Summary Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Date: {timestamp}\n")
        f.write(f"Total Tests: {total_tests}\n")
        n_success = sum(1 for r in all_results if r.get('fir_rmse') is not None)
        n_fail = sum(1 for r in all_results if r.get('fir_rmse') is None)
        f.write(f"Successful Tests: {n_success}\n")
        f.write(f"Failed Tests: {n_fail}\n\n")

        best_results = []
        for result in all_results:
            if result.get('fir_rmse') is not None:
                method = result.get('method', 'unknown')
                nd = result.get('nd', 0)
                n_files = result.get('n_files', 0)
                time_dur = result.get('time_duration', 'full')
                config_name = f"{method}_nd{nd}_{n_files}files_{time_dur}"
                best_results.append((method, result['fir_rmse'], config_name))

        if best_results:
            best_results.sort(key=lambda x: x[1])
            f.write("Top 5 Best Performing Configurations (by FIR RMSE):\n")
            f.write("-" * 60 + "\n")
            for i, (kernel, rmse, test_name) in enumerate(best_results[:5]):
                f.write(f"{i+1}. {test_name}: RMSE = {rmse:.3e}\n")
            f.write("\n")

        method_stats: Dict[str, List[float]] = {}
        for result in all_results:
            if result.get('status') == 'success':
                method = result.get('kernel', result.get('method', 'unknown'))
                metric = result.get('fir_rmse')
                if metric is None and result.get('gp_rmse_real') is not None:
                    metric = (result['gp_rmse_real'] + result['gp_rmse_imag']) / 2
                if metric is not None:
                    method_stats.setdefault(method, []).append(metric)

        if method_stats:
            f.write("Method Performance Summary (Average RMSE):\n")
            f.write("-" * 60 + "\n")
            method_avg = []
            for method, rmse_list in method_stats.items():
                avg_rmse = np.mean(rmse_list)
                std_rmse = np.std(rmse_list) if len(rmse_list) > 1 else 0
                method_avg.append((method, avg_rmse, std_rmse, len(rmse_list)))
            method_avg.sort(key=lambda x: x[1])
            for method, avg, std, count in method_avg:
                f.write(f"{method}: {avg:.3e} +/- {std:.3e} (n={count})\n")


# =====================
# CLI entry point
# =====================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test-mode':
        print("Running in comprehensive test mode...")

        mat_pattern = 'input/*.mat'
        mat_files = sorted(glob.glob(mat_pattern))
        if not mat_files:
            print(f"Error: No MAT files found matching pattern: {mat_pattern}")
            sys.exit(1)

        print(f"Found {len(mat_files)} MAT files (sorted):")
        for i, f in enumerate(mat_files, 1):
            print(f"  [{i}] {f}")
        print()

        validation_mat = None
        if len(mat_files) >= 2:
            validation_mat = mat_files[-1]
            print(f"Using LAST file for validation (test data): {validation_mat}")
            print(f"Training files will use first {len(mat_files)-1} files")
        else:
            print("ERROR: Need at least 2 MAT files for train/test split!")
            print(f"   Found only {len(mat_files)} file(s)")
            sys.exit(1)

        print(f"NOTE: This validation file will be EXCLUDED from training data")
        print(f"NOTE: This ensures proper train/test separation")
        print()

        # Phase 1: FRF
        print("=" * 80)
        print("PHASE 1: Testing with FRF (Log-scale frequency analysis)")
        print("=" * 80)
        print()

        run_comprehensive_test(
            mat_files,
            output_base_dir='test_output_frf',
            fir_validation_mat=validation_mat,
            freq_method='frf',
            use_grid_search=True,
            grid_search_max_combinations=500000,
        )

        print("\n" + "=" * 80)
        print("PHASE 1 COMPLETED: FRF tests finished")
        print("=" * 80)
        print()

    else:
        # Normal pipeline mode -- delegate to unified_pipeline main()
        from src.pipeline.unified_pipeline import main
        main()
