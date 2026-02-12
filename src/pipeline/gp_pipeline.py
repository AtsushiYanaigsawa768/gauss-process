#!/usr/bin/env python3
"""
gp_pipeline.py -- Core pipeline orchestrator for GP-based (and classical/ML)
system identification from frequency-response data.

Public API:
    run_gp_pipeline(config)
    run_unified_system_identification(omega, G_complex, ...)
"""
from __future__ import annotations
import argparse, gc
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.gpr.kernels import create_kernel
from src.gpr.gpr_fitting import GaussianProcessRegressor
from src.gpr.visualization import plot_gp_results, plot_complex_gp
from src.pipeline.data_loader import (
    load_frf_data, run_frequency_response, run_fourier_transform,
    generate_validation_data_from_mat,
)

# ── Step 1: Load frequency data ─────────────────────────────────────

def _load_frequency_data(config, output_dir: Path):
    """Return (omega, G_complex, freq_csv) with train/test separation."""
    if config.use_existing:
        print(f"Using existing frequency data from: {config.use_existing}")
        freq_csv = Path(config.use_existing)
        if config.fir_validation_mat is not None:
            print(f"\n{'='*70}\nWARNING: Using existing CSV - cannot verify "
                  f"training/test data separation\n{'='*70}\n")
    else:
        training_mat_files = (
            config.mat_files.copy() if isinstance(config.mat_files, list)
            else config.mat_files)
        if config.fir_validation_mat is not None:
            vp = Path(config.fir_validation_mat).resolve()
            if isinstance(training_mat_files, list):
                orig = len(training_mat_files)
                training_mat_files = [f for f in training_mat_files
                                      if Path(f).resolve() != vp]
                excl = orig - len(training_mat_files)
                if excl > 0:
                    print(f"\nDATA SEPARATION: Excluded {excl} file(s); "
                          f"{len(training_mat_files)} remaining\n")
                if not training_mat_files:
                    raise ValueError(
                        "Training and test data cannot use the same file! "
                        "Provide at least 2 distinct MAT files.")
        if hasattr(config, 'freq_method') and config.freq_method == 'fourier':
            freq_csv = run_fourier_transform(
                training_mat_files, output_dir,
                config.n_files, config.time_duration, config.nd)
        else:
            freq_csv = run_frequency_response(
                training_mat_files, output_dir,
                config.n_files, config.time_duration, config.nd)

    print("Loading frequency domain data...")
    frf_df = load_frf_data(freq_csv)
    omega = frf_df['omega_rad_s'].values
    G_complex = frf_df['ReG'].values + 1j * frf_df['ImG'].values
    return omega, G_complex, freq_csv

# ── Step 2a: Classical / ML branch ──────────────────────────────────

def _run_classical_method(omega, G_complex, config, output_dir):
    print(f"\nUsing {config.method} method...")
    results, G_pred, estimator = run_unified_system_identification(
        omega, G_complex, method=config.method, config=config,
        output_dir=output_dir / config.method, return_predictor=True)
    results['config'] = {'method': config.method, 'normalize': config.normalize,
                         'n_files': config.n_files, 'time_duration': config.time_duration}
    print(f"{config.method.upper()} regression complete.")
    if config.extract_fir:
        _extract_fir_with_predictor(
            results, omega, G_pred, lambda w: estimator.predict(w),
            config, output_dir, label=config.method.upper())
    return results

# ── Step 3: GP fitting ──────────────────────────────────────────────

def _prepare_validation(config, X_scaler):
    """Load grid-search validation data if applicable. Returns (vX_re, vy_re, vX_im, vy_im)."""
    use_gs = getattr(config, 'use_grid_search', False)
    if not (use_gs and getattr(config, 'fir_validation_mat', None)):
        return None, None, None, None
    print(f"\n{'='*70}\nGRID SEARCH VALIDATION DATA (TEST DATA)\n{'='*70}")
    print(f"  File: {config.fir_validation_mat}  (EXCLUDED from training)\n{'='*70}")
    try:
        omega_v, Gr, Gi = generate_validation_data_from_mat(
            config.fir_validation_mat, nd=150,
            freq_method=getattr(config, 'freq_method', 'frf'))
        Xv = np.log10(omega_v).reshape(-1, 1) if config.log_frequency else omega_v.reshape(-1, 1)
        if config.normalize:
            Xv = X_scaler.transform(Xv)
        print(f"  Validation data prepared: {len(Gr)} points")
        return Xv, Gr, Xv, Gi
    except Exception as e:
        print(f"  Warning: Could not load validation data: {e}")
        return None, None, None, None


def _fit_one_gp(config, kernel_params, X_norm, y, y_scaler, val_X, val_y):
    """Create, fit, and predict with a single GP."""
    gp = GaussianProcessRegressor(
        kernel=create_kernel(config.kernel, **kernel_params),
        noise_variance=config.noise_variance)
    gs = getattr(config, 'use_grid_search', False)
    mx = getattr(config, 'grid_search_max_combinations', 5000)
    gp.fit(X_norm, y, optimize=config.optimize, n_restarts=config.n_restarts,
           use_grid_search=gs, max_grid_combinations=mx,
           validation_X=val_X, validation_y=val_y, y_scaler=y_scaler)
    pred, std = gp.predict(X_norm, return_std=True)
    return gp, pred, std


def _fit_gp_separate(omega, G_complex, X_gp, config, output_dir, kp):
    """Separate GPs for Real and Imaginary parts."""
    print("Fitting separate GPs for real and imaginary parts...")
    results: Dict = {}
    if config.normalize:
        Xs = StandardScaler(); Xn = Xs.fit_transform(X_gp)
        yrs = StandardScaler(); yr = yrs.fit_transform(np.real(G_complex).reshape(-1,1)).ravel()
        yis = StandardScaler(); yi = yis.fit_transform(np.imag(G_complex).reshape(-1,1)).ravel()
    else:
        Xs = None; Xn = X_gp
        yr = np.real(G_complex); yi = np.imag(G_complex)
        yrs = yis = None
    vXr, vyr, vXi, vyi = _prepare_validation(config, Xs)
    print("\nFitting GP for Real part...")
    gp_r, pr, sr = _fit_one_gp(config, kp, Xn, yr, yrs if config.normalize else None, vXr, vyr)
    print("\nFitting GP for Imaginary part...")
    gp_i, pi_, si = _fit_one_gp(config, kp, Xn, yi, yis if config.normalize else None, vXi, vyi)
    if config.normalize:
        pr = yrs.inverse_transform(pr.reshape(-1,1)).ravel(); sr = sr * yrs.scale_
        yo_r = yrs.inverse_transform(yr.reshape(-1,1)).ravel()
        pi_ = yis.inverse_transform(pi_.reshape(-1,1)).ravel(); si = si * yis.scale_
        yo_i = yis.inverse_transform(yi.reshape(-1,1)).ravel()
    else:
        yo_r = yr; yo_i = yi
    results['real'] = plot_gp_results(omega, yo_r, pr, sr, 'Real{G(jw)}', output_dir/'gp_real.png')
    results['imag'] = plot_gp_results(omega, yo_i, pi_, si, 'Imag{G(jw)}', output_dir/'gp_imag.png')
    Gp = pr + 1j * pi_
    plot_complex_gp(omega, G_complex, Gp, sr, si, output_dir/'gp_complex')
    results['kernel_params'] = {
        'real': gp_r.kernel.get_params().tolist(), 'imag': gp_i.kernel.get_params().tolist(),
        'noise_real': float(gp_r.noise_variance), 'noise_imag': float(gp_i.noise_variance)}
    return results, Gp, gp_r, gp_i, Xs, yrs, yis


def _fit_gp_polar(omega, G_complex, X_gp, config, output_dir, kp):
    """GPs for log-magnitude and unwrapped phase."""
    print("Fitting GPs for magnitude and phase...")
    results: Dict = {}
    mag = np.abs(G_complex); phase = np.unwrap(np.angle(G_complex)); lm = np.log(mag)
    if config.normalize:
        Xs = StandardScaler(); Xn = Xs.fit_transform(X_gp)
        ms = StandardScaler(); ym = ms.fit_transform(lm.reshape(-1,1)).ravel()
        ps = StandardScaler(); yp = ps.fit_transform(phase.reshape(-1,1)).ravel()
    else:
        Xs = None; Xn = X_gp; ym = lm; yp = phase; ms = ps = None
    gs = getattr(config, 'use_grid_search', False)
    mx = getattr(config, 'grid_search_max_combinations', 5000)
    gp_m = GaussianProcessRegressor(kernel=create_kernel(config.kernel, **kp),
                                     noise_variance=config.noise_variance)
    gp_m.fit(Xn, ym, optimize=config.optimize, n_restarts=config.n_restarts,
             use_grid_search=gs, max_grid_combinations=mx)
    pm, sm = gp_m.predict(Xn, return_std=True)
    gp_p = GaussianProcessRegressor(kernel=create_kernel(config.kernel, **kp),
                                     noise_variance=config.noise_variance)
    gp_p.fit(Xn, yp, optimize=config.optimize, n_restarts=config.n_restarts,
             use_grid_search=gs, max_grid_combinations=mx)
    pp, sp = gp_p.predict(Xn, return_std=True)
    if config.normalize:
        pm = ms.inverse_transform(pm.reshape(-1,1)).ravel(); sm = sm * ms.scale_
        pp = ps.inverse_transform(pp.reshape(-1,1)).ravel(); sp = sp * ps.scale_
    mag_pred = np.exp(pm)
    results['magnitude'] = plot_gp_results(omega, mag, mag_pred, None, '|G(jw)|', output_dir/'gp_magnitude.png')
    results['phase'] = plot_gp_results(omega, phase, pp, sp, 'Phase [rad]', output_dir/'gp_phase.png')
    Gp = mag_pred * np.exp(1j * pp)
    plot_complex_gp(omega, G_complex, Gp, None, None, output_dir/'gp_complex')
    results['kernel_params'] = {
        'magnitude': gp_m.kernel.get_params().tolist(), 'phase': gp_p.kernel.get_params().tolist(),
        'noise_mag': float(gp_m.noise_variance), 'noise_phase': float(gp_p.noise_variance)}
    return results, Gp, gp_m, gp_p, Xs, ms, ps

# ── Step 4: FIR extraction ──────────────────────────────────────────

def _resolve_validation_mat(config) -> Optional[Path]:
    if not config.fir_validation_mat:
        return None
    p = Path(config.fir_validation_mat)
    if not p.exists():
        print(f"Warning: Validation MAT file not found: {p}"); return None
    print(f"\nFIR MODEL VALIDATION DATA\n  File: {p}\n")
    return p


def _build_gp_predict_closure(config, omega, gp_a, gp_b, Xs, sa, sb):
    """Build a closure that predicts G(jw) at arbitrary omega using fitted GPs."""
    def gp_predict_at_omega(omega_new):
        X = omega_new.reshape(-1, 1).copy()
        if config.log_frequency:
            omin = np.min(omega[omega > 0]) if np.any(omega > 0) else 1e-3
            mask = X.ravel() <= omin * 0.1
            if np.any(mask): X[mask] = omin
            Xg = np.log10(X)
        else:
            Xg = X
        Xn = Xs.transform(Xg) if config.normalize else Xg
        a = gp_a.predict(Xn); b = gp_b.predict(Xn)
        if config.normalize:
            a = sa.inverse_transform(a.reshape(-1,1)).ravel()
            b = sb.inverse_transform(b.reshape(-1,1)).ravel()
        if config.gp_mode == 'separate':
            return a + 1j * b
        return np.exp(a) * np.exp(1j * b)
    return gp_predict_at_omega


def _evaluate_wave_mat(fir_results, results, config, output_dir):
    """Evaluate FIR coefficients on Wave.mat using fir_validation utilities."""
    wave = Path('Wave.mat')
    vmat = Path(config.fir_validation_mat) if config.fir_validation_mat else None
    if not wave.exists() or vmat == wave:
        return
    if 'fir_coefficients' not in fir_results:
        return
    print(f"\nAdditional FIR Evaluation with Wave.mat")
    try:
        from src.fir_model.fir_validation import validate_fir_with_mat
        prefix = f"{config.method}_fir_wave" if hasattr(config, 'method') else "gp_fir_wave"
        wave_res = validate_fir_with_mat(
            fir_results['fir_coefficients'], wave,
            output_dir=output_dir, prefix=prefix, detrend=False)
        wave_res['validation_file'] = 'Wave.mat'
        results['fir_extraction_wave'] = wave_res
    except Exception as e:
        print(f"  Error during Wave.mat evaluation: {e}")


def _extract_fir_with_predictor(results, omega, G_smoothed, predict_fn,
                                 config, output_dir, label="GP"):
    """Shared FIR extraction for both GP and classical/ML paths."""
    print(f"\n{'='*70}\nExtracting FIR model from {label} predictions\n{'='*70}")
    validation_mat = _resolve_validation_mat(config)
    try:
        from gp_to_fir_direct_fixed import gp_to_fir_direct_pipeline as _pipe
        fir_results = _pipe(
            omega=omega, G=G_smoothed, gp_predict_func=predict_fn,
            mat_file=validation_mat, output_dir=output_dir,
            N_fft=None, fir_length=config.fir_length)
        results['fir_extraction'] = fir_results
        print(f"FIR extraction complete. Results saved to {output_dir}")
        _evaluate_wave_mat(fir_results, results, config, output_dir)
    except ImportError:
        try:
            from gp_to_fir_direct import gp_to_fir_direct_pipeline as _pipe2
            fir_results = _pipe2(
                omega=omega, G=G_smoothed, gp_predict_func=predict_fn,
                mat_file=validation_mat, output_dir=output_dir,
                fir_length=config.fir_length)
            results['fir_extraction'] = fir_results
        except ImportError:
            print("Warning: gp_to_fir_direct modules not available")
        except Exception as e:
            print(f"Error during FIR extraction: {e}")
    except Exception as e:
        print(f"Error during FIR extraction: {e}")

# ── Main orchestrator ────────────────────────────────────────────────

def run_gp_pipeline(config) -> Dict:
    """Run the complete frequency response -> regression pipeline."""
    output_dir = Path(config.out_dir); output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1
    omega, G_complex, _ = _load_frequency_data(config, output_dir)

    # Step 2: branch by method
    if hasattr(config, 'is_gp') and not config.is_gp:
        return _run_classical_method(omega, G_complex, config, output_dir)

    # GP path
    X = omega.reshape(-1, 1)
    X_gp = np.log10(X) if config.log_frequency else X
    kp = {}
    if config.kernel == 'matern' and config.nu is not None:
        kp['nu'] = config.nu
    print(f"Creating {config.kernel} kernel...")

    # Step 3
    if config.gp_mode == 'separate':
        res, Gp, ga, gb, Xs, sa, sb = _fit_gp_separate(omega, G_complex, X_gp, config, output_dir, kp)
    else:
        res, Gp, ga, gb, Xs, sa, sb = _fit_gp_polar(omega, G_complex, X_gp, config, output_dir, kp)

    res['config'] = {'kernel': config.kernel, 'gp_mode': config.gp_mode,
                     'normalize': config.normalize, 'log_frequency': config.log_frequency,
                     'optimize': config.optimize}
    print(f"GP regression complete. Output saved to {output_dir}")

    # Step 4
    if config.extract_fir:
        pfn = _build_gp_predict_closure(config, omega, ga, gb, Xs, sa, sb)
        _extract_fir_with_predictor(res, omega, Gp, pfn, config, output_dir)

    plt.close('all'); gc.collect()
    return res

# ── Classical / ML identification ────────────────────────────────────

def run_unified_system_identification(
    omega: np.ndarray, G_complex: np.ndarray, method: str = 'gp',
    config=None, output_dir: Optional[Path] = None, return_predictor: bool = False,
) -> Union[Dict, Tuple[Dict, np.ndarray, object]]:
    """Run classical or ML system identification."""
    from src.classical_methods.ml_methods import create_estimator
    from src.gpr.visualization import configure_plot_style

    if method == 'gp':
        return {}

    if method in ['lpm', 'lrmp']:
        U, Y = np.ones_like(G_complex), G_complex
        if method == 'lpm':
            est = create_estimator('lpm', order=2, half_window=5)
            est.fit(omega, Y, U, estimate_transient=True)
        else:
            est = create_estimator('lrmp', prior_poles=[0.9+0.1j, 0.9-0.1j, 0.8, 0.7],
                                   order=5, half_window=10)
            est.fit(omega, Y, U, Ts=0.01)
        Gp = est.predict(omega)
    elif method in ['rf', 'gbr', 'svm']:
        ml_kw = {'rf': {'n_estimators':100,'max_depth':10},
                 'gbr': {'n_estimators':100,'learning_rate':0.1,'max_depth':5},
                 'svm': {'kernel':'rbf','C':1.0,'gamma':'scale'}}[method]
        est = create_estimator(method, normalize=True, **ml_kw)
        est.fit(omega, G_complex); Gp = est.predict(omega)
    else:
        nn = 2 if not hasattr(config, 'n_numerator') else config.n_numerator
        nd = 4 if not hasattr(config, 'n_denominator') else config.n_denominator
        est = create_estimator(method, n_numerator=nn, n_denominator=nd)
        if method == 'ml':
            est.fit(omega, G_complex, X_measured=np.ones_like(G_complex), Y_measured=G_complex)
        else:
            est.fit(omega, G_complex)
        Gp = est.predict(omega)

    res_c = G_complex - Gp
    rmse = np.sqrt(np.mean(np.abs(res_c)**2))
    rr = np.sqrt(np.mean((np.real(G_complex)-np.real(Gp))**2))
    ri = np.sqrt(np.mean((np.imag(G_complex)-np.imag(Gp))**2))
    r2r = 1 - np.sum((np.real(G_complex)-np.real(Gp))**2) / np.sum((np.real(G_complex)-np.mean(np.real(G_complex)))**2)
    r2i = 1 - np.sum((np.imag(G_complex)-np.imag(Gp))**2) / np.sum((np.imag(G_complex)-np.mean(np.imag(G_complex)))**2)
    results = {'method': method, 'rmse': float(rmse), 'rmse_real': float(rr),
               'rmse_imag': float(ri), 'r2_real': float(r2r), 'r2_imag': float(r2i)}

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        configure_plot_style()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(np.real(G_complex), np.imag(G_complex), 'k.', markersize=14, label='Measured', alpha=0.6)
        ax.plot(np.real(Gp), np.imag(Gp), 'r-', linewidth=4.0, label=f'{method.upper()} fit')
        ax.set_xlabel('Real{G(jw)}', fontsize=32, fontweight='bold')
        ax.set_ylabel('Imag{G(jw)}', fontsize=32, fontweight='bold')
        ax.set_title(f'Nyquist Plot - {method.upper()} (RMSE={rmse:.3e})',
                     fontsize=36, fontweight='bold', pad=20)
        ax.legend(fontsize=26, framealpha=0.9, edgecolor='black', loc='best')
        ax.grid(True, alpha=0.3, linewidth=1.5); ax.axis('equal')
        ax.tick_params(labelsize=24, width=2.5, length=10)
        plt.tight_layout()
        plt.savefig(output_dir / f'{method}_complex_nyquist.png', dpi=300, bbox_inches='tight')
        plt.close()

    return (results, Gp, est) if return_predictor else results
