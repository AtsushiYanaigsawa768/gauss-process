#!/usr/bin/env python3
"""
config.py

Dataclass-based configuration objects for the system identification pipeline.

Three independent configuration groups:
- GPConfig:        Gaussian Process regression settings (kernel, noise, etc.)
- FIRConfig:       FIR model extraction settings (length, validation file)
- FrequencyConfig: Frequency analysis settings (file count, nd, method)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class GPConfig:
    """Configuration for GP regression.

    Attributes:
        kernel_type:       Kernel name passed to ``create_kernel()``
                           (default: 'rbf').
        kernel_params:     Extra keyword arguments forwarded to the kernel
                           constructor (e.g. ``{'nu': 2.5}`` for Matern).
        noise_variance:    Initial observation-noise variance.  Set to 0 for
                           pure (noise-free) GP regression.
        optimize:          Whether to optimise kernel hyper-parameters.
        n_restarts:        Number of random restarts for gradient-based
                           hyper-parameter optimisation.
        normalize_inputs:  Standardise GP input features before fitting.
        normalize_outputs: Standardise GP targets before fitting.
        log_frequency:     Use log10(omega) as the GP input feature.
        gp_mode:           Decomposition mode -- ``'separate'`` fits real and
                           imaginary parts independently; ``'polar'`` fits
                           log-magnitude and unwrapped phase.
    """
    kernel_type: str = 'rbf'
    kernel_params: Dict = field(default_factory=dict)
    noise_variance: float = 1e-6
    optimize: bool = True
    n_restarts: int = 3
    normalize_inputs: bool = True
    normalize_outputs: bool = True
    log_frequency: bool = True
    gp_mode: str = 'separate'


@dataclass
class FIRConfig:
    """Configuration for FIR extraction.

    Attributes:
        extract_fir:    If True, run FIR model extraction after GP fitting.
        fir_length:     Number of FIR filter taps.
        validation_mat: Path to a ``.mat`` file with ``[time, output, input]``
                        used for time-domain FIR validation.
    """
    extract_fir: bool = False
    fir_length: int = 1024
    validation_mat: Optional[str] = None


@dataclass
class FrequencyConfig:
    """Configuration for frequency analysis.

    Attributes:
        n_files:       Number of MAT files to include when computing the FRF.
        time_duration: Time duration (seconds) to use from each file.
                       ``None`` means use the full recording.
        nd:            Number of frequency grid points (N_d).
        freq_method:   ``'frf'`` for log-scale FRF estimation,
                       ``'fourier'`` for FFT-based estimation.
    """
    n_files: int = 1
    time_duration: Optional[float] = None
    nd: int = 100
    freq_method: str = 'frf'
