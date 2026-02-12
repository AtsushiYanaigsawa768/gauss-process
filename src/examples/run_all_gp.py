#!/usr/bin/env python3
"""
Run all GP regression models sequentially.

All scripts below use data measured over 10 hours x 10 trials for
simplicity and reduced execution time.  To use the full dataset,
modify each individual function accordingly.
"""

# 2: T-distribution GP regression
from src.gpr.t_distribution import main as gpflow_t_main
gpflow_t_main()

# 3: Linear interpolation (not a GP method)
from src.gpr.linear_interpolation import main as linear_main
linear_main()

# 5: Iteratively-trimmed GP (ITGP)
from src.gpr.itgp import main as ITGP_robust_main
ITGP_robust_main()
