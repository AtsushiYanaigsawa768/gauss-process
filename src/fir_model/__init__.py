# FIR model identification and validation
#
# Modules:
#   fir_helpers          - Low-level helper functions (uniform grid, Hermitian, IDFT)
#   fir_fitting          - GP-to-FIR pipeline (paper mode + legacy irfft)
#   fir_validation       - Consolidated time-domain validation utilities
#   fir_legacy           - Legacy IRFFT-based GP-to-FIR (original gp_to_fir_direct.py)
#   lms_filter           - Basic LMS adaptive FIR identification
#   rls_filter           - RLS-based real-time FIR identification
#   partial_update_lms   - Partial-Update (M-Max) LMS identification
#   kernel_regularized   - Kernel-regularized FIR (DC, SS2, SI kernels)
