"""
Gaussian Process kernel library.

Provides Kernel ABC, CombinedKernel base, 12 concrete kernels, and
a ``create_kernel()`` factory.  Kernels: RBFKernel, MaternKernel,
RationalQuadraticKernel, ExponentialKernel, TCKernel, DCKernel,
DIKernel, FirstOrderStableSplineKernel, SecondOrderStableSplineKernel,
HighFrequencyStableSplineKernel, StableSplineKernel.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

# -- Helper functions -------------------------------------------------------

def _prepare_kernel_inputs(X: np.ndarray) -> np.ndarray:
    """Flatten *X* to a 1-D float array for element-wise kernel computations."""
    return np.asarray(X, dtype=float).reshape(-1)

def _heaviside(x: np.ndarray) -> np.ndarray:
    """Element-wise Heaviside step function (H(x) = 1 for x >= 0)."""
    return (x >= 0.0).astype(float)

# -- Kernel base classes ----------------------------------------------------

class Kernel(ABC):
    """Abstract base class for all GP kernels."""
    def __init__(self, **kwargs):
        self.params = kwargs
        self.bounds = self._get_default_bounds()

    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute kernel matrix K(X1, X2). If X2 is None, compute K(X1, X1)."""

    @abstractmethod
    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        """Return default bounds for hyperparameter optimization."""

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Get current hyperparameters as array."""

    @abstractmethod
    def set_params(self, params: np.ndarray) -> None:
        """Set hyperparameters from array."""

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of hyperparameters."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"


class CombinedKernel(Kernel):
    """Base class for combined kernels (sum, product)."""
    def __init__(self, kernels: List[Kernel]):
        self.kernels = kernels
        super().__init__()

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        bounds: List[Tuple[float, float]] = []
        for kernel in self.kernels:
            bounds.extend(kernel.bounds)
        return bounds

    def get_params(self) -> np.ndarray:
        params = []
        for kernel in self.kernels:
            params.extend(kernel.get_params())
        return np.array(params)

    def set_params(self, params: np.ndarray) -> None:
        idx = 0
        for kernel in self.kernels:
            n = kernel.n_params
            kernel.set_params(params[idx:idx + n])
            idx += n

    @property
    def n_params(self) -> int:
        return sum(k.n_params for k in self.kernels)

# -- Standard GP kernels ----------------------------------------------------

class RBFKernel(Kernel):
    """Radial Basis Function (Squared Exponential) kernel."""
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        super().__init__(length_scale=length_scale, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        dists = cdist(X1, X2, metric='sqeuclidean')
        return self.params['variance'] * np.exp(
            -0.5 * dists / (self.params['length_scale'] ** 2))

    def _get_default_bounds(self):
        return [(1e-3, 1e3), (1e-3, 1e3)]
    def get_params(self):
        return np.array([self.params['length_scale'], self.params['variance']])
    def set_params(self, params):
        self.params['length_scale'] = params[0]; self.params['variance'] = params[1]
    @property
    def n_params(self):
        return 2


class MaternKernel(Kernel):
    """Matern kernel with parameter nu."""
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, nu: float = 1.5):
        super().__init__(length_scale=length_scale, variance=variance, nu=nu)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        dists = cdist(X1, X2) / self.params['length_scale']
        nu = self.params['nu']
        if nu == 0.5:
            K = np.exp(-dists)
        elif nu == 1.5:
            K = (1.0 + np.sqrt(3.0) * dists) * np.exp(-np.sqrt(3.0) * dists)
        elif nu == 2.5:
            K = (1.0 + np.sqrt(5.0) * dists + 5.0 / 3.0 * dists ** 2) * np.exp(
                -np.sqrt(5.0) * dists)
        else:
            from scipy.special import kv, gamma
            K = dists ** nu
            K *= kv(nu, np.sqrt(2.0 * nu) * dists)
            K *= 2.0 ** (1.0 - nu) / gamma(nu)
            K[dists == 0] = 1.0
        K *= self.params['variance']
        return K

    def _get_default_bounds(self):
        return [(1e-3, 1e3), (1e-3, 1e3)]
    def get_params(self):
        return np.array([self.params['length_scale'], self.params['variance']])
    def set_params(self, params):
        self.params['length_scale'] = params[0]; self.params['variance'] = params[1]
    @property
    def n_params(self):
        return 2


class RationalQuadraticKernel(Kernel):
    """Rational Quadratic kernel."""
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, alpha: float = 1.0):
        super().__init__(length_scale=length_scale, variance=variance, alpha=alpha)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        dists_sq = cdist(X1, X2, metric='sqeuclidean')
        return self.params['variance'] * (
            1.0 + dists_sq / (2.0 * self.params['alpha'] * self.params['length_scale'] ** 2)
        ) ** (-self.params['alpha'])

    def _get_default_bounds(self):
        return [(1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3)]
    def get_params(self):
        return np.array([self.params['length_scale'], self.params['variance'], self.params['alpha']])
    def set_params(self, params):
        self.params['length_scale'] = params[0]; self.params['variance'] = params[1]; self.params['alpha'] = params[2]
    @property
    def n_params(self):
        return 3

# -- Frequency-domain kernels -----------------------------------------------

class ExponentialKernel(Kernel):
    """First-order stable spline kernel with Heaviside support."""
    def __init__(self, omega: float = 1.0, variance: float = 1.0):
        super().__init__(omega=omega, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        x1, x2 = _prepare_kernel_inputs(X1), _prepare_kernel_inputs(X2)
        H1, H2 = _heaviside(x1), _heaviside(x2)
        sum_grid = x1[:, None] + x2[None, :]
        K = np.exp(-self.params['omega'] * sum_grid)
        K *= H1[:, None] * H2[None, :]
        return self.params['variance'] * K

    def _get_default_bounds(self):
        return [(1e-3, 1e3), (1e-3, 1e3)]
    def get_params(self):
        return np.array([self.params['omega'], self.params['variance']])
    def set_params(self, params):
        self.params['omega'] = params[0]; self.params['variance'] = params[1]
    @property
    def n_params(self):
        return 2


class TCKernel(Kernel):
    """Turned Correlated kernel with Heaviside support."""
    def __init__(self, omega: float = 1.0, variance: float = 1.0):
        super().__init__(omega=omega, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        x1, x2 = _prepare_kernel_inputs(X1), _prepare_kernel_inputs(X2)
        H1, H2 = _heaviside(x1), _heaviside(x2)
        max_grid = np.maximum.outer(x1, x2)
        K = np.exp(-self.params['omega'] * max_grid)
        K *= H1[:, None] * H2[None, :]
        return self.params['variance'] * K

    def _get_default_bounds(self):
        return [(1e-3, 1e3), (1e-3, 1e3)]
    def get_params(self):
        return np.array([self.params['omega'], self.params['variance']])
    def set_params(self, params):
        self.params['omega'] = params[0]; self.params['variance'] = params[1]
    @property
    def n_params(self):
        return 2


class DCKernel(Kernel):
    """Diagonal correlated kernel."""
    def __init__(self, alpha: float = 0.9, beta: float = 1.0, rho: float = 0.5):
        super().__init__(alpha=alpha, beta=beta, rho=rho)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        i = np.rint(_prepare_kernel_inputs(X1)).astype(int)
        j = np.rint(_prepare_kernel_inputs(X2)).astype(int)
        sum_idx = (i[:, None] + j[None, :]) / 2.0
        diff_idx = np.abs(i[:, None] - j[None, :])
        K = (self.params['alpha'] ** sum_idx) * (self.params['rho'] ** diff_idx)
        return self.params['beta'] * K

    def _get_default_bounds(self):
        return [(1e-3, 0.999), (1e-3, 1e3), (-0.999, 0.999)]
    def get_params(self):
        return np.array([self.params['alpha'], self.params['beta'], self.params['rho']])
    def set_params(self, params):
        self.params['alpha'] = params[0]; self.params['beta'] = params[1]; self.params['rho'] = params[2]
    @property
    def n_params(self):
        return 3


class DIKernel(Kernel):
    """Diagonal-independent kernel."""
    def __init__(self, beta: float = 1.0, alpha: float = 0.9):
        super().__init__(beta=beta, alpha=alpha)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        i = np.rint(_prepare_kernel_inputs(X1)).astype(int)
        j = np.rint(_prepare_kernel_inputs(X2)).astype(int)
        K = np.zeros((i.size, j.size), dtype=float)
        diag_mask = i[:, None] == j[None, :]
        if np.any(diag_mask):
            row_idx, col_idx = np.where(diag_mask)
            K[row_idx, col_idx] = self.params['beta'] * (self.params['alpha'] ** i[row_idx])
        return K

    def _get_default_bounds(self):
        return [(1e-3, 1e3), (1e-3, 0.999)]
    def get_params(self):
        return np.array([self.params['beta'], self.params['alpha']])
    def set_params(self, params):
        self.params['beta'] = params[0]; self.params['alpha'] = params[1]
    @property
    def n_params(self):
        return 2

# -- Stable-spline kernels --------------------------------------------------

class FirstOrderStableSplineKernel(Kernel):
    """First-order stable spline kernel."""
    def __init__(self, beta: float = 1.0, variance: float = 1.0):
        super().__init__(beta=beta, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        s, t = _prepare_kernel_inputs(X1), _prepare_kernel_inputs(X2)
        return self.params['variance'] * np.exp(-self.params['beta'] * np.minimum.outer(s, t))

    def _get_default_bounds(self):
        return [(1e-3, 1e3), (1e-3, 1e3)]
    def get_params(self):
        return np.array([self.params['beta'], self.params['variance']])
    def set_params(self, params):
        self.params['beta'] = params[0]; self.params['variance'] = params[1]
    @property
    def n_params(self):
        return 2


class SecondOrderStableSplineKernel(Kernel):
    """Second-order stable spline kernel."""
    def __init__(self, beta: float = 1.0, variance: float = 1.0):
        super().__init__(beta=beta, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        s, t = _prepare_kernel_inputs(X1), _prepare_kernel_inputs(X2)
        beta = self.params['beta']
        sum_grid = s[:, None] + t[None, :]
        max_grid = np.maximum.outer(s, t)
        first_term = 0.5 * np.exp(-beta * (sum_grid + max_grid))
        second_term = (1.0 / 6.0) * np.exp(-3.0 * beta * max_grid)
        K = first_term - second_term
        return self.params['variance'] * K

    def _get_default_bounds(self):
        return [(1e-3, 1e3), (1e-3, 1e3)]
    def get_params(self):
        return np.array([self.params['beta'], self.params['variance']])
    def set_params(self, params):
        self.params['beta'] = params[0]; self.params['variance'] = params[1]
    @property
    def n_params(self):
        return 2


class HighFrequencyStableSplineKernel(Kernel):
    """High-frequency stable spline kernel."""
    def __init__(self, beta: float = 1.0, variance: float = 1.0):
        super().__init__(beta=beta, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        s, t = _prepare_kernel_inputs(X1), _prepare_kernel_inputs(X2)
        s_idx, t_idx = np.rint(s).astype(int), np.rint(t).astype(int)
        sign = np.power(-1.0, s_idx[:, None] + t_idx[None, :])
        max_term = np.maximum(np.exp(-self.params['beta'] * s[:, None]),
                              np.exp(-self.params['beta'] * t[None, :]))
        return self.params['variance'] * sign * max_term

    def _get_default_bounds(self):
        return [(1e-3, 1e3), (1e-3, 1e3)]
    def get_params(self):
        return np.array([self.params['beta'], self.params['variance']])
    def set_params(self, params):
        self.params['beta'] = params[0]; self.params['variance'] = params[1]
    @property
    def n_params(self):
        return 2


class StableSplineKernel(Kernel):
    """Non-stationary stable spline kernel with exponential warping."""
    def __init__(self, beta: float = 0.5, variance: float = 1.0):
        super().__init__(beta=beta, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        x1, x2 = _prepare_kernel_inputs(X1), _prepare_kernel_inputs(X2)
        beta = self.params['beta']
        exp_x1, exp_x2 = np.exp(-beta * x1), np.exp(-beta * x2)
        r = np.minimum.outer(exp_x1, exp_x2)
        R = np.maximum.outer(exp_x1, exp_x2)
        K = 0.5 * r ** 2 * (R - r / 3.0)
        return self.params['variance'] * K

    def _get_default_bounds(self):
        return [(1e-3, 1e3), (1e-3, 1e3)]
    def get_params(self):
        return np.array([self.params['beta'], self.params['variance']])
    def set_params(self, params):
        self.params['beta'] = params[0]; self.params['variance'] = params[1]
    @property
    def n_params(self):
        return 2

# -- Kernel factory ----------------------------------------------------------

def create_kernel(kernel_type: str, **kwargs) -> Kernel:
    """Create a kernel instance by short name.

    Available: rbf, matern, matern12, matern32, matern52, rq,
    exp, tc, dc, di, ss1, ss2, sshf, stable_spline
    """
    kernel_map = {
        'rbf': RBFKernel,
        'matern': MaternKernel,
        'matern12': lambda **kw: MaternKernel(nu=0.5, **{k: v for k, v in kw.items() if k != 'nu'}),
        'matern32': lambda **kw: MaternKernel(nu=1.5, **{k: v for k, v in kw.items() if k != 'nu'}),
        'matern52': lambda **kw: MaternKernel(nu=2.5, **{k: v for k, v in kw.items() if k != 'nu'}),
        'rq': RationalQuadraticKernel,
        'exp': ExponentialKernel,
        'tc': TCKernel,
        'dc': DCKernel,
        'di': DIKernel,
        'ss1': FirstOrderStableSplineKernel,
        'ss2': SecondOrderStableSplineKernel,
        'sshf': HighFrequencyStableSplineKernel,
        'stable_spline': StableSplineKernel,
    }
    if kernel_type not in kernel_map:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Available: {list(kernel_map.keys())}")
    creator = kernel_map[kernel_type]
    return creator(**kwargs) if callable(creator) else creator(**kwargs)
