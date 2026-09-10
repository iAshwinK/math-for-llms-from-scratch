"""Shared tolerances, type aliases and shape validation.

Everything in the library speaks `Matrix` / `Vector`, which are just
`np.ndarray` with a float64 dtype attached at the type level. mypy will not
check the *shape* for you — that is what `require_2d` and friends are for.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Float64 everywhere. float32 shows up deliberately, in the numerical-honesty
# experiments, and nowhere else.
Scalar = np.float64
Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

# Default tolerances for comparing against the np.linalg oracle. These are
# loose enough for accumulated float64 error on small matrices and tight
# enough that a genuine algorithmic bug fails the test.
RTOL = 1e-10
ATOL = 1e-12


def as_matrix(A: object) -> Matrix:
    """Coerce input to a 2-D float64 array, copying only when necessary."""
    arr = np.asarray(A, dtype=np.float64)
    if arr.ndim != 2:
        msg = f"expected a 2-D array, got shape {arr.shape}"
        raise ValueError(msg)
    return arr


def require_2d(name: str, A: Matrix) -> None:
    if A.ndim != 2:
        msg = f"{name} must be 2-D, got shape {A.shape}"
        raise ValueError(msg)


def require_conformable(A: Matrix, B: Matrix) -> None:
    """Check that A @ B is defined: cols(A) == rows(B)."""
    require_2d("A", A)
    require_2d("B", B)
    if A.shape[1] != B.shape[0]:
        msg = (
            f"shapes {A.shape} and {B.shape} are not conformable: "
            f"A has {A.shape[1]} columns but B has {B.shape[0]} rows"
        )
        raise ValueError(msg)


def default_rng(seed: int = 0) -> np.random.Generator:
    """The modern Generator API. Never `np.random.seed`."""
    return np.random.default_rng(seed)
