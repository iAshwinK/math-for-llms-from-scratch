"""Shared fixtures and the matrix strategies every test module reuses.

`matrices()` is the workhorse. It is deliberately written once, here, because
every week's properties will be stated against the same families of matrix:
square, tall, wide, singular, ill-conditioned.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from linalg_lab._util import Matrix, default_rng

# Entries bounded away from the extremes: we are testing algorithms, not
# float64's behaviour at 1e300. Denormals and infinities get their own
# targeted tests rather than polluting every property.
ENTRIES = st.floats(
    min_value=-1e3,
    max_value=1e3,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


def matrices(
    min_rows: int = 1,
    max_rows: int = 8,
    min_cols: int = 1,
    max_cols: int = 8,
) -> st.SearchStrategy[Matrix]:
    """Float64 2-D arrays with dimensions in the given ranges."""
    return arrays(
        dtype=np.float64,
        shape=array_shapes(
            min_dims=2,
            max_dims=2,
            min_side=min(min_rows, min_cols),
            max_side=max(max_rows, max_cols),
        ),
        elements=ENTRIES,
    )


@st.composite
def conformable_pair(draw: st.DrawFn, max_side: int = 8) -> tuple[Matrix, Matrix]:
    """A pair (A, B) with cols(A) == rows(B), so A @ B is defined."""
    m = draw(st.integers(1, max_side))
    k = draw(st.integers(1, max_side))
    n = draw(st.integers(1, max_side))
    A = draw(arrays(np.float64, (m, k), elements=ENTRIES))
    B = draw(arrays(np.float64, (k, n), elements=ENTRIES))
    return A, B


@pytest.fixture
def rng() -> np.random.Generator:
    return default_rng(20260901)


@pytest.fixture
def hilbert_5() -> Matrix:
    """The Hilbert matrix — the standard ill-conditioned test case.

    cond(H_5) is around 5e5; H_12 is past what float64 can solve at all.
    Week 12 comes back to this.
    """
    i = np.arange(1, 6, dtype=np.float64)
    H = 1.0 / (i[:, None] + i[None, :] - 1.0)
    return np.asarray(H, dtype=np.float64)
