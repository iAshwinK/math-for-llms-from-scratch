"""Tests for the three readings of AB.

Two layers, and both matter:

* **Oracle tests** — every implementation must agree with `np.linalg`'s `@`.
  This catches the ordinary bugs.
* **Property tests** — the algebraic identities that make matrix
  multiplication what it is: associativity, distributivity, the transpose
  rule, and the rank bound. These catch the bugs where you get the right
  answer on square matrices and the wrong one on tall ones.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from conftest import ENTRIES, conformable_pair
from linalg_lab._util import ATOL, RTOL, Matrix
from linalg_lab.matmul import (
    matmul_broadcast,
    matmul_columns,
    matmul_outer,
    matmul_rows,
)

MatMul = Callable[[Matrix, Matrix], Matrix]

IMPLEMENTATIONS: list[MatMul] = [
    matmul_rows,
    matmul_columns,
    matmul_outer,
    matmul_broadcast,
]
IDS = [f.__name__ for f in IMPLEMENTATIONS]


# ------------------------------------------------------- oracle agreement ---


@pytest.mark.parametrize("matmul", IMPLEMENTATIONS, ids=IDS)
@pytest.mark.parametrize(
    ("m", "k", "n"),
    [
        (3, 3, 3),  # square
        (5, 2, 4),  # tall A, wide B
        (2, 5, 3),  # wide A
        (1, 4, 1),  # row times column -> 1x1
        (4, 1, 4),  # single outer product -> rank 1
        (1, 1, 1),  # degenerate
        (7, 3, 1),  # matrix times vector, the Ax = b shape
    ],
    ids=["square", "tall", "wide", "inner", "outer", "scalar", "Ax"],
)
def test_matches_numpy_oracle(
    matmul: MatMul, m: int, k: int, n: int, rng: np.random.Generator
) -> None:
    A = rng.standard_normal((m, k))
    B = rng.standard_normal((k, n))
    assert np.allclose(matmul(A, B), A @ B, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("matmul", IMPLEMENTATIONS, ids=IDS)
@settings(max_examples=50, deadline=None)
@given(pair=conformable_pair())
def test_matches_numpy_oracle_property(
    matmul: MatMul, pair: tuple[Matrix, Matrix]
) -> None:
    A, B = pair
    assert np.allclose(matmul(A, B), A @ B, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("matmul", IMPLEMENTATIONS, ids=IDS)
@settings(max_examples=50, deadline=None)
@given(pair=conformable_pair())
def test_shape_is_m_by_n(matmul: MatMul, pair: tuple[Matrix, Matrix]) -> None:
    A, B = pair
    C = matmul(A, B)
    assert C.shape == (A.shape[0], B.shape[1])
    assert C.dtype == np.float64


# ------------------------------------------------ algebraic properties -----


@settings(max_examples=30, deadline=None)
@given(
    A=arrays(np.float64, (4, 3), elements=ENTRIES),
    B=arrays(np.float64, (3, 5), elements=ENTRIES),
    C=arrays(np.float64, (5, 2), elements=ENTRIES),
)
def test_associativity(A: Matrix, B: Matrix, C: Matrix) -> None:
    """(AB)C == A(BC). The identity behind the cost question in §3."""
    left = matmul_columns(matmul_columns(A, B), C)
    right = matmul_columns(A, matmul_columns(B, C))
    assert np.allclose(left, right, rtol=1e-8, atol=1e-8)


@settings(max_examples=30, deadline=None)
@given(
    A=arrays(np.float64, (4, 3), elements=ENTRIES),
    B=arrays(np.float64, (3, 5), elements=ENTRIES),
    C=arrays(np.float64, (3, 5), elements=ENTRIES),
)
def test_distributivity(A: Matrix, B: Matrix, C: Matrix) -> None:
    """A(B + C) == AB + AC."""
    left = matmul_columns(A, B + C)
    right = matmul_columns(A, B) + matmul_columns(A, C)
    assert np.allclose(left, right, rtol=1e-8, atol=1e-8)


@settings(max_examples=30, deadline=None)
@given(pair=conformable_pair())
def test_transpose_reverses_order(pair: tuple[Matrix, Matrix]) -> None:
    """(AB)^T == B^T A^T."""
    A, B = pair
    left = matmul_columns(A, B).T
    right = matmul_columns(B.T, A.T)
    assert np.allclose(left, right, rtol=1e-8, atol=1e-8)


@settings(max_examples=30, deadline=None)
@given(A=arrays(np.float64, (5, 5), elements=ENTRIES))
def test_identity_is_neutral(A: Matrix) -> None:
    ident = np.eye(5)
    assert np.allclose(matmul_columns(A, ident), A, rtol=1e-9, atol=1e-9)
    assert np.allclose(matmul_columns(ident, A), A, rtol=1e-9, atol=1e-9)


@settings(max_examples=30, deadline=None)
@given(
    a=arrays(np.float64, (6, 1), elements=ENTRIES),
    b=arrays(np.float64, (1, 4), elements=ENTRIES),
)
def test_outer_product_has_rank_at_most_one(a: Matrix, b: Matrix) -> None:
    """k = 1 means the product is a single rank-1 term.

    This is the whole content of reading 3, and the seed of Eckart-Young
    in week 11. np.linalg is forbidden in src/ but it is the oracle here.
    """
    C = matmul_outer(a, b)
    assert np.linalg.matrix_rank(C, tol=1e-9) <= 1


@settings(max_examples=30, deadline=None)
@given(pair=conformable_pair())
def test_all_implementations_agree(pair: tuple[Matrix, Matrix]) -> None:
    """The three readings are readings of one object, not three operations."""
    A, B = pair
    reference = matmul_rows(A, B)
    for impl in (matmul_columns, matmul_outer, matmul_broadcast):
        assert np.allclose(impl(A, B), reference, rtol=1e-9, atol=1e-9)


# --------------------------------------------------- shape validation ------


@pytest.mark.parametrize("matmul", IMPLEMENTATIONS, ids=IDS)
def test_nonconformable_raises(matmul: MatMul) -> None:
    A = np.ones((3, 4))
    B = np.ones((5, 2))
    with pytest.raises(ValueError, match="not conformable"):
        matmul(A, B)


@pytest.mark.parametrize("matmul", IMPLEMENTATIONS, ids=IDS)
def test_1d_input_raises(matmul: MatMul) -> None:
    """`(k,)` is not `(k, 1)`. Shape discipline starts here, not in week 5."""
    A = np.ones((3, 4))
    b = np.ones(4)
    with pytest.raises(ValueError, match="must be 2-D"):
        matmul(A, b)


@settings(max_examples=20, deadline=None)
@given(n=st.integers(1, 6))
def test_column_reading_is_a_linear_combination(n: int) -> None:
    """Explicitly check the claim in the docstring of `matmul_columns`.

    Column j of AB equals sum_k B[k, j] * A[:, k] — the columns of A,
    weighted by column j of B.
    """
    rng = np.random.default_rng(n)
    A = rng.standard_normal((4, n))
    B = rng.standard_normal((n, 3))
    C = matmul_columns(A, B)
    for j in range(3):
        combination = np.zeros(4)
        for k in range(n):
            combination = combination + B[k, j] * A[:, k]
        assert np.allclose(C[:, j], combination, rtol=1e-9, atol=1e-9)
