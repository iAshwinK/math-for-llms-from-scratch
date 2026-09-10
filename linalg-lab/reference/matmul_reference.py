"""Matrix multiplication, three ways.

Strang's point in lecture 3 is that `AB` is one object with several equally
valid readings, and which reading you hold in your head determines whether the
rest of the course feels obvious or feels memorised:

1. **Row times column.** `C[i, j]` is the dot product of row `i` of A with
   column `j` of B. This is the definition, and the least useful picture.
2. **Columns.** Column `j` of C is a *linear combination of the columns of A*,
   with the weights taken from column `j` of B. This is the reading that makes
   column space, rank and `Ax = b` fall out later.
3. **Outer products.** `C` is a sum over `k` of rank-1 matrices
   `a_k b_k^T`, where `a_k` is column `k` of A and `b_k` is row `k` of B.
   This is the reading that SVD and low-rank approximation are built on: keep
   the first few terms of this sum and you have `rank_k_approx`.

All three produce the same array. They differ in what they loop over — rows,
columns, or the inner dimension — and in their memory access pattern, which
is why they benchmark differently.

House rules (see README §rules): no `np.linalg`, no `@` in this module (`@`
*is* the thing being implemented), and no loops over individual entries. Each
function loops exactly once, over rows / columns / rank, and does the rest
with a single broadcast expression.
"""

from __future__ import annotations

import numpy as np

from ._util import Matrix, require_conformable

__all__ = [
    "matmul_broadcast",
    "matmul_columns",
    "matmul_outer",
    "matmul_rows",
]


def matmul_rows(A: Matrix, B: Matrix) -> Matrix:
    """Reading 1 — row times column, one row of C at a time.

    Row `i` of C is `sum_k A[i, k] * B[k, :]`. Broadcasting `A[i][:, None]`
    against B multiplies each row of B by its scalar weight; summing over
    axis 0 collapses the inner dimension.

    Loops: `m` iterations (one per row of A).
    """
    require_conformable(A, B)
    m, _ = A.shape
    n = B.shape[1]
    C = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        # A[i] is (k,) -> (k, 1); B is (k, n); product is (k, n)
        C[i, :] = np.sum(A[i][:, None] * B, axis=0)
    return C


def matmul_columns(A: Matrix, B: Matrix) -> Matrix:
    """Reading 2 — each column of C is a linear combination of A's columns.

    Column `j` of C is `A @ B[:, j]`, i.e. the columns of A weighted by the
    entries of column `j` of B. Broadcasting `B[:, j]` (shape `(k,)`) against
    A (shape `(m, k)`) scales each *column* of A; summing over axis 1 adds
    them up.

    This is the reading to make automatic. `Ax = b` is solvable exactly when
    `b` is one of these combinations — which is to say, when `b` lies in the
    column space of A.

    Loops: `n` iterations (one per column of B).
    """
    require_conformable(A, B)
    m = A.shape[0]
    n = B.shape[1]
    C = np.empty((m, n), dtype=np.float64)
    for j in range(n):
        # A is (m, k); B[:, j] is (k,) broadcast across rows; sum over k
        C[:, j] = np.sum(A * B[:, j], axis=1)
    return C


def matmul_outer(A: Matrix, B: Matrix) -> Matrix:
    """Reading 3 — sum of rank-1 outer products.

    `C = sum_k outer(A[:, k], B[k, :])`. Each term is a full `m x n` matrix
    of rank 1; the sum of `k` of them has rank at most `k`, which is the
    entire content of "rank of a product is bounded by the inner dimension".

    Truncate this sum after the largest few terms and you have low-rank
    approximation — the same identity that week 11's `rank_k_approx` uses
    on the SVD.

    Loops: `k` iterations (one per rank-1 term).
    """
    require_conformable(A, B)
    m, k = A.shape
    n = B.shape[1]
    C = np.zeros((m, n), dtype=np.float64)
    for p in range(k):
        # outer(A[:, p], B[p, :]) via broadcasting: (m, 1) * (1, n) -> (m, n)
        C += A[:, p][:, None] * B[p, :][None, :]
    return C


def matmul_broadcast(A: Matrix, B: Matrix) -> Matrix:
    """No loop at all — and a lesson in when that is a bad idea.

    `A[:, :, None] * B[None, :, :]` materialises an `m x k x n` array before
    reducing it. For `n = 1000` in float64 that is 8 GB. It is the cleanest
    expression of the definition and completely unusable at scale.

    Keep it in the benchmark table precisely because it is the trap: NumPy
    will happily let you allocate an array 1000x larger than your answer.
    """
    require_conformable(A, B)
    product = A[:, :, None] * B[None, :, :]
    return np.asarray(np.sum(product, axis=1), dtype=np.float64)
