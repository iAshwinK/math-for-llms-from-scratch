"""Matrix multiplication, three ways — YOUR implementation goes here.

Strang's lecture 3 point is that `AB` is one object with three readings, and
which one you hold determines whether the rest of the course feels obvious or
memorised:

1. **Row times column.** `C[i, j]` is row `i` of A dotted with column `j` of
   B. The definition, and the least useful picture.
2. **Columns.** Column `j` of C is a *linear combination of the columns of
   A*, weighted by column `j` of B. This is the reading that makes column
   space, rank and `Ax = b` fall out later.
3. **Outer products.** `C = sum over k of a_k b_k^T` — a sum of rank-1
   matrices. This is the reading SVD and low-rank approximation are built on.

Each function below loops exactly once — over rows, columns, or the inner
dimension — and does the rest with a single broadcast expression.

Constraints for this module specifically:
    - no `np.linalg`
    - no `@` (this module *is* `@`)
    - no loops over individual entries

`tests/test_matmul.py` already contains the tests. Make them pass.
A worked version is in `reference/matmul_reference.py` — open it only after
yours passes, then compare.
"""

from __future__ import annotations

import numpy as np  # you will need this

from ._util import Matrix, require_conformable

__all__ = [
    "matmul_broadcast",
    "matmul_columns",
    "matmul_outer",
    "matmul_rows",
]


def matmul_rows(A: Matrix, B: Matrix) -> Matrix:
    """Reading 1 — build C one row at a time.

    Row `i` of C is `sum_k A[i, k] * B[k, :]`: each row of B scaled by its
    weight from row `i` of A, added up.

    Hint: `A[i]` has shape `(k,)`. You need those k numbers to scale k *rows*
    of B, so they must line up along axis 0. What shape does that need to be,
    and how do you get there?
    """
    require_conformable(A, B)

    if A.shape[1] != B.shape[0]:
        raise ValueError("Invalid matrix shape")

    C: Matrix = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for k in range(A.shape[1]):
            C[i, :] += A[i, k] * B[k, :]
    return C


def matmul_columns(A: Matrix, B: Matrix) -> Matrix:
    """Reading 2 — each column of C is a linear combination of A's columns.

    Column `j` of C is the columns of A weighted by the entries of column `j`
    of B. Make this one automatic: `Ax = b` is solvable exactly when `b` is
    one of these combinations, i.e. when `b` is in the column space of A.

    Hint: `B[:, j]` has shape `(k,)` and A has shape `(m, k)`. Those line up
    on the trailing axis already. What does `A * B[:, j]` do, and what do you
    sum over afterwards?
    """
    require_conformable(A, B)

    C: Matrix = np.zeros((A.shape[0], B.shape[1]))

    for j in range(B.shape[1]):
        C[:, j] = (A * B[:, j]).sum(axis=1)

    return C


def matmul_outer(A: Matrix, B: Matrix) -> Matrix:
    """Reading 3 — accumulate rank-1 outer products.

    `C = sum_p outer(A[:, p], B[p, :])`. Each term is a full `m x n` matrix of
    rank 1; summing `k` of them gives rank at most `k`, which is the entire
    content of "the rank of a product is bounded by the inner dimension".

    Truncate this sum after the largest few terms and you have low-rank
    approximation — week 11's `rank_k_approx` is this identity on the SVD.

    Hint: an outer product is `(m, 1) * (1, n)`. Start C at zeros.
    """
    require_conformable(A, B)
    C: Matrix = np.zeros((A.shape[0], B.shape[1]))

    for j in range(A.shape[1]):
        C += A[:, j].reshape(-1, 1) * B[j, :]

    return C


def matmul_broadcast(A: Matrix, B: Matrix) -> Matrix:
    """No loop at all — and a lesson in when that's the wrong idea.

    Line up A as `(m, k, 1)` and B as `(1, k, n)`, multiply, and sum away the
    middle axis. One expression, no loop.

    Then work out how big that intermediate array is for n = 1000 in float64,
    before you run the benchmark. That number is the point of this function.
    """
    require_conformable(A, B)

    A_3d = A[:, :, np.newaxis]
    B_3d = B[np.newaxis, :, :]

    return (A_3d * B_3d).sum(axis=1)
