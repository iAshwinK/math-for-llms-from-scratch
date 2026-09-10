"""linalg-lab — the core of numpy.linalg, rebuilt from the mathematics.

Modules land week by week, mirroring the 18.06 syllabus:

    matmul       w1     row / column / outer-product readings of AB
    elimination  w1-2   rref, lu_factor, solve_lu, det
    subspaces    w3-4   rank, nullspace, colspace, four_subspaces
    projections  w5     project, lstsq_normal_equations
    qr           w6     gram_schmidt, modified_gram_schmidt, householder_qr
    eigen        w8-9   power_iteration, inverse_iteration, qr_algorithm
    spd          w9     cholesky, is_positive_definite, nearest_psd
    svd          w10-11 svd, pinv, rank_k_approx
"""

from __future__ import annotations

from .matmul import matmul_broadcast, matmul_columns, matmul_outer, matmul_rows

__version__ = "0.1.0"

__all__ = [
    "matmul_broadcast",
    "matmul_columns",
    "matmul_outer",
    "matmul_rows",
]
