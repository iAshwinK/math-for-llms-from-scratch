# linalg-lab

Rebuilding the core of `numpy.linalg` from the mathematics, week by week,
alongside MIT 18.06. The linear algebra is the deliverable; the code is
reinforcement. Ratio is 60/40, not 20/80.

[![CI](https://github.com/USERNAME/linalg-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/linalg-lab/actions/workflows/ci.yml)

## The rules

1. **Allowed in `src/`:** NumPy arrays, elementwise and broadcast ops,
   slicing, `@`, `reshape`, reductions.
2. **Forbidden in `src/`:** anything in `np.linalg` or `scipy.linalg`. No
   `solve`, `inv`, `qr`, `eig`, `svd`, `det`, `cholesky`, `lstsq`, `pinv`.
   (`matmul.py` additionally forbids `@` — it *is* `@`.)
3. **Required in `tests/`:** compare everything against `np.linalg` as the
   oracle. The oracle is allowed there and nowhere else.
4. **No loops over elements.** Loops over *columns* are fine and often
   necessary. Loops over individual entries are a bug — vectorise the inner
   operation with slices.

Rule 4 is the one that builds the Python skill.

## Quickstart

```bash
uv sync --all-extras          # create .venv, install everything
uv run pytest                 # tests
uv run ruff check . && uv run ruff format --check .
uv run mypy                   # strict
uv run pre-commit install     # once, so the hooks fire on commit

uv run python benchmarks/bench_matmul.py          # n = 50, 200
uv run python benchmarks/bench_matmul.py --full   # adds n = 1000
```

## Layout

```
src/linalg_lab/
  matmul.py       w1     row / column / outer-product readings of AB
  elimination.py  w1-2   rref, lu_factor, solve_lu, det
  subspaces.py    w3-4   rank, nullspace, colspace, four_subspaces
  projections.py  w5     project, lstsq_normal_equations
  qr.py           w6     gram_schmidt, modified_gram_schmidt, householder_qr
  eigen.py        w8-9   power_iteration, inverse_iteration, qr_algorithm
  spd.py          w9     cholesky, is_positive_definite, nearest_psd
  svd.py          w10-11 svd, pinv, rank_k_approx
  _util.py               tolerances, shape validation, typed aliases
tests/            pytest + hypothesis
benchmarks/       loop vs vectorised vs np.linalg; results.md grows weekly
notebooks/        exploration only — no logic lives here
apps/             the applications from §5
```

## How this repo is used

I write the implementations. Claude Code is a tutor, not an author — see
`CLAUDE.md` and the skills in `.claude/skills/` (`/week`, `/explain`,
`/stuck`, `/review`).

Each week: stubs in `src/` raise `NotImplementedError`, the tests are already
written, and the week is done when they go green. Worked versions live in
`reference/`, which is **not** imported by anything — open it only after
mine passes, then compare.

Red tests mean the current week is unfinished. That's the intended state.

## Progress

| Week | Module | Status |
|---|---|---|
| 0 | scaffold, CI | ✅ |
| 1 | `matmul` three ways + benchmark | stubs + tests written — implementing |
| 2 | `lu_factor`, `solve_lu` | |
| 3 | `rref`, `rank`, `nullspace_basis` | |

## Week 1 — the three readings of AB

`AB` is one object with three readings, and which one you hold determines
whether the rest of 18.06 feels obvious or memorised.

| Reading | Loops over | The identity |
|---|---|---|
| `matmul_rows` | rows of A | `C[i,j]` = row `i` of A · column `j` of B |
| `matmul_columns` | columns of B | column `j` of C is a linear combination of A's **columns** |
| `matmul_outer` | the inner dimension `k` | `C = Σₖ aₖ bₖᵀ`, a sum of rank-1 matrices |

`matmul_columns` is the one to make automatic: `Ax = b` is solvable exactly
when `b` is one of those combinations, i.e. when `b` is in the column space
of A. `matmul_outer` is the seed of low-rank approximation — truncate the sum
and you have week 11's `rank_k_approx`.

`matmul_broadcast` is included as the trap: no loop at all, but it
materialises an `m×k×n` intermediate, 8 GB at n = 1000.

See `benchmarks/results.md` for the numbers.

## Property list (grown weekly)

- [x] all three readings agree with each other and with `@`
- [x] associativity, distributivity, `(AB)ᵀ = BᵀAᵀ`, identity is neutral
- [x] `rank(ab ᵀ) ≤ 1`
- [ ] LU reconstructs A; `P⁻¹LU = A` with pivoting
- [ ] residual `‖Ax − b‖` small for the solvers
- [ ] Q orthogonal
- [ ] singular values non-negative and descending
- [ ] `rank_k_approx` error matches the Eckart–Young bound
- [ ] `A @ pinv(A) @ A == A`
