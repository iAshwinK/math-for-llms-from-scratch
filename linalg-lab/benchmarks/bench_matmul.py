"""Benchmark the readings of AB against a Java-style triple loop and LAPACK.

Run with:

    uv run python benchmarks/bench_matmul.py            # n = 50, 200
    uv run python benchmarks/bench_matmul.py --full     # adds n = 1000

Writes a markdown table to `benchmarks/results.md`, appending a new section
each week. Two numbers are the point of this table:

* your vectorised version vs the naive triple loop — proof you learned
  vectorisation;
* your vectorised version vs `A @ B` — proof that LAPACK exists and you
  should use it. Roughly 50x, and it does not shrink because you got
  cleverer. Cache blocking and hand-written SIMD are not things you are
  going to beat in NumPy.

Guards worth reading: `matmul_naive` is O(n^3) in interpreted Python and is
skipped above n = 200. `matmul_broadcast` allocates an m*k*n intermediate —
8 GB at n = 1000 — and is skipped above n = 200 for that reason.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linalg_lab._util import Matrix, default_rng
from linalg_lab.matmul import (
    matmul_broadcast,
    matmul_columns,
    matmul_outer,
    matmul_rows,
)

MatMul = Callable[[Matrix, Matrix], Matrix]

# Implementations that blow up on large n, and the size past which we skip.
MAX_N: dict[str, int] = {
    "naive triple loop": 200,
    "matmul_broadcast": 200,
}


def matmul_naive(A: Matrix, B: Matrix) -> Matrix:
    """The Java translation. Present only as the number to beat.

    This is exactly what the instinct from `for (int i = 0; ...)` produces,
    and it is the thing twelve weeks of this project is meant to kill. It
    lives in benchmarks/, never in src/ — house rule 4.
    """
    m, k = A.shape
    n = B.shape[1]
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                total += A[i, p] * B[p, j]
            C[i, j] = total
    return C


def time_it(fn: MatMul, A: Matrix, B: Matrix, repeats: int = 3) -> float:
    """Best-of-N wall time in seconds.

    Minimum, not mean: we want the machine's best effort, unpolluted by
    whatever else the OS decided to do. `perf_counter`, never `time.time`.
    """
    fn(A, B)  # warm up — first call pays page-fault and allocation costs
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(A, B)
        best = min(best, time.perf_counter() - start)
    return best


def run(sizes: list[int]) -> str:
    rng = default_rng(20260901)
    implementations: list[tuple[str, MatMul]] = [
        ("naive triple loop", matmul_naive),
        ("matmul_rows", matmul_rows),
        ("matmul_columns", matmul_columns),
        ("matmul_outer", matmul_outer),
        ("matmul_broadcast", matmul_broadcast),
        ("np @ (LAPACK/BLAS)", lambda A, B: A @ B),
    ]

    rows: list[str] = []
    for n in sizes:
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))

        # Correctness before speed. A fast wrong answer is not a result.
        reference = A @ B

        # Collect raw seconds first, format second. Formatting early and then
        # parsing the string back out is how you end up reporting LAPACK as
        # 2x slower than itself.
        timings: list[tuple[str, float | None]] = []
        for name, fn in implementations:
            if n > MAX_N.get(name, 10**9):
                timings.append((name, None))
                continue

            repeats = 1 if name == "naive triple loop" else 3
            elapsed = time_it(fn, A, B, repeats=repeats)
            assert np.allclose(fn(A, B), reference, rtol=1e-9, atol=1e-9), name
            timings.append((name, elapsed))

        baseline = next(
            (t for name, t in timings if name.startswith("np @") and t is not None),
            None,
        )
        for name, secs in timings:
            if secs is None:
                rows.append(f"| {n} | {name} | skipped | — | — |")
                continue
            ratio = f"{secs / baseline:.0f}x" if baseline else "—"
            rows.append(f"| {n} | {name} | {secs * 1e3:.2f} ms | {ratio} | ✓ |")

    header = (
        f"\n## matmul — week 1 · {datetime.now(UTC):%Y-%m-%d}\n\n"
        f"`{platform.python_version()}` on `{platform.machine()}`, "
        f"NumPy {np.__version__}. Square matrices, best of 3.\n\n"
        "| n | implementation | time | vs LAPACK | matches oracle |\n"
        "|---|---|---|---|---|\n"
    )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="include n = 1000 (slow; skips the two memory-bound impls)",
    )
    args = parser.parse_args()

    sizes = [50, 200, 1000] if args.full else [50, 200]
    table = run(sizes)
    print(table)

    out = Path(__file__).parent / "results.md"
    if not out.exists():
        out.write_text("# Benchmark log\n\nGrown weekly. Oldest first.\n")
    with out.open("a") as fh:
        fh.write(table)
    print(f"appended to {out}")


if __name__ == "__main__":
    main()
