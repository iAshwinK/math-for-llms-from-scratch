"""The trivial test from the Week 0 checklist.

Its job is to prove the scaffolding works: the package is importable from the
`src/` layout, numpy is present, and CI is wired up correctly. Do not delete
it — when a dependency change breaks the environment, this is the test whose
failure is legible.
"""

from __future__ import annotations

import numpy as np

import linalg_lab
from linalg_lab._util import default_rng


def test_package_imports() -> None:
    assert linalg_lab.__version__ == "0.1.0"


def test_numpy_is_modern() -> None:
    """NPY rules and the Generator API both assume NumPy >= 2."""
    assert int(np.__version__.split(".")[0]) >= 2


def test_rng_is_reproducible() -> None:
    a = default_rng(42).standard_normal(5)
    b = default_rng(42).standard_normal(5)
    assert np.array_equal(a, b)
