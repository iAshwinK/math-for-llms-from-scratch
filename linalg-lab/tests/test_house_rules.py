"""The house rules, enforced instead of merely written down.

CLAUDE.md and README both state that `np.linalg` is banned in `src/`. Neither
is enforcement — a note in a markdown file does not stop anything. This module
parses the source and fails the build, which does.

The check is AST-based rather than a regex over the text, so `np.linalg` in a
docstring or a comment is fine (the docstrings discuss the oracle constantly)
and only real attribute access or imports trip it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "linalg_lab"

FORBIDDEN_MODULES = {"numpy.linalg", "scipy.linalg", "scipy"}
FORBIDDEN_ATTRS = {"linalg"}


def source_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if p.name != "__init__.py")


def test_source_directory_is_not_empty() -> None:
    """Guard the guard: a typo'd path would silently pass every check below."""
    assert source_files(), f"no modules found under {SRC}"


@pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
def test_no_linalg_imports(path: Path) -> None:
    """House rule 2 — nothing from np.linalg or scipy.linalg in src/."""
    tree = ast.parse(path.read_text(), filename=str(path))
    offences: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in FORBIDDEN_MODULES:
            offences.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Import):
            offences.extend(
                f"line {node.lineno}: import {alias.name}"
                for alias in node.names
                if alias.name in FORBIDDEN_MODULES
            )
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            offences.append(f"line {node.lineno}: .{node.attr} attribute access")

    assert not offences, (
        f"{path.name} reaches for the oracle:\n  "
        + "\n  ".join(offences)
        + "\n\nnp.linalg is allowed in tests/ only. Implement it, or call an "
        "earlier week's implementation."
    )


def test_matmul_does_not_use_matmul_operator() -> None:
    """`matmul.py` implements `@`, so using `@` there would be circular.

    This rule is specific to week 1's module — `@` is allowed everywhere else
    in `src/` by house rule 1.
    """
    path = SRC / "matmul.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult)
    ]
    assert not lines, (
        f"matmul.py uses the `@` operator at line(s) {lines}. "
        "That module is the implementation of `@`; use elementwise ops, "
        "broadcasting and reductions instead."
    )


@pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
def test_no_triple_nested_loops(path: Path) -> None:
    """House rule 4, approximately.

    Genuine element-wise iteration is undecidable in general, but three nested
    `for` loops in numerical code is the Java transliteration essentially
    every time. Two is the legitimate ceiling (a loop over pivots containing
    a loop over columns). Three means the inner operation wasn't vectorised.
    """

    def depth(node: ast.AST, current: int = 0) -> int:
        deepest = current
        for child in ast.iter_child_nodes(node):
            nxt = current + 1 if isinstance(child, ast.For | ast.While) else current
            deepest = max(deepest, depth(child, nxt))
        return deepest

    tree = ast.parse(path.read_text(), filename=str(path))
    assert depth(tree) < 3, (
        f"{path.name} has three or more nested loops. Loop over the sequential "
        "dimension only (pivots, columns, rank-1 terms) and update the "
        "remaining submatrix with one broadcast expression."
    )
