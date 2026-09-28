"""heroes_control may import only numpy, scipy, the stdlib and itself; heroes_safety only
numpy, the stdlib and itself. Allowlist, so a new dependency fails loudly (spec 2.1)."""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
ALLOWED = {
    "heroes_control": {"numpy", "scipy", "heroes_control"},
    "heroes_safety": {"numpy", "heroes_safety"},
}


def imported_modules(path: Path) -> list[str]:
    mods = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.append(node.module or "")
    return mods


@pytest.mark.parametrize("pkg", sorted(ALLOWED))
def test_pure_packages_import_boundary(pkg):
    allowed = ALLOWED[pkg] | set(sys.stdlib_module_names)
    offenders = [
        f"{path.name}: {m}"
        for path in (ROOT / pkg).rglob("*.py")
        for m in imported_modules(path)
        if m.split(".")[0] not in allowed
    ]
    assert not offenders, offenders


def test_boundary_scan_catches_violation(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import mujoco\nfrom heroes_sim.plant import Plant\n")
    assert imported_modules(bad) == ["mujoco", "heroes_sim.plant"]
