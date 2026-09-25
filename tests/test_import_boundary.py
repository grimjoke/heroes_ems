"""heroes_control / heroes_safety must not import sim-side or physics packages."""

import ast
from pathlib import Path

FORBIDDEN = {"mujoco", "myosuite", "rospy", "heroes_sim"}
ROOT = Path(__file__).resolve().parents[1] / "src"


def test_pure_packages_import_boundary():
    offenders = []
    for pkg in ("heroes_control", "heroes_safety"):
        for path in (ROOT / pkg).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    mods = [node.module or ""]
                else:
                    continue
                offenders += [f"{path.name}: {m}" for m in mods if m.split(".")[0] in FORBIDDEN]
    assert not offenders, offenders
