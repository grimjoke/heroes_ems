import json

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from heroes_sim.config import load_run_config, load_run_file
from heroes_sim.runner import run
from heroes_sim.sweep import SweepSpec, aggregate, make_sweep, points, run_files

SPEC = {
    "name": "t",
    "scenario": "configs/scenarios/step_targets.yaml",
    "overrides": {"scenario.duration_s": 1.0, "scenario.closed_loop": False},
    "grid": {"controller.joints.r_elbow_flex.gain": [10.0, 20.0], "emg.white_noise.std_mv": [0.01]},
    "seeds": [0, 1],
}


def test_points_product_seeds_fastest():
    p = points(SweepSpec.model_validate(SPEC))
    assert [(s, pr["controller.joints.r_elbow_flex.gain"]) for s, pr in p] == [
        (0, 10.0),
        (1, 10.0),
        (0, 20.0),
        (1, 20.0),
    ]


def test_make_sweep_writes_self_contained_run_files(tmp_path):
    out = make_sweep(SweepSpec.model_validate(SPEC), tmp_path)
    files = run_files(out)
    assert [f.name for f in files] == ["0000.yaml", "0001.yaml", "0002.yaml", "0003.yaml"]
    index = pd.read_csv(out / "index.csv")
    assert list(index["seed"]) == [0, 1, 0, 1]
    cfg, seed, params = load_run_file(files[2])
    assert seed == 0 and params["controller.joints.r_elbow_flex.gain"] == 20.0
    assert cfg.controller.joints["r_elbow_flex"].gain == 20.0 and cfg.scenario.duration_s == 1.0
    assert yaml.safe_load(files[2].read_text())["config"]["patient"]["name"] == "healthy"
    with pytest.raises(FileExistsError):
        make_sweep(SweepSpec.model_validate(SPEC), tmp_path)


def test_bad_grid_value_fails_at_make_time(tmp_path):
    bad = {**SPEC, "grid": {"controller.joints.r_elbow_flex.kp": [-1.0]}}
    with pytest.raises(ValidationError):
        make_sweep(SweepSpec.model_validate(bad), tmp_path)


def test_run_file_reproduces_direct_run(tmp_path):
    out = make_sweep(SweepSpec.model_validate(SPEC), tmp_path)
    cfg_file, seed, params = load_run_file(run_files(out)[3])
    direct = load_run_config(
        "configs/scenarios/step_targets.yaml", overrides={**SPEC["overrides"], **params}
    )
    assert cfg_file == direct
    a, b = run(cfg_file, seed).log, run(direct, seed).log
    pd.testing.assert_frame_equal(a, b)


def test_aggregate_joins_and_marks_missing(tmp_path):
    out = make_sweep(SweepSpec.model_validate(SPEC), tmp_path)
    for i, rmse in ((0, 0.1), (2, 0.3)):
        d = out / "results" / f"{i:04d}"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"seed": 0, "tracking_rmse_j": rmse}))
    table = aggregate(out)
    assert list(table["done"]) == [True, False, True, False]
    assert table.loc[2, "tracking_rmse_j"] == 0.3 and np.isnan(table.loc[1, "tracking_rmse_j"])
