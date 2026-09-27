"""Episode-level sanity checks (M1: open loop, volitional only)."""

import numpy as np
import pandas as pd
import pytest

from heroes_sim import metrics
from heroes_sim.config import load_run_config
from heroes_sim.recorder import write_run
from heroes_sim.runner import run

J = "r_elbow_flex"
SCI = "configs/patients/sci_c5.yaml"


def cfg_for(name, patient=None, **overrides):
    return load_run_config(
        f"configs/scenarios/{name}.yaml", overrides=overrides, patient_path=patient
    )


def test_healthy_reaches_step_targets():
    cfg = cfg_for("step_targets", **{"scenario.duration_s": 7.0})
    log = run(cfg, seed=0).log
    for t_end, tgt in ((3.95, 1.2), (6.95, 1.8)):  # end of each 3 s hold
        q = log.loc[(log.t - t_end).abs().idxmin(), f"q_{J}"]
        assert q == pytest.approx(tgt, abs=0.05)


def test_no_intent_stays_near_rest():
    log = run(cfg_for("no_intent", **{"scenario.duration_s": 5.0}), seed=0).log
    assert metrics.max_excursion(log, J) < 0.1
    assert log[f"target_{J}"].isna().all()


def test_mvc_torque_signs_and_impairment():
    out = {}
    for patient in (None, SCI):
        res = run(cfg_for("mvc_calibration", patient, **{"scenario.rest_s": 0.5}), seed=0)
        out[patient] = {r.group: r for r in res.mvc}
        assert res.log[f"q_{J}"].eq(res.log[f"q_{J}"].iloc[0]).all()  # isometric
    healthy, sci = out[None], out[SCI]
    assert healthy["elbow_flexors"].torque[0] > 30 and healthy["elbow_extensors"].torque[0] < -20
    assert healthy["elbow_flexors"].activation[:3].min() > 0.9
    assert 0 < sci["elbow_flexors"].torque[0] < healthy["elbow_flexors"].torque[0]
    assert sci["elbow_extensors"].torque[0] > healthy["elbow_extensors"].torque[0]


def test_same_seed_identical_parquet(tmp_path):
    cfg = cfg_for("no_intent", **{"scenario.duration_s": 1.0})
    paths = [
        write_run(tmp_path / str(i), run(cfg, seed=s).log, cfg, s) for i, s in enumerate((3, 3, 4))
    ]
    blobs = [(p / "log.parquet").read_bytes() for p in paths]
    assert blobs[0] == blobs[1]
    a, c = (pd.read_parquet(p / "log.parquet") for p in (paths[0], paths[2]))
    assert not np.array_equal(a["u_vol_BIClong"], c["u_vol_BIClong"])
