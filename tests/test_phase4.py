"""
Phase 4 Unit & Integration Tests — Dataset Preprocessing, TraceMobility, and Sim-to-Real Analysis.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig
from src.env.mobility import TraceMobility, LinearMobility
from scripts.preprocess_riobus import preprocess_single_file


def test_preprocess_single_file_unit_conversion():
    """T4.1: Speed unit conversion km/h -> m/s and timestamp cleanup."""
    raw_df = pd.DataFrame({
        "month": [1, 1, 1],
        "day": [28, 28, 28],
        "hour": [6, 6, 6],
        "minute": [5, 6, 7],
        "second": [0, 0, 0],
        "order": ["BUS_01", "BUS_01", "BUS_01"],
        "speed": [36.0, 54.0, 72.0],  # km/h -> 10 m/s, 15 m/s, 20 m/s
        "x": [10.0, 20.0, 30.0],
        "y": [5.0, 5.0, 5.0],
        "delta_time": [0, 60, 120]
    })
    tmp_file = Path("tmp_test_bus.csv")
    raw_df.to_csv(tmp_file, index=False)

    cfg = {"dt": 0.1, "max_duration_seconds": 120.0, "max_velocity_mps": 35.0, "sim_area_x": 1000.0, "sim_area_y": 1000.0}
    clean_df = preprocess_single_file(tmp_file, cfg)

    if tmp_file.exists():
        tmp_file.unlink()

    assert not clean_df.empty
    assert "v_mps" in clean_df.columns
    assert np.max(clean_df["v_mps"]) <= 25.0
    assert np.min(clean_df["v_mps"]) >= 9.0


def test_trace_mobility_interface():
    """T4.2: TraceMobility interface matches LinearMobility."""
    dummy_df = pd.DataFrame({
        "vehicle_id": ["veh_0", "veh_0", "veh_0"],
        "t_rel": [0.0, 0.1, 0.2],
        "x_m": [100.0, 105.0, 110.0],
        "y_m": [50.0, 50.0, 50.0],
        "v_mps": [15.0, 15.0, 15.0]
    })
    mob = TraceMobility(dummy_df, vehicle_id="veh_0", loop=True)

    init_pos, init_v = mob.generate_initial_state()
    assert np.allclose(init_pos, [100.0, 50.0])
    assert init_v == pytest.approx(15.0)

    next_pos = mob.step(init_pos, init_v, dt=0.1)
    assert np.allclose(next_pos, [105.0, 50.0])
    assert mob.current_velocity() == pytest.approx(15.0)


def test_environment_with_trace_mobility():
    """T4.2: VehicularEdgeEnvironment correctly initializes with mobility_type='trace'."""
    trace_path = Path("data/processed/riobus_corridor.csv")
    if not trace_path.exists():
        pytest.skip("Trace CSV file does not exist.")

    cfg = EnvironmentConfig(mobility_type="trace", trace_file=str(trace_path), episode_length=10)
    env = VehicularEdgeEnvironment(cfg)
    obs, info = env.reset(seed=42)

    assert env.config.mobility_type == "trace"
    assert isinstance(env.users[0].mobility_model, TraceMobility)

    next_obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert next_obs is not None
