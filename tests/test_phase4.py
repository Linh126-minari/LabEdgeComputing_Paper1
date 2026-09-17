"""
Phase 4 Unit & Integration Tests — Rio bus GPS preprocessing, TraceMobility, trace env.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig
from src.env.mobility import TraceMobility
from scripts.preprocess_riobus import clean_raw, extract_window, project_local, haversine_m

CFG = {
    "bbox": {"lat_min": -23.1, "lat_max": -22.7, "lon_min": -43.8, "lon_max": -43.1},
    "max_recorded_speed_kmh": 120.0, "max_implied_speed_kmh": 120.0,
    "max_trip_gap_s": 180.0, "min_trip_points": 2, "exclude_dates": ["2019-03-04"],
    "window_duration_s": 120.0, "out_dt": 1.0, "max_interp_gap_s": 180.0,
    "min_segment_s": 30.0, "keep_radius_m": 1000.0, "clip_radius_m": 1500.0,
    "anchor_xy": [250.0, 250.0], "num_slots": 2, "min_window_coverage": 0.5,
}
LAT0, LON0 = -22.9, -43.2


def _raw(order, date, hms, lats, lons, speeds):
    return pd.DataFrame({
        "date": date, "time": [f"2025-11-25T{t}.000Z" for t in hms], "order": order, "line": "775.0",
        "latitude": [str(v) for v in lats], "longitude": [str(v) for v in lons], "speed": [str(v) for v in speeds],
    })


def test_clean_fixes_timestamp_from_date_column():
    """`time` mang ngày giả 2025-11-25; ngày thật phải lấy từ `date`."""
    df = _raw("A1", "02-12-2019", ["08:00:00", "08:01:00"], [LAT0, LAT0 + 0.001], [LON0, LON0], [20, 20])
    clean, _ = clean_raw(df, CFG)
    assert clean.ts.dt.date.unique().tolist() == [pd.Timestamp("2019-02-12").date()]
    assert clean.line.iloc[0] == "775"


def test_clean_drops_duplicates_jumps_excluded_dates_and_splits_trips():
    lats = [LAT0, LAT0, LAT0 + 0.0005, LAT0 + 0.2, LAT0 + 0.001, LAT0 + 0.0015]   # +0.2° ≈ 22 km trong 60 s
    df = pd.concat([
        _raw("A1", "02-12-2019", ["08:00:00", "08:00:00", "08:01:00", "08:02:00", "08:03:00", "08:30:00"],
             lats, [LON0] * 6, [10] * 6),
        _raw("B1", "03-04-2019", ["08:00:00", "08:01:00"], [LAT0] * 2, [LON0] * 2, [10, 10]),
    ])
    clean, rep = clean_raw(df, CFG)
    assert rep["dropped_duplicates"] == 1
    assert rep["dropped_gps_jumps"] >= 1
    assert rep["dropped_excluded_dates"] == 2
    assert (clean.lat < LAT0 + 0.1).all()
    # khoảng trống 27 phút → tách chuyến; điểm lẻ 08:30 bị loại vì chuyến < min_trip_points
    assert clean.trip_id.nunique() == 1 and len(clean) == 3


def test_projection_matches_haversine():
    x, y = project_local(LAT0 + 0.01, LON0 + 0.01, LAT0, LON0)
    assert np.hypot(x, y) == pytest.approx(haversine_m(LAT0, LON0, LAT0 + 0.01, LON0 + 0.01), rel=1e-3)


def test_extract_window_shared_clock_and_slot_handover():
    """Hai xe đồng thời → 2 slot cùng đồng hồ; xe thứ ba đến sau nhận slot trống (active)."""
    base = pd.Timestamp("2019-02-12 08:00:00")
    rows = []
    for trip, (order, t_start, t_end, dlat) in enumerate(
            [("A", -10, 70, 0.0), ("B", -10, 130, 0.001), ("C", 75, 130, 0.0)]):
        for k, t in enumerate(np.arange(t_start, t_end + 1, 20)):
            rows.append({"order": order, "line": "1", "trip_id": trip, "speed_kmh": 20.0,
                         "ts": base + pd.Timedelta(seconds=float(t)), "lat": LAT0 + dlat + 1e-5 * k, "lon": LON0})
    day = pd.DataFrame(rows)
    w = extract_window(day, base, (LAT0, LON0), CFG, "w")
    assert w.vehicle_id.nunique() == 2
    assert w.groupby("vehicle_id").t_rel.apply(lambda s: s.tolist()).map(tuple).nunique() == 1
    slot_a = w[w.bus_id.isin(["A", "C"])].vehicle_id.unique()
    assert len(slot_a) == 1                          # C thay A trong cùng slot
    s = w[w.vehicle_id == slot_a[0]]
    assert (s.active == 0).any() and (s.active == 1).mean() > 0.8
    assert (s[s.active == 0].v_mps == 0).all()
    assert w.x_m.iloc[0] == pytest.approx(250.0, abs=1.0)   # hotspot ↦ anchor


def test_trace_mobility_time_interpolation():
    """Trace lưu 1 s, mô phỏng dt = 0.1 s → vị trí nội suy theo thời gian."""
    df = pd.DataFrame({"vehicle_id": "v", "t_rel": [0.0, 1.0, 2.0, 3.0], "x_m": [100.0, 110.0, 120.0, 120.0],
                       "y_m": [50.0] * 4, "v_mps": [10.0] * 4, "active": [1, 1, 0, 0]})
    mob = TraceMobility(df, vehicle_id="v", loop=True)
    pos, v = mob.generate_initial_state()
    assert np.allclose(pos, [100.0, 50.0]) and v == pytest.approx(10.0)
    for _ in range(5):
        pos = mob.step(pos, v, dt=0.1)
    assert np.allclose(pos, [105.0, 50.0])
    assert mob.current_direction == pytest.approx(0.0)
    assert mob.is_active
    for _ in range(20):                               # t = 2.5 s
        pos = mob.step(pos, v, dt=0.1)
    assert not mob.is_active


def test_trace_mobility_legacy_format():
    df = pd.DataFrame({"vehicle_id": "veh_0", "t_rel": [0.0, 0.1, 0.2], "x_m": [100.0, 105.0, 110.0],
                       "y_m": [50.0] * 3, "v_mps": [15.0] * 3})
    mob = TraceMobility(df, vehicle_id="veh_0")
    pos, v = mob.generate_initial_state()
    assert np.allclose(mob.step(pos, v, dt=0.1), [105.0, 50.0])


def _window_file(tmp_path):
    rows = []
    for wid, x0 in [("w1", 100.0), ("w2", 600.0)]:
        for k in range(15):
            t = np.arange(0.0, 200.0, 1.0)
            rows.append(pd.DataFrame({"window_id": wid, "vehicle_id": f"slot{k:02d}", "bus_id": f"B{k}", "line": "1",
                                      "t_rel": t, "x_m": x0 + 5.0 * t, "y_m": 200.0 + 10 * k, "v_mps": 5.0, "active": 1}))
    p = tmp_path / "windows.parquet"
    pd.concat(rows).to_parquet(p, index=False)
    return p


def test_environment_trace_reset_uses_trace_positions_and_windows(tmp_path):
    """reset() không được gán vị trí ngẫu nhiên đè lên trace; mỗi cửa sổ là một đồng hồ chung."""
    cfg = EnvironmentConfig(mobility_type="trace", trace_file=str(_window_file(tmp_path)), episode_length=5)
    env = VehicularEdgeEnvironment(cfg)
    seen = set()
    for seed in range(8):
        env.reset(seed=seed)
        x0 = {"w1": 100.0, "w2": 600.0}[env.trace_window_id]
        seen.add(env.trace_window_id)
        assert isinstance(env.users[0].mobility_model, TraceMobility)
        # vị trí khớp trace tại thời điểm hiện tại của mobility (đã tiến tới task đầu tiên)
        mob = env.users[0].mobility_model
        assert env.users[0].position[0] == pytest.approx(x0 + 5.0 * mob.t, abs=1e-6)
    assert seen == {"w1", "w2"}

    env.step(env.action_space.sample())
    assert env.users[0].velocity == pytest.approx(5.0)


def test_environment_fixed_window(tmp_path):
    cfg = EnvironmentConfig(mobility_type="trace", trace_file=str(_window_file(tmp_path)),
                            trace_window_id="w2", episode_length=5)
    env = VehicularEdgeEnvironment(cfg)
    for seed in range(3):
        env.reset(seed=seed)
        assert env.trace_window_id == "w2"


def test_environment_with_real_riobus_trace():
    trace_path = Path("data/processed/riobus/riobus_train.parquet")
    if not trace_path.exists():
        pytest.skip("Chưa chạy scripts/preprocess_riobus.py")
    env = VehicularEdgeEnvironment(EnvironmentConfig(mobility_type="trace", trace_file=str(trace_path), episode_length=10))
    obs, info = env.reset(seed=42)
    next_obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert next_obs is not None and np.isfinite(r)


def test_inactive_trace_slots_are_not_offload_targets(tmp_path):
    """Slot không có xe thật (active=0): không offload tới AUX đó; user inactive chỉ tính local."""
    rows = []
    for k in range(15):
        t = np.arange(0.0, 200.0, 1.0)
        rows.append(pd.DataFrame({"window_id": "w", "vehicle_id": f"slot{k:02d}", "bus_id": "B", "line": "1",
                                  "t_rel": t, "x_m": 250.0 + k, "y_m": 250.0, "v_mps": 0.0,
                                  "active": 1 if k < 12 else 0}))
    p = tmp_path / "w.parquet"
    pd.concat(rows).to_parquet(p, index=False)
    env = VehicularEdgeEnvironment(EnvironmentConfig(mobility_type="trace", trace_file=str(p), episode_length=5))
    env.reset(seed=0)
    user = env.users[0]
    assert env._is_present(user)
    present_aux = [env._is_present(a) for a in env.auxiliary_vehicles]
    assert present_aux.count(False) == 1                 # 12 slot active: 10 user + 2 AUX; AUX thứ 3 inactive
    env.auxiliary_vehicles = [a for a in env.auxiliary_vehicles if not env._is_present(a)]
    env.aux_cpu_available = env.aux_cpu_available[:1]
    ok, _ = env._aux_feasible(user)
    assert not ok

    user.mobility_model.active[:] = False
    mask = env.action_mask().reshape(tuple(env.action_space.nvec))
    assert mask[-1].any() and not mask[:-1].any()
