"""
T0.12 / Phase 1 — Bất biến của môi trường (env invariants).

Sau T1.4 env theo Gymnasium API:
    reset() -> (state, info);  step() -> (state, reward, terminated, truncated, info)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig
from src.env.entities import Task


def make_env(**overrides) -> VehicularEdgeEnvironment:
    np.random.seed(42)
    cfg = EnvironmentConfig(episode_length=1000, task_arrival_rate=5.0, **overrides)
    env = VehicularEdgeEnvironment(cfg)
    env.reset(seed=42)
    return env


@pytest.fixture
def env():
    return make_env()


def test_alpha_constraints_on_random_actions(env):
    """Ràng buộc [A] 23b-23d trên 1000 action ngẫu nhiên."""
    rng = np.random.default_rng(0)
    for _ in range(1000):
        action = np.array([rng.integers(0, 10), rng.integers(0, 2), rng.integers(0, 10)])
        alpha_l, alpha_j, alpha_n, _, _ = env._decode_action_to_alphas(action)
        assert abs(alpha_l + alpha_j + alpha_n - 1.0) < 1e-9
        assert alpha_j * alpha_n == pytest.approx(0.0)
        assert 0.0 <= alpha_l <= 1.0


def test_state_dim_formula(env):
    """State dim = 5 + 2(J+N) (thêm nᵉ); action 4 chiều [α_l, đích, β, N_i]."""
    j = len(env.mec_servers)
    n = len(env.auxiliary_vehicles)
    assert env.observation_space.shape[0] == 5 + 2 * (j + n)
    assert len(env.action_space.nvec) == (4 if env.config.use_4d_action else 3)


def test_state_in_unit_range(env):
    state, _ = env.reset(seed=1)
    assert state.shape == env.observation_space.shape
    for _ in range(200):
        action = env.action_space.sample()
        state, _, terminated, truncated, _ = env.step(action)
        assert np.all(state >= 0.0) and np.all(state <= 1.0), state
        if terminated or truncated:
            state, _ = env.reset()


def test_step_outputs_positive_and_bounded(env):
    for _ in range(200):
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        assert info["task_delay"] > 0.0
        assert info["task_energy"] > 0.0
        assert info["task_cost"] >= 0.0
        assert 0.0 <= info["task_security_risk"] <= 1.0
        assert info["task_secure_capacity"] >= 0.0
        assert np.isfinite(reward)
        if terminated or truncated:
            env.reset()


def test_security_metrics_in_unit_interval(env):
    """p_att ∈ [0,1] cho nhiều bán kính vùng phát."""
    for radius in [50, 100, 150, 200, 300]:
        p_att, u = env._compute_security_metrics(
            d_sender_range=radius, link_rate_bps=5e6)
        assert 0.0 <= p_att <= 1.0
        assert u >= 0.0


def test_attack_probability_calibrated(env):
    """
    T1.7 DoD: p_att có phương sai khác 0 (V2I ≠ V2V) và trung bình ∈ [0.1, 0.6].

    Tính tay: V2I d=200 → n^e=1.34 → p_att=0.605; V2V d=150 → n^e=0.75 → p_att=0.407.
    """
    p_v2i, _ = env._compute_security_metrics(
        d_sender_range=env.config.security_range_v2i, link_rate_bps=5e6)
    p_v2v, _ = env._compute_security_metrics(
        d_sender_range=env.config.security_range_v2v, link_rate_bps=5e6)
    assert abs(p_v2i - 0.605) < 0.02
    assert abs(p_v2v - 0.407) < 0.02
    assert p_v2i != pytest.approx(p_v2v)  # phương sai khác 0
    assert 0.1 <= (p_v2i + p_v2v) / 2 <= 0.6


def test_rate_monotone_decreasing_in_distance(env):
    rates = [env.channel_model.calc_rate(d, 0.2, 1e6)
             for d in [10, 50, 100, 200, 300, 500, 800]]
    assert all(r1 > r2 for r1, r2 in zip(rates, rates[1:]))


def test_no_nan_inf_long_run():
    """Không NaN/inf sau 10 000 quyết định; không transition lỗi nào (T1.4 DoD)."""
    env = make_env()
    state, _ = env.reset(seed=7)
    for i in range(10_000):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        assert "error" not in info, f"transition lỗi tại step {i} — T1.4 cấm"
        assert np.all(np.isfinite(state)), f"state NaN/inf at step {i}"
        assert np.isfinite(reward), f"reward NaN/inf at step {i}"
        assert np.isfinite(info["task_delay"]) and np.isfinite(info["task_energy"])
        if terminated or truncated:
            state, _ = env.reset()


def test_offload_energy_exceeds_tx_plus_enc(env):
    """T0.6 DoD: với alpha_l=0 (offload hết), E_total > E_tx + E_enc vì có E^w."""
    mec = env.mec_servers[0]
    user = env.users[0]
    found = False
    for _ in range(200):
        user.position = np.array(mec.position, dtype=float)
        _, _, terminated, truncated, info = env.step(np.array([0, 0, 9]))
        if info.get("offload_location") == "MEC_0":
            found = True
            assert info["task_energy_wait"] > 0.0
            assert info["task_energy"] > (
                info["task_energy_compute_tx"] + info["task_energy_enc"])
            break
        if terminated or truncated:
            env.reset()
    assert found, "Không có bước nào offload thành công sang MEC"


def test_energy_components_sum_to_total(env):
    """Bất biến: E_total = E_compute_tx + E_enc + E_wait."""
    for _ in range(100):
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        total = (info["task_energy_compute_tx"] + info["task_energy_enc"]
                 + info["task_energy_wait"])
        assert abs(total - info["task_energy"]) < 1e-9
        if terminated or truncated:
            env.reset()
