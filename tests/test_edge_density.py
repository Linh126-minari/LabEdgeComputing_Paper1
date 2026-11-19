"""
Tests for Phase 1: Edge Vehicle Density State Representation
Ref: [P3] Xing Chen et al., TMC 2025, Sec IV-B
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig


def test_state_dim_without_edge_density():
    """Tương thích ngược: Khi edge_density_in_state=False, state dim = 5 + 2*(J+N)."""
    cfg = EnvironmentConfig(edge_density_in_state=False)
    env = VehicularEdgeEnvironment(cfg)
    state, info = env.reset(seed=42)
    expected_dim = 5 + 2 * (cfg.num_mec_servers + cfg.num_auxiliary_vehicles)
    assert state.shape[0] == expected_dim
    assert env.observation_space.shape[0] == expected_dim


def test_state_dim_with_edge_density():
    """Khi edge_density_in_state=True, state dim = 5 + 2*(J+N) + J."""
    cfg = EnvironmentConfig(edge_density_in_state=True)
    env = VehicularEdgeEnvironment(cfg)
    state, info = env.reset(seed=42)
    expected_dim = 5 + 2 * (cfg.num_mec_servers + cfg.num_auxiliary_vehicles) + cfg.num_mec_servers
    assert state.shape[0] == expected_dim
    assert env.observation_space.shape[0] == expected_dim


def test_edge_density_values_bounded():
    """Mật độ rho_ej luôn nằm trong đoạn [0.0, 1.0]."""
    cfg = EnvironmentConfig(edge_density_in_state=True, num_users=20, edge_capacity=10)
    env = VehicularEdgeEnvironment(cfg)
    for seed in [10, 20, 30, 42]:
        env.reset(seed=seed)
        densities = env._compute_edge_density()
        assert len(densities) == cfg.num_mec_servers
        for rho in densities:
            assert 0.0 <= rho <= 1.0


def test_step_maintains_consistent_state_with_density():
    """Đảm bảo chu kỳ step() chạy mượt mà, trả về state đúng kích thước khi có edge density."""
    cfg = EnvironmentConfig(edge_density_in_state=True, episode_length=20)
    env = VehicularEdgeEnvironment(cfg)
    state, info = env.reset(seed=42)
    expected_dim = 5 + 2 * (cfg.num_mec_servers + cfg.num_auxiliary_vehicles) + cfg.num_mec_servers

    for _ in range(10):
        action = env.action_space.sample()
        next_state, reward, term, trunc, info = env.step(action)
        assert next_state.shape[0] == expected_dim
        assert np.all(np.isfinite(next_state))
        if term or trunc:
            break
