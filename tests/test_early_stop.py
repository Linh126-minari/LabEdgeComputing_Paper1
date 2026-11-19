"""
Unit tests for Phase 2: Inference Early Stopping Threshold
Ref: [P3] Xing Chen et al., TMC 2025, Sec IV-C (Decision termination threshold tau = 0.03)
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig
from src.agents.policy import DRLAgent


def test_evaluate_episode_default_backward_compatible():
    """Mặc định early_stop_threshold=0.0 thì không dừng sớm và chạy đủ số bước."""
    cfg = EnvironmentConfig(num_users=2, num_mec_servers=1, num_auxiliary_vehicles=1, episode_length=30)
    env = VehicularEdgeEnvironment(cfg)
    agent = DRLAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.nvec)

    res = agent.evaluate_episode(env, max_steps=30, early_stop_threshold=0.0)
    assert "early_stopped" in res
    assert res["early_stopped"] is False
    assert res["steps"] == 30


def test_evaluate_episode_early_stopping_triggered():
    """Khi action ổn định và early_stop_threshold > 0, evaluate_episode phải dừng sớm và set early_stopped=True."""
    cfg = EnvironmentConfig(num_users=2, num_mec_servers=1, num_auxiliary_vehicles=1, episode_length=50)
    env = VehicularEdgeEnvironment(cfg)
    agent = DRLAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.nvec)

    # Mock select_action để luôn trả về 1 action cố định
    fixed_action = np.array([5, 0, 3, 1], dtype=np.int64) if len(env.action_space.nvec) == 4 else np.array([5, 0, 3], dtype=np.int64)
    agent.select_action = lambda state, training=False, mask=None: fixed_action.copy()

    patience = 10
    res = agent.evaluate_episode(env, max_steps=50, early_stop_threshold=0.03, early_stop_patience=patience)
    
    assert res["early_stopped"] is True
    # Bắt đầu so sánh từ step thứ 2 (khi có prev_action), sau đó cần `patience` bước liên tiếp bằng nhau
    # Do đó tổng số bước chạy sẽ là 1 + patience = 11, nhỏ hơn nhiều so với max_steps=50
    assert res["steps"] == 1 + patience
    assert res["steps"] < 50


def test_evaluate_episode_no_early_stop_when_actions_fluctuate():
    """Khi các action liên tục biến động vượt quá threshold, không kích hoạt early stopping."""
    cfg = EnvironmentConfig(num_users=2, num_mec_servers=1, num_auxiliary_vehicles=1, episode_length=20)
    env = VehicularEdgeEnvironment(cfg)
    agent = DRLAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.nvec)

    # Mock select_action xen kẽ 2 action khác nhau xa (delta > 0.03)
    a1 = np.zeros(len(env.action_space.nvec), dtype=np.int64)
    a2 = np.ones(len(env.action_space.nvec), dtype=np.int64)
    state_toggle = {"flag": False}

    def alternating_action(state, training=False, mask=None):
        state_toggle["flag"] = not state_toggle["flag"]
        return a1.copy() if state_toggle["flag"] else a2.copy()

    agent.select_action = alternating_action

    res = agent.evaluate_episode(env, max_steps=20, early_stop_threshold=0.03, early_stop_patience=5)
    assert res["early_stopped"] is False
    assert res["steps"] == 20
