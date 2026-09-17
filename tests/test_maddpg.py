"""MADDPG thật (src/agents/maddpg.py) + env đa agent event-driven."""
import sys
from pathlib import Path
import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env.core import EnvironmentConfig
from src.env.multi_agent import MultiAgentVehicularEdgeEnv
from src.agents.maddpg import MADDPGAgent, head_masks_from_joint

NVEC = np.array([10, 2, 10])


def test_head_masks_from_joint():
    M = np.zeros(NVEC, bool); M[-1] = True
    hm = head_masks_from_joint(M.flatten(), NVEC)
    assert hm[:10].tolist() == [False] * 9 + [True]            # chỉ local
    M[:, 1, :] = True                                          # AUX khả thi, MEC không
    hm = head_masks_from_joint(M.flatten(), NVEC)
    assert hm[:10].all() and hm[10:12].tolist() == [False, True] and hm[12:].all()


@pytest.fixture
def setup():
    torch.manual_seed(0); np.random.seed(0)
    ma = MultiAgentVehicularEdgeEnv(EnvironmentConfig(episode_length=40), reward_mode="individual")
    ag = MADDPGAgent(obs_dim=ma.observation_space.shape[0], action_dim=ma.action_space.nvec, n_agents=ma.num_users,
                     batch_size=16, learning_starts=16, total_training_steps=200, hidden_size=32)
    return ma, ag


def test_ma_env_event_driven_and_records_history(setup):
    ma, _ = setup
    assert not ma.env.config.background_users_enabled
    obs, info = ma.reset(seed=1)
    acts = {a: np.array([9, 0, 0]) for i, a in enumerate(ma.agents) if ma.has_task(i)}
    _, _, _, _, info2 = ma.step(acts)
    assert info2["__common__"]["elapsed_time"] > 0
    assert sum(info2[a]["acted"] for a in ma.agents) == len(acts) == len(ma.env.task_history)
    assert any(info2[a]["has_task"] for a in ma.agents)       # luôn dừng ở một thời điểm quyết định


def test_actions_respect_masks(setup):
    ma, ag = setup
    obs_d, info_d = ma.reset(seed=2)
    obs, masks = ag._obs_masks(ma, obs_d, info_d)
    masks[:, 10:12] = [True, False]                            # cấm AUX
    for training in (True, False):
        for _ in range(20):
            oh = ag.select_actions(obs, masks, training=training)
            assert np.all(oh[masks == False] == 0)


def test_training_updates_networks_and_returns_metrics(setup):
    ma, ag = setup
    before = [p.clone() for p in ag.actor.parameters()]
    stats = ag.train_episode(ma, max_steps=40)
    assert ag.buffer.size > ag.learning_starts
    assert any(not torch.equal(b, p) for b, p in zip(before, ag.actor.parameters()))
    assert np.isfinite(stats["total_reward"]) and "deadline_violation_rate" in stats
    ev = ag.evaluate_episode(ma, max_steps=10)
    assert "avg_cost_energy" in ev
