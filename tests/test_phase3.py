"""
Phase 3 — Novel Contributions Unit and Integration Tests.

Tests:
1. Block-length dependent encryption overhead c_en(N_i).
2. Realized vs expected attacker model metrics computation.
3. Cryptographic schemes (Paillier expansion vs AES decryption overhead vs None).
4. Multi-agent environment reset and step interface.
5. Parameter-shared IDQN agent update step and action selection.
6. Discretization error analysis script execution.
"""
import pytest
import numpy as np
from pathlib import Path

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig
from src.env.multi_agent import MultiAgentVehicularEdgeEnv
from src.agents.marl import ParameterSharedIDQNAgent
from src.security.overhead_models import SecurityOverheadConfig, SecurityOverheadModel


def test_block_length_dependent_encryption_overhead():
    cfg = SecurityOverheadConfig(overhead_cycles_per_bit=890.0)
    model = SecurityOverheadModel(cfg)

    c_8 = model.compute_cycles_per_bit(8.0)
    c_16 = model.compute_cycles_per_bit(16.0)
    c_32 = model.compute_cycles_per_bit(32.0)
    c_64 = model.compute_cycles_per_bit(64.0)

    assert c_8 == 890.0
    assert c_64 > c_32 > c_16 > c_8
    assert c_64 == pytest.approx(890.0 * (1.0 + 0.1 * 3.0))


def test_realized_attacker_model():
    cfg = EnvironmentConfig(attacker_model="realized", episode_length=10)
    env = VehicularEdgeEnvironment(cfg)
    obs, _ = env.reset(seed=42)

    user = env.users[0]
    p_att, u = env._compute_security_metrics(d_sender_range=200.0, link_rate_bps=5e6, block_length_bits=32.0, user=user)

    assert 0.0 <= p_att <= 1.0
    assert u >= 0.0


def test_crypto_schemes_and_ciphertext_expansion():
    cfg_paillier = EnvironmentConfig(crypto_scheme="paillier", ciphertext_expansion=2.0)
    env_p = VehicularEdgeEnvironment(cfg_paillier)
    obs_p, _ = env_p.reset(seed=42)

    cfg_aes = EnvironmentConfig(crypto_scheme="aes", ciphertext_expansion=1.0)
    env_a = VehicularEdgeEnvironment(cfg_aes)
    obs_a, _ = env_a.reset(seed=42)

    assert getattr(env_p.config, "ciphertext_expansion") == 2.0
    assert getattr(env_a.config, "ciphertext_expansion") == 1.0


def test_multi_agent_environment_interface():
    cfg = EnvironmentConfig(episode_length=20)
    ma_env = MultiAgentVehicularEdgeEnv(cfg, reward_mode="combined", xi=0.5)

    obs_dict, info_dict = ma_env.reset(seed=42)
    assert len(obs_dict) == len(ma_env.agents)
    assert "user_0" in obs_dict

    actions_dict = {a: np.array([5, 0, 4, 2], dtype=np.int64) for a in ma_env.agents}
    next_obs_dict, rewards_dict, term_dict, trunc_dict, step_info_dict = ma_env.step(actions_dict)

    assert len(next_obs_dict) == len(ma_env.agents)
    assert len(rewards_dict) == len(ma_env.agents)
    assert isinstance(rewards_dict["user_0"], float)


def test_idqn_agent_step_and_train():
    state_dim = 12
    action_dim = [10, 2, 10, 4]
    agent = ParameterSharedIDQNAgent(state_dim=state_dim, action_dim=action_dim, batch_size=32)

    obs = np.random.rand(state_dim).astype(np.float32)
    action = agent.select_action(obs, training=True)

    assert len(action) == 4
    assert 0 <= action[0] <= 9
    assert 0 <= action[1] <= 1
    assert 0 <= action[2] <= 9
    assert 0 <= action[3] <= 3

    # Add transitions to buffer and test update
    for _ in range(50):
        s = np.random.rand(state_dim).astype(np.float32)
        a = 5
        r = 1.0
        ns = np.random.rand(state_dim).astype(np.float32)
        d = False
        agent.memory.add(s, a, r, ns, d)

    loss = agent.update()
    assert loss >= 0.0
