"""
Phase 2 — Baseline Unit and Integration Tests.

Tests:
1. Baseline policies execution and action masking compliance.
2. All-Local policy cost structure (zero offloading price).
3. Greedy policy performance vs Random policy.
4. CMSO policy convex optimization bounds compliance.
5. GBD closed-form Eq(39) block length output range.
6. Evaluation harness integration test.
"""
import pytest
import numpy as np
from pathlib import Path

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig
from src.baselines import (
    AllLocalPolicy,
    AllMECPolicy,
    AllAUXPolicy,
    RandomPolicy,
    GreedyPolicy,
    PaperAAlgorithm1Policy,
    CMSOPolicy,
    GBDClosedFormSecurityBaseline,
)


def make_test_env():
    cfg = EnvironmentConfig(episode_length=50)
    env = VehicularEdgeEnvironment(cfg)
    env.reset(seed=42)
    return env


def test_all_local_policy():
    env = make_test_env()
    policy = AllLocalPolicy()
    obs, _ = env.reset(seed=42)
    action = policy.select_action(obs, env=env)
    assert np.array_equal(action, np.array([9, 0, 0]))

    obs, reward, term, trunc, info = env.step(action)
    assert "task_cost" in info or "cost" in info
    # All local has zero offloading server price
    assert info.get("task_cost_service", 0.0) == 0.0


def test_all_mec_and_aux_policies():
    env = make_test_env()
    mec_policy = AllMECPolicy()
    aux_policy = AllAUXPolicy()
    obs, _ = env.reset(seed=42)

    mec_action = mec_policy.select_action(obs, env=env)
    aux_action = aux_policy.select_action(obs, env=env)

    assert len(mec_action) == 3
    assert len(aux_action) == 3


def test_random_policy_respects_action_mask():
    env = make_test_env()
    policy = RandomPolicy(seed=42)
    obs, _ = env.reset(seed=42)

    for _ in range(20):
        action = policy.select_action(obs, env=env)
        mask = env.action_mask()
        idx = np.ravel_multi_index(action, (10, 2, 10))
        assert mask[idx] == True, "RandomPolicy selected a masked/infeasible action"
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            break


def test_greedy_policy_outperforms_random():
    env = make_test_env()
    greedy = GreedyPolicy()
    random = RandomPolicy(seed=42)

    greedy_costs = []
    random_costs = []

    # Run greedy
    obs, _ = env.reset(seed=100)
    for _ in range(30):
        act = greedy.select_action(obs, env=env)
        obs, r, term, trunc, info = env.step(act)
        cost_val = info.get("task_cost", info.get("cost"))
        if cost_val is not None:
            greedy_costs.append(cost_val)
        if term or trunc:
            break

    # Run random
    obs, _ = env.reset(seed=100)
    for _ in range(30):
        act = random.select_action(obs, env=env)
        obs, r, term, trunc, info = env.step(act)
        cost_val = info.get("task_cost", info.get("cost"))
        if cost_val is not None:
            random_costs.append(cost_val)
        if term or trunc:
            break

    assert np.mean(greedy_costs) <= np.mean(random_costs), "Greedy policy cost should be <= Random policy cost"


def test_paper_a_alg1_policy():
    env = make_test_env()
    policy = PaperAAlgorithm1Policy()
    obs, _ = env.reset(seed=42)
    action = policy.select_action(obs, env=env)
    assert len(action) == 3
    assert 0 <= action[0] <= 9
    assert 0 <= action[1] <= 1


def test_cmso_policy_bounds():
    env = make_test_env()
    policy = CMSOPolicy()
    obs, _ = env.reset(seed=42)
    action = policy.select_action(obs, env=env)
    assert len(action) == 3
    assert 0 <= action[0] <= 9


def test_gbd_closed_form_eq39():
    gbd = GBDClosedFormSecurityBaseline(frame_length_bits=64)
    # n^e >= 1 -> N_f = 64
    assert gbd.compute_optimal_block_length(1.2) == 64
    # n^e = 0.7 -> in [1, 64]
    n_i = gbd.compute_optimal_block_length(0.7)
    assert 1 <= n_i <= 64
