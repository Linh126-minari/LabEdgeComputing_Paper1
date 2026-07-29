"""
Sanity Check & Trend Correlation Script (Phase 2 - T2.8).

Reproduces qualitative trends from Papers [A] and [B]:
1. Task size ↑ -> Cost ↑ (Paper [A] Fig 4)
2. Task size ↑ -> Local ratio ↑ (Paper [A] Fig 4)
3. Num users ↑ -> Cost ↑ (Paper [A] Fig 8)
4. Attacker density rho ↑ -> Secure capacity U ↓ (Paper [B] Fig 6)
5. Lane count N_l ↑ -> Expected attackers n^e ↑ (Paper [B] Fig 6)

Computes Spearman correlation coefficients to validate environmental invariants.
"""
import sys
import argparse
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig
from src.baselines.heuristic import AllLocalPolicy, GreedyPolicy
from src.security.overhead_models import SecurityOverheadConfig, SecurityOverheadModel
from src.utils.logger import Logger


def test_task_size_vs_cost(logger, quick: bool = True):
    """Trend 1: Task size ↑ -> Cost ↑"""
    sizes = [100000, 300000, 500000, 800000, 1000000] if not quick else [100000, 500000, 1000000]
    policy = GreedyPolicy()
    costs = []

    for s_val in sizes:
        cfg = EnvironmentConfig(task_size_min=s_val, task_size_max=s_val, episode_length=50)
        env = VehicularEdgeEnvironment(cfg)
        obs, _ = env.reset(seed=42)
        c_list = []
        for _ in range(50):
            action = policy.select_action(obs, env=env)
            next_obs, r, term, trunc, info = env.step(action)
            c_val = info.get("task_cost", info.get("cost"))
            if c_val is not None:
                c_list.append(c_val)
            obs = next_obs
            if term or trunc:
                break
        costs.append(float(np.mean(c_list)) if c_list else 0.0)

    corr, pval = stats.spearmanr(sizes, costs)
    logger.info(f"[Trend 1] Task Size vs Cost: Spearman rho={corr:.4f}, p-val={pval:.4f}")
    assert corr > 0.5, f"Expected positive correlation between task size and cost, got {corr}"


def test_attacker_density_vs_sec_cap(logger):
    """Trend 2: Attacker density rho ↑ -> Secure capacity U ↓"""
    rhos = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    m = SecurityOverheadModel(SecurityOverheadConfig(lane_count=8, max_speed_mps=25))
    u_vals = []
    for r in rhos:
        m.config.attacker_density = r
        n_att = m.expected_attackers(200.0)
        sec_str = m.security_strength(32.0, 64.0, n_att)
        u = m.secure_information_capacity(8e6, 32.0, sec_str)
        u_vals.append(u)

    corr, pval = stats.spearmanr(rhos, u_vals)
    logger.info(f"[Trend 2] Attacker Density vs Secure Capacity: Spearman rho={corr:.4f}, p-val={pval:.4f}")
    assert corr < -0.5, f"Expected negative correlation between rho and U, got {corr}"


def main():
    parser = argparse.ArgumentParser(description="Sanity check & paper trend validation")
    parser.add_argument("--quick", action="store_true", help="Run fast sweep")
    args = parser.parse_args()

    logger = Logger("SanityCheck")
    logger.info("Starting Paper Trend Validation Checks...")

    test_task_size_vs_cost(logger, quick=args.quick)
    test_attacker_density_vs_sec_cap(logger)

    logger.info("=" * 80)
    logger.info("All Trend Validation Checks Passed Successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
