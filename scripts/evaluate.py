"""
Evaluation Harness for Vehicular Edge Computing (Phase 2 - T2.6).

Runs multi-policy, multi-seed evaluation on identical environment task arrival
and vehicle mobility trajectories. Produces raw CSV results and summary table with 95% CIs.

Usage:
    python scripts/evaluate.py --episodes 20 --seeds 5
"""
import sys
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig
from src.baselines import (
    AllLocalPolicy,
    AllMECPolicy,
    AllAUXPolicy,
    RandomPolicy,
    GreedyPolicy,
    PaperAAlgorithm1Policy,
    CMSOPolicy,
)
from src.utils.metrics import MetricsCollector
from src.utils.logger import Logger


def get_policy(name: str, seed: int):
    name_lower = name.lower()
    if name_lower == "all_local":
        return AllLocalPolicy()
    elif name_lower == "all_mec":
        return AllMECPolicy()
    elif name_lower == "all_aux":
        return AllAUXPolicy()
    elif name_lower == "random":
        return RandomPolicy(seed=seed)
    elif name_lower == "greedy":
        return GreedyPolicy()
    elif name_lower == "paper_a_alg1":
        return PaperAAlgorithm1Policy()
    elif name_lower == "cmso":
        return CMSOPolicy()
    else:
        raise ValueError(f"Unknown policy name: {name}")


def evaluate_policy_on_seed(policy_name: str, env_seed: int, num_episodes: int) -> list:
    """Evaluate a single policy on a specific environment seed over num_episodes."""
    cfg = EnvironmentConfig(episode_length=200)
    env = VehicularEdgeEnvironment(cfg)
    policy = get_policy(policy_name, seed=env_seed)
    
    episode_results = []

    for ep in range(num_episodes):
        # Env reset seed ensures identical trajectory for all policies on same (env_seed, ep)
        ep_seed = env_seed * 10000 + ep
        obs, info = env.reset(seed=ep_seed)
        collector = MetricsCollector()

        for step in range(cfg.episode_length):
            action = policy.select_action(obs, env=env, training=False)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            collector.record_step(reward, step_info)
            obs = next_obs
            if terminated or truncated:
                break

        summary = collector.compute_summary()
        summary["policy"] = policy_name
        summary["env_seed"] = env_seed
        summary["episode"] = ep
        episode_results.append(summary)

    return episode_results


def compute_confidence_interval(data: list) -> tuple:
    """Compute mean and 95% confidence interval margin."""
    arr = np.array(data, dtype=np.float64)
    if len(arr) <= 1:
        return float(np.mean(arr)), 0.0
    mean = float(np.mean(arr))
    sem = stats.sem(arr)
    margin = float(sem * stats.t.ppf((1 + 0.95) / 2.0, len(arr) - 1))
    return mean, margin


def main():
    parser = argparse.ArgumentParser(description="Multi-policy, multi-seed evaluation harness")
    parser.add_argument("--policies", type=str,
                        default="all_local,all_mec,all_aux,random,greedy,paper_a_alg1,cmso",
                        help="Comma-separated policy names")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per seed")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    logger = Logger("EvaluationHarness", log_dir=args.output_dir)
    logger.info(f"Starting evaluation: policies={args.policies}, seeds={args.seeds}, episodes={args.episodes}")

    policy_list = [p.strip() for p in args.policies.split(",") if p.strip()]
    raw_results = []

    for policy_name in policy_list:
        logger.info(f"Evaluating policy: {policy_name}...")
        for seed_idx in range(args.seeds):
            env_seed = 42 + seed_idx
            res = evaluate_policy_on_seed(policy_name, env_seed, args.episodes)
            raw_results.extend(res)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save raw results CSV
    raw_csv = output_path / "eval_raw_results.csv"
    if raw_results:
        fieldnames = list(raw_results[0].keys())
        with open(raw_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw_results)
        logger.info(f"Raw results saved to {raw_csv}")

    # Compute summary table with 95% CIs
    metrics_to_aggregate = [
        "total_reward", "avg_cost", "avg_delay", "p95_delay",
        "avg_energy", "avg_sec_capacity", "deadline_violation_rate", "link_loss_failure_rate"
    ]

    summary_rows = []
    for policy_name in policy_list:
        p_data = [r for r in raw_results if r["policy"] == policy_name]
        row = {"policy": policy_name, "num_samples": len(p_data)}
        for m in metrics_to_aggregate:
            vals = [r[m] for r in p_data]
            mean, ci = compute_confidence_interval(vals)
            row[f"{m}_mean"] = mean
            row[f"{m}_ci95"] = ci
        summary_rows.append(row)

    summary_csv = output_path / "eval_summary.csv"
    if summary_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Summary table saved to {summary_csv}")

    logger.info("=" * 80)
    logger.info("Evaluation Completed Successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
