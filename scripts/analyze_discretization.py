"""
Action Space Discretization Sensitivity Analysis (Phase 3 - Task T3.5).

Evaluates discretization granularities alpha_levels in {5, 10, 20, 50} against
continuous CMSO (Convex Optimization Algorithm 3 Paper [A]) solutions to quantify
discretization error and justify discrete action space choices.
"""
import sys
import argparse
import csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig
from src.baselines.cmso import CMSOPolicy
from src.utils.logger import Logger


def main():
    parser = argparse.ArgumentParser(description="Action Discretization Analysis")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    logger = Logger("DiscretizationAnalysis")
    logger.info("Starting Action Space Discretization Sensitivity Analysis...")

    levels_list = [5, 10, 20, 50]
    cmso_policy = CMSOPolicy()
    results = []

    for levels in levels_list:
        cfg = EnvironmentConfig(alpha_levels=levels, episode_length=50, num_auxiliary_vehicles=15, v2v_range=400.0)
        env = VehicularEdgeEnvironment(cfg)
        obs, _ = env.reset(seed=42)

        errors = []
        costs = []

        for _ in range(50):
            user = env.users[0]
            if env.task_queue[user.id] and env._aux_feasible(user)[0]:
                aux_ok, aux_idx = env._aux_feasible(user)
                aux = env.auxiliary_vehicles[aux_idx]
                task = env.task_queue[user.id][0]
                tx_rate = env.channel_model.calc_rate(
                    user.get_distance(aux), user.tx_power, env.config.bandwidth_v2v
                )
                
                # Calculate bounds
                cycles_total = task.size * task.cycles_per_bit
                tau = max(task.max_delay, env.config.deadline_floor)
                alpha_l_ub = (tau * user.cpu_frequency) / max(cycles_total, 1.0)
                alpha_l_lb = 1.0 - (tau * tx_rate) / max(task.size, 1.0)
                alpha_min = max(0.0, float(alpha_l_lb))
                alpha_max = min(1.0, float(alpha_l_ub))

                # Continuous optimal alpha
                if alpha_min > alpha_max:
                    alpha_opt = 1.0
                else:
                    alpha_opt = cmso_policy._solve_convex(env, user, aux, task, tx_rate, alpha_min, alpha_max)
                
                # Discrete alpha
                disc_level = int(np.clip(round(alpha_opt * (levels - 1)), 0, levels - 1))
                alpha_disc = disc_level / float(max(levels - 1, 1))

                error = abs(alpha_opt - alpha_disc)
                errors.append(error)

            action = cmso_policy.select_action(obs, env=env)
            obs, r, term, trunc, info = env.step(action)
            if "task_cost" in info:
                costs.append(info["task_cost"])
            if term or trunc:
                break

        results.append({
            "alpha_levels": levels,
            "mean_discretization_error": float(np.mean(errors)) if errors else 0.0,
            "max_discretization_error": float(np.max(errors)) if errors else 0.0,
            "avg_cost": float(np.mean(costs)) if costs else 0.0,
        })

        logger.info(f"Levels={levels:2d}: Mean Error={results[-1]['mean_discretization_error']:.6f}, Avg Cost={results[-1]['avg_cost']:.4f}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_file = output_path / "discretization_analysis.csv"

    fieldnames = list(results[0].keys())
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Discretization analysis saved to {csv_file}")
    logger.info("=" * 80)
    logger.info("Discretization Analysis Completed Successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
