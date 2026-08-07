"""
3-Model Comparison Script: LagrangianDQN vs DuelingDQN vs MADDPG
================================================================
Reads output directories of the 3 models, computes statistical comparison metrics,
exports summary CSV, and generates 300 DPI publication plots.
"""

import sys
import json
import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.3,
})


def compare_models(base_output_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = {
        "LagrangianDQN": base_output_dir / "LagrangianDQN",
        "DuelingDQN": base_output_dir / "DuelingDQN",
        "MADDPG": base_output_dir / "MADDPG",
    }

    summary_rows = []
    model_data = {}

    for model_name, path in model_dirs.items():
        if not path.exists():
            # Fallback if outputs/CSO-Sim-Framework/ is used for LagrangianDQN
            fallback = base_output_dir / "CSO-Sim-Framework"
            if model_name == "LagrangianDQN" and fallback.exists():
                path = fallback
            else:
                print(f"Warning: Directory for {model_name} not found at {path}")
                continue

        hist_file = path / "training_history.json"
        comp_file = path / "reward_components.csv"

        if not hist_file.exists() or not comp_file.exists():
            print(f"Warning: Missing output files in {path}")
            continue

        with open(hist_file, "r") as f:
            history = json.load(f)

        comp_df = pd.read_csv(comp_file)
        train_df = pd.DataFrame(history["training"])
        eval_df = pd.DataFrame(history["evaluation"])

        # Compute summary over last 50 episodes
        last_50_comp = comp_df.tail(50)
        last_50_train = train_df.tail(50)

        row = {
            "model": model_name,
            "mean_reward": float(last_50_train["total_reward"].mean()),
            "std_reward": float(last_50_train["total_reward"].std()),
            "avg_delay_s": float(eval_df.tail(5)["avg_delay"].mean()) if not eval_df.empty else 0.0,
            "deadline_violation_rate": float(last_50_comp["deadline_violation_rate"].mean()),
            "energy_violation_rate": float(last_50_comp["energy_violation_rate"].mean()),
            "avg_cost_energy": float(last_50_comp["avg_cost_energy"].mean()),
            "avg_cost_service": float(last_50_comp["avg_cost_service"].mean()),
            "avg_secure_capacity": float(last_50_comp["avg_secure_capacity"].mean()),
        }
        summary_rows.append(row)

        window = 20
        train_df["reward_smooth"] = train_df["total_reward"].rolling(window=window, min_periods=1).mean()
        comp_df["viol_T_smooth"] = comp_df["deadline_violation_rate"].rolling(window=window, min_periods=1).mean()

        model_data[model_name] = {
            "train": train_df,
            "eval": eval_df,
            "comp": comp_df,
        }

    # Save summary CSV
    summary_csv = output_dir / "model_comparison_summary.csv"
    if summary_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved 3-Model Comparison Summary to {summary_csv}")

    # Plot Comparison Figure
    if model_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        colors = {"LagrangianDQN": "#1f77b4", "DuelingDQN": "#ff7f0e", "MADDPG": "#2ca02c"}

        for m_name, m_dict in model_data.items():
            c = colors.get(m_name, "#7f7f7f")
            # Plot reward
            ax1.plot(m_dict["train"]["episode"], m_dict["train"]["reward_smooth"], label=m_name, color=c)
            # Plot violation rate
            ax2.plot(m_dict["comp"]["episode"], m_dict["comp"]["viol_T_smooth"] * 100, label=m_name, color=c)

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward (cents)")
        ax1.set_title("Training Reward Convergence (3 Models)")
        ax1.legend(loc="lower right")
        ax1.grid(True)

        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Deadline Violation Rate (%)")
        ax2.axhline(y=5.0, color="red", linestyle="--", label="Target Threshold (δ_T = 5%)")
        ax2.set_title("Deadline Constraint Satisfaction (3 Models)")
        ax2.legend(loc="upper right")
        ax2.grid(True)

        fig.tight_layout()
        plot_file = output_dir / "plots" / "fig_3model_comparison.png"
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_file, dpi=300)
        plt.close(fig)
        print(f"Saved 3-Model Comparison Plot to {plot_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare 3 DRL Models")
    parser.add_argument("--base_output_dir", type=str, default="outputs", help="Base outputs directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    compare_models(Path(args.base_output_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
