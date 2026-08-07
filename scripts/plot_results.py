"""
Publication-Quality Result Plotter for CSO-Sim-Framework
=========================================================
Generates 4 figures summarizing training performance, constraint satisfaction,
cost & resource utilization, and Lagrangian dual multiplier convergence.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set publication style
plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.3,
})


def plot_all_results(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    with open(data_dir / "training_history.json", "r") as f:
        history = json.load(f)

    with open(data_dir / "lambda_history.json", "r") as f:
        lambdas = json.load(f)

    comp_df = pd.read_csv(data_dir / "reward_components.csv")

    train_data = pd.DataFrame(history["training"])
    eval_data = pd.DataFrame(history["evaluation"])
    lambda_df = pd.DataFrame(lambdas) if lambdas else pd.DataFrame()

    window = 20
    train_data["reward_smooth"] = train_data["total_reward"].rolling(window=window, min_periods=1).mean()
    train_data["loss_smooth"] = train_data["avg_loss"].rolling(window=window, min_periods=1).mean()
    comp_df["viol_T_smooth"] = comp_df["deadline_violation_rate"].rolling(window=window, min_periods=1).mean()

    # ---------------------------------------------------------
    # FIGURE 1: Training Reward & Loss Convergence
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Reward subplot
    ax1.plot(train_data["episode"], train_data["total_reward"], alpha=0.25, color="#1f77b4", label="Episode Reward (Raw)")
    ax1.plot(train_data["episode"], train_data["reward_smooth"], color="#1f77b4", linewidth=2.0, label="Moving Average (w=20)")
    if not eval_data.empty:
        ax1.plot(eval_data["episode"], eval_data["avg_reward"], "ro-", markersize=4, label="Evaluation Checkpoints")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward (cents)")
    ax1.set_title("Training Reward Convergence")
    ax1.legend(loc="lower right")
    ax1.grid(True)

    # Loss & Epsilon subplot
    ax2.plot(train_data["episode"], train_data["loss_smooth"], color="#d62728", linewidth=2.0, label="Loss (Moving Avg)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Average Loss", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(True)

    ax2_eps = ax2.twinx()
    ax2_eps.plot(train_data["episode"], train_data["epsilon"], "--", color="#7f7f7f", label="Epsilon Schedule")
    ax2_eps.set_ylabel("Epsilon (Exploration Rate)", color="#7f7f7f")
    ax2_eps.tick_params(axis="y", labelcolor="#7f7f7f")

    # Combine legends for twinx
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_eps.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax2.set_title("Loss & Epsilon Schedule")

    fig.tight_layout()
    fig1_path = output_dir / "fig1_training_reward_loss.png"
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig1_path}")

    # ---------------------------------------------------------
    # FIGURE 2: Constraint Satisfaction & Delay Metrics
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Deadline violation rate
    ax1.plot(comp_df["episode"], comp_df["deadline_violation_rate"] * 100, alpha=0.3, color="#ff7f0e", label="Raw Violation Rate")
    ax1.plot(comp_df["episode"], comp_df["viol_T_smooth"] * 100, color="#ff7f0e", linewidth=2.0, label="Smoothed Rate (w=20)")
    ax1.axhline(y=5.0, color="red", linestyle="--", linewidth=1.5, label="Target Constraint Threshold (δ_T = 5%)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Deadline Violation Rate (%)")
    ax1.set_title("Deadline Constraint Satisfaction")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    # Evaluation average delay
    if not eval_data.empty:
        ax2.plot(eval_data["episode"], eval_data["avg_delay"], "go-", markersize=4, label="Avg Task Delay")
        ax2.axhline(y=1.0, color="gray", linestyle=":", label="Application Deadline τ_app = 1.0s")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Average Delay (seconds)")
        ax2.set_title("Evaluation Task Delay Progression")
        ax2.legend(loc="upper right")
        ax2.grid(True)

    fig.tight_layout()
    fig2_path = output_dir / "fig2_constraint_violations.png"
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig2_path}")

    # ---------------------------------------------------------
    # FIGURE 3: Cost Components & Resource Utilization
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Cost breakdown
    energy_cost_smooth = comp_df["avg_cost_energy"].rolling(window=20, min_periods=1).mean()
    service_cost_smooth = comp_df["avg_cost_service"].rolling(window=20, min_periods=1).mean()

    ax1.plot(comp_df["episode"], energy_cost_smooth, color="#2ca02c", linewidth=2.0, label="Energy Cost (cents)")
    ax1.plot(comp_df["episode"], service_cost_smooth, color="#9467bd", linewidth=2.0, label="Offload Service Cost (cents)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cost (cents)")
    ax1.set_title("Economic Cost Component Breakdown")
    ax1.legend(loc="center right")
    ax1.grid(True)

    # Resource Utilization
    mec_util_smooth = comp_df["mec_utilization"].rolling(window=20, min_periods=1).mean()
    eps_n_smooth = comp_df["avg_eps_n"].rolling(window=20, min_periods=1).mean()

    ax2.plot(comp_df["episode"], mec_util_smooth * 100, color="#8c564b", linewidth=2.0, label="MEC CPU Utilization (%)")
    ax2.plot(comp_df["episode"], eps_n_smooth, color="#e377c2", linewidth=2.0, label="AUX Average Load (eps_n)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Utilization / Load")
    ax2.set_title("Edge Infrastructure Resource Utilization")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    fig.tight_layout()
    fig3_path = output_dir / "fig3_cost_and_resource_breakdown.png"
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig3_path}")

    # ---------------------------------------------------------
    # FIGURE 4: Lagrangian Dual Multipliers (λ_T and λ_E)
    # ---------------------------------------------------------
    if not lambda_df.empty:
        fig, ax1 = plt.subplots(figsize=(7, 4.5))

        ax1.plot(lambda_df["episode"], lambda_df["lambda_T"], color="#1f77b4", linewidth=2.0, label="λ_T (Deadline Multiplier)")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("λ_T (Deadline Multiplier)", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(lambda_df["episode"], lambda_df["lambda_E"], color="#ff7f0e", linewidth=2.0, label="λ_E (Energy Multiplier)")
        ax2.set_ylabel("λ_E (Energy Multiplier)", color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
        ax1.set_title("Lagrangian Dual Multiplier Convergence")

        fig.tight_layout()
        fig4_path = output_dir / "fig4_lagrangian_multipliers.png"
        fig.savefig(fig4_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {fig4_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot 500-episode training results")
    parser.add_argument("--data_dir", type=str, default="outputs/CSO-Sim-Framework", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="outputs/CSO-Sim-Framework/plots", help="Plot output directory")
    args = parser.parse_args()

    plot_all_results(Path(args.data_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
