"""
Publication-Quality Comparison Plotter for VEC Algorithms (LagrangianDQN vs DuelingDQN)
========================================================================================
Đọc dữ liệu từ thư mục runs của scripts/run_experiments.py và thư mục eval của scripts/sim2real_study.py:
1. Fig 1: Hội tụ Objective & Vi phạm ràng buộc (Synthetic vs Rio Bus)
2. Fig 2: Động học nhân tử PID-Lagrangian (λ_T, λ_E, Integral states)
3. Fig 3: Ma trận Chuyển giao Sim-to-Real (Transfer Matrix 2x2 Bar Chart)
4. Fig 4: Phân rã Chi phí (Energy, Service, Security) & Tỷ lệ Offload
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# Style thiết lập chuẩn bài báo khoa học
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.35,
})

AGENT_COLORS = {
    "lagrangian_dqn": "#1f77b4",  # Blue (Proposed PID-Lagrangian)
    "dueling_dqn": "#ff7f0e",     # Orange (Baseline [A])
    "maddpg": "#2ca02c",          # Green
}
AGENT_LABELS = {
    "lagrangian_dqn": "PID-LagrangianDQN (Đề xuất)",
    "dueling_dqn": "DuelingDQN (Baseline [A])",
    "maddpg": "MADDPG",
}


def load_reward_components(runs_dir: Path) -> pd.DataFrame:
    frames = []
    for csv in runs_dir.glob("*/*/seed*/reward_components.csv"):
        parts = csv.parts
        agent, domain, seed = parts[-4], parts[-3], parts[-2]
        try:
            df = pd.read_csv(csv)
            df["cost_security"] = df.get("avg_cost_security", 0.0)
            df["cost_service"] = df.get("avg_cost_service", 0.0)
            df["cost_energy"] = df.get("avg_cost_energy", 0.0)
            df["objective"] = df["cost_energy"] + df["cost_service"] + df["cost_security"]
            frames.append(df.assign(agent=agent, domain=domain, seed=seed))
        except Exception as e:
            print(f"Bỏ qua lỗi file {csv}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_convergence(df: pd.DataFrame, out_path: Path, window: int = 10):
    """Biểu đồ 1: Hội tụ Objective và Tỷ lệ vi phạm Deadline/Energy."""
    if df.empty:
        print("Không có dữ liệu reward_components để vẽ hội tụ.")
        return

    domains = [d for d in ["synthetic", "real"] if d in df.domain.unique()]
    if not domains:
        domains = sorted(df.domain.unique())

    fig, axes = plt.subplots(3, len(domains), figsize=(6.5 * len(domains), 10), squeeze=False)

    for col_idx, dom in enumerate(domains):
        dom_df = df[df.domain == dom]
        dom_title = "Synthetic Mobility" if dom == "synthetic" else "Rio Bus Real GPS Trace"

        # 1. Objective Cost
        ax_obj = axes[0][col_idx]
        for agent, g in dom_df.groupby("agent"):
            curve = g.groupby("episode")["objective"].mean().rolling(window, min_periods=1).mean()
            c = AGENT_COLORS.get(agent, "#333333")
            lbl = AGENT_LABELS.get(agent, agent)
            ax_obj.plot(curve.index, curve.values, label=lbl, color=c, lw=2.2)
        ax_obj.set_title(f"{dom_title}\nTổng Chi phí Objective ($C_{{total}}$)")
        ax_obj.set_ylabel("Chi phí (cents / task)")
        ax_obj.set_xlabel("Episode")
        ax_obj.legend(loc="upper right")

        # 2. Deadline Violation Rate
        ax_viol = axes[1][col_idx]
        for agent, g in dom_df.groupby("agent"):
            curve = g.groupby("episode")["deadline_violation_rate"].mean().rolling(window, min_periods=1).mean() * 100
            c = AGENT_COLORS.get(agent, "#333333")
            lbl = AGENT_LABELS.get(agent, agent)
            ax_viol.plot(curve.index, curve.values, label=lbl, color=c, lw=2.0)
        ax_viol.axhline(5.0, color="red", linestyle="--", alpha=0.8, label="Ngưỡng ràng buộc $\delta_T = 5\%$")
        ax_viol.set_title(f"Tỷ lệ Vi phạm Hạn chót ($viol_T$)")
        ax_viol.set_ylabel("Vi phạm Deadline (%)")
        ax_viol.set_xlabel("Episode")
        ax_viol.legend(loc="upper right")

        # 3. Energy Violation Rate
        ax_e = axes[2][col_idx]
        for agent, g in dom_df.groupby("agent"):
            curve = g.groupby("episode")["energy_violation_rate"].mean().rolling(window, min_periods=1).mean()
            c = AGENT_COLORS.get(agent, "#333333")
            lbl = AGENT_LABELS.get(agent, agent)
            ax_e.plot(curve.index, curve.values, label=lbl, color=c, lw=2.0)
        ax_e.axhline(0.0, color="gray", linestyle=":", alpha=0.7)
        ax_e.set_title(f"Năng lượng Vượt ngưỡng ($viol_E$)")
        ax_e.set_ylabel("Vượt $E_{max} = 0.4$J (Joule)")
        ax_e.set_xlabel("Episode")
        ax_e.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Đã lưu biểu đồ hội tụ: {out_path}")


def plot_pid_dynamics(runs_dir: Path, out_path: Path):
    """Biểu đồ 2: Động thái nhân tử PID-Lagrangian (λ_T, λ_E, Integral)."""
    lambda_files = list(runs_dir.glob("lagrangian_dqn/*/seed*/lambda_history.json"))
    if not lambda_files:
        print("Chưa tìm thấy lambda_history.json của lagrangian_dqn.")
        return

    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(13, 5))

    for lf in lambda_files:
        domain = lf.parts[-3]
        seed = lf.parts[-2]
        label_prefix = f"{domain}-{seed}"
        try:
            with open(lf, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            if df.empty or "episode" not in df.columns:
                continue

            ls = "-" if domain == "real" else "--"
            ax_t.plot(df["episode"], df["lambda_T"], label=f"{label_prefix} ($\lambda_T$)", linestyle=ls, lw=1.8)
            if "integral_T" in df.columns:
                ax_t.plot(df["episode"], df["integral_T"], label=f"{label_prefix} ($I_T$)", linestyle=":", alpha=0.6)

            ax_e.plot(df["episode"], df["lambda_E"], label=f"{label_prefix} ($\lambda_E$)", linestyle=ls, lw=1.8)
            if "integral_E" in df.columns:
                ax_e.plot(df["episode"], df["integral_E"], label=f"{label_prefix} ($I_E$)", linestyle=":", alpha=0.6)
        except Exception as e:
            print(f"Lỗi đọc {lf}: {e}")

    ax_t.set_title("Hội tụ Nhân tử Deadline $\lambda_T$ (với PID Damping)")
    ax_t.set_xlabel("Episode")
    ax_t.set_ylabel("Giá trị $\lambda_T$")
    ax_t.legend(loc="upper right")

    ax_e.set_title("Hội tụ Nhân tử Năng lượng $\lambda_E$")
    ax_e.set_xlabel("Episode")
    ax_e.set_ylabel("Giá trị $\lambda_E$")
    ax_e.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Đã lưu biểu đồ động thái PID: {out_path}")


def plot_sim2real_transfer(eval_dir: Path, out_path: Path):
    """Biểu đồ 3: So sánh Ma trận Chuyển giao Sim-to-Real giữa các thuật toán."""
    files = list(eval_dir.glob("transfer_matrix_*.csv"))
    if not files:
        print("Chưa có transfer_matrix_*.csv trong eval_dir để vẽ biểu đồ chuyển giao.")
        return

    dfs = []
    for f in files:
        agent_name = f.stem.replace("transfer_matrix_", "")
        df = pd.read_csv(f)
        dfs.append(df.assign(agent=agent_name))
    all_df = pd.concat(dfs, ignore_index=True)

    if all_df.empty or "train" not in all_df.columns:
        return

    all_df["pair"] = all_df["train"].str.capitalize() + " $\\to$ " + all_df["test"].str.capitalize()
    pairs = all_df["pair"].unique()
    agents = all_df["agent"].unique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(pairs))
    width = 0.35

    for i, ag in enumerate(agents):
        ag_df = all_df[all_df.agent == ag]
        ag_df = ag_df.set_index("pair").reindex(pairs).reset_index()
        offset = (i - len(agents) / 2 + 0.5) * width
        c = AGENT_COLORS.get(ag, "#333333")
        lbl = AGENT_LABELS.get(ag, ag)

        # 1. Objective Cents
        obj_vals = ag_df["objective_cents_mean"]
        ax1.bar(x + offset, obj_vals, width, label=lbl, color=c, alpha=0.88, edgecolor="black", linewidth=0.8)

        # 2. Deadline Violation Rate
        viol_vals = ag_df["viol_T_mean"] * 100
        ax2.bar(x + offset, viol_vals, width, label=lbl, color=c, alpha=0.88, edgecolor="black", linewidth=0.8)

    ax1.set_title("So sánh Chi phí Trung bình ($C_{total}$)")
    ax1.set_ylabel("Chi phí (cents / task)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pairs, rotation=15)
    ax1.legend()

    ax2.set_title("So sánh Tỷ lệ Vi phạm Hạn chót ($viol_T$)")
    ax2.set_ylabel("Tỷ lệ vi phạm (%)")
    ax2.axhline(5.0, color="red", linestyle="--", label="Ngưỡng $\delta_T=5\%$")
    ax2.set_xticks(x)
    ax2.set_xticklabels(pairs, rotation=15)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Đã lưu biểu đồ Sim-to-Real: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Publication Comparison Plotter")
    ap.add_argument("--runs-dir", default="outputs/runs", help="Thư mục chứa kết quả train")
    ap.add_argument("--eval-dir", default="outputs/eval", help="Thư mục chứa kết quả sim2real eval")
    ap.add_argument("--out-dir", default="outputs/plots", help="Thư mục xuất ảnh đồ thị")
    ap.add_argument("--window", type=int, default=10, help="Cửa sổ trung bình trượt")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = ROOT / args.runs_dir
    eval_dir = ROOT / args.eval_dir

    print("=" * 70)
    print("VẼ TỔNG HỢP BIỂU ĐỒ BÁO CÁO KHOA HỌC")
    print("=" * 70)

    # 1. Biểu đồ hội tụ
    df_comp = load_reward_components(runs_dir)
    plot_convergence(df_comp, out_dir / "fig1_convergence_comparison.png", window=args.window)

    # 2. Biểu đồ động thái PID
    plot_pid_dynamics(runs_dir, out_dir / "fig2_pid_multiplier_dynamics.png")

    # 3. Biểu đồ Sim-to-Real
    plot_sim2real_transfer(eval_dir, out_dir / "fig3_sim2real_transfer.png")

    print("\nHoàn tất xuất toàn bộ biểu đồ tại:", out_dir)


if __name__ == "__main__":
    main()
