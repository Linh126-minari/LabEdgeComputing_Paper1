"""
Đường hội tụ theo episode cho các run của scripts/run_experiments.py — dùng để chọn số episode.

Metric chung (không phụ thuộc cách định hình reward): objective = cost_energy + cost_service
+ cost_security (cents/task), kèm tỉ lệ vi phạm deadline/năng lượng, lấy trung bình trượt.

  python scripts/plot_convergence.py --runs-dir outputs/runs --out outputs/convergence.png
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(runs_dir: Path) -> pd.DataFrame:
    frames = []
    for csv in runs_dir.glob("*/*/seed*/reward_components.csv"):
        agent, domain, seed = csv.parts[-4], csv.parts[-3], csv.parts[-2]
        df = pd.read_csv(csv)
        df["objective"] = df["avg_cost_energy"] + df["avg_cost_service"] + df.get("avg_cost_security", 0.0)
        frames.append(df.assign(agent=agent, domain=domain, seed=seed))
    if not frames:
        raise SystemExit(f"Không thấy reward_components.csv trong {runs_dir}")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", default="convergence.png")
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()
    df = load(Path(args.runs_dir))
    metrics = [("objective", "objective (cents/task)"), ("deadline_violation_rate", "viol_T"),
               ("energy_violation_rate", "viol_E"), ("offload_ratio", "offload ratio")]
    domains = sorted(df.domain.unique())
    fig, axes = plt.subplots(len(metrics), len(domains), figsize=(6 * len(domains), 3.2 * len(metrics)), squeeze=False)
    for c, dom in enumerate(domains):
        for r, (col, label) in enumerate(metrics):
            ax = axes[r][c]
            for agent, g in df[df.domain == dom].groupby("agent"):
                curve = g.groupby("episode")[col].mean().rolling(args.window, min_periods=1).mean()
                ax.plot(curve.index, curve.values, label=agent)
            ax.set_title(f"{dom} — {label}")
            ax.set_xlabel("episode")
            if r == 0:
                ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    last = df[df.episode > df.episode.max() * 0.8].groupby(["domain", "agent"])[[m for m, _ in metrics]].mean()
    print("Trung bình 20% episode cuối:\n" + last.round(4).to_string())
    print(f"Đã lưu {args.out}")


if __name__ == "__main__":
    main()
