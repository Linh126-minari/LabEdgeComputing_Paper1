"""
Sim-to-Real Transfer & Gap Study (Phase 4 - T4.3, làm lại)
==========================================================
1. Khoảng cách phân phối: KS vận tốc & thời gian lưu trú MEC — synthetic (Gauss cắt cụt
   theo [B] Eq(1)) vs GPS thật Rio bus (riobus_train, chỉ điểm active).
2. Ma trận chuyển giao 2×2 THẬT: huấn luyện agent trên mỗi miền
   (synthetic = env/scenario_default, real = env/scenario_riobus → riobus_train.parquet),
   đánh giá greedy trên cả synthetic lẫn riobus_test.parquet (ngày khác, không rò rỉ), nhiều seed.

Metric so sánh là metric CHUNG, không phụ thuộc cách định hình reward khi huấn luyện:
  objective = cost_energy + cost_service + cost_security   (cents/task, càng nhỏ càng tốt)
kèm tỉ lệ vi phạm deadline/năng lượng. (Reward huấn luyện của agent Lagrangian không trừ
penalty còn agent penalty cố định thì có — không được so reward huấn luyện giữa hai loại.)

Chạy:
  python scripts/sim2real_study.py --agent lagrangian_dqn --episodes 500 --seeds 0 1 2
  python scripts/sim2real_study.py --episodes 5 --episode-length 100 --eval-episodes 2   # smoke
  python scripts/sim2real_study.py --runs-dir outputs/runs --agent maddpg --seeds 0 1 2   # chỉ đánh giá model
      đã train bằng scripts/run_experiments.py (không train lại)
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.env.core import VehicularEdgeEnvironment
from src.utils.logger import Logger
from train import train, create_env_config, build_agent

logger = Logger("Sim2RealStudy")

TRAIN_DOMAINS = {"synthetic": ["env=scenario_default"],
                 "real": ["env=scenario_riobus",
                          "env.mobility.trace_file=data/processed/riobus/riobus_train.parquet"]}
TEST_DOMAINS = {"synthetic": ["env=scenario_default"],
                "real": ["env=scenario_riobus",
                         "env.mobility.trace_file=data/processed/riobus/riobus_test.parquet"]}


def compose_cfg(overrides):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name="config", overrides=overrides)


# ------------------------------------------------------------------ 1. distribution gap

def synthetic_velocities(n: int, seed: int = 0) -> np.ndarray:
    """Vận tốc synthetic của env — [B] Eq(1): Gauss cắt cụt (μ, σ, v_min, v_max) từ EnvironmentConfig."""
    from src.env.core import EnvironmentConfig
    c = EnvironmentConfig()
    rng = np.random.default_rng(seed)
    out = np.empty(0)
    while out.size < n:
        v = rng.normal(c.velocity_mean, c.velocity_std, size=2 * n)
        out = np.concatenate([out, v[(v >= c.velocity_min) & (v <= c.velocity_max)]])
    return out[:n]


def synthetic_residence_times(n: int = 2000, radius: float = 300.0, seed: int = 0):
    """Lượt đi thẳng qua vùng phủ MEC bán kính `radius`: dây cung ngẫu nhiên / vận tốc synthetic."""
    rng = np.random.default_rng(seed)
    offset = rng.uniform(-np.pi / 2, np.pi / 2, size=n)      # góc đi vào so với pháp tuyến
    chord = 2 * radius * np.abs(np.cos(offset))
    return chord / synthetic_velocities(n, seed + 1)


def distribution_gap(stats_json: Path, train_parquet: Path) -> dict:
    import json
    real = pd.read_parquet(train_parquet)
    v_real = real.loc[real.active == 1, "v_mps"].to_numpy()
    v_syn = synthetic_velocities(len(v_real))
    ks_v = stats.ks_2samp(v_syn, v_real)

    from scripts.preprocess_riobus import residence_times
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/data/riobus.yaml").read_text())
    res_real = residence_times(real, cfg)
    res_syn = synthetic_residence_times()
    ks_r = stats.ks_2samp(res_syn, res_real)
    return {
        "velocity": {"ks": ks_v.statistic, "p": ks_v.pvalue,
                     "synthetic_mean": float(v_syn.mean()), "real_mean": float(v_real.mean())},
        "mec_residence_s": {"ks": ks_r.statistic, "p": ks_r.pvalue,
                            "synthetic_median": float(np.median(res_syn)), "real_median": float(np.median(res_real))},
    }


# ------------------------------------------------------------------ 2. transfer matrix

def evaluate(agent, env: VehicularEdgeEnvironment, n_episodes: int, max_steps: int, seed: int):
    rows = []
    for ep in range(n_episodes):
        np.random.seed(10_000 + 100 * seed + ep)     # cùng chuỗi đánh giá cho mọi agent
        m = agent.evaluate_episode(env, max_steps=max_steps)
        rows.append({
            "objective_cents": m["avg_cost"],   # = năng lượng + dịch vụ + rủi ro rò rỉ (cùng hàm mục tiêu env)
            "cost_energy": m["avg_cost_energy"], "cost_service": m["avg_cost_service"],
            "cost_security": m["avg_cost_security"], "offload_ratio": m["offload_ratio"],
            "aux_share": m["aux_share"], "mec_share": m["mec_share"], "block_length_aux": m["avg_block_length_aux"],
            "secure_capacity_bps": m["avg_secure_capacity"], "avg_delay_s": m["avg_delay"],
            "viol_T": m["deadline_violation_rate"], "viol_E": m["energy_violation_rate"],
            "link_loss": m["link_loss_failure_rate"], "avg_tau_eff": m["avg_tau_eff"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Sim-to-Real transfer study")
    ap.add_argument("--agent", default="lagrangian_dqn")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--episode-length", type=int, default=None, help="mặc định theo params_master")
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output_dir", default="outputs/sim2real")
    ap.add_argument("--runs-dir", default=None, help="thư mục của run_experiments.py → nạp model thay vì train")
    args = ap.parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gap = distribution_gap(ROOT / "data/processed/riobus/riobus_stats.json",
                           ROOT / "data/processed/riobus/riobus_train.parquet")
    pd.json_normalize(gap).to_csv(out_dir / "distribution_gap.csv", index=False)
    logger.info(f"Distribution gap: {gap}")

    common = [f"agent={args.agent}", f"general.num_episodes={args.episodes}", f"general.device={args.device}",
              "general.eval_interval=1000000", "general.log_interval=10"]
    if args.episode_length:
        common.append(f"env.episode_length={args.episode_length}")

    records = []
    for seed in args.seeds:
        for train_dom, tr_over in TRAIN_DOMAINS.items():
            cfg = compose_cfg(tr_over + common + [f"general.random_seed={seed}",
                                                  f"general.project_name=sim2real/{args.agent}/{train_dom}_seed{seed}"])
            if args.runs_dir:
                model = ROOT / args.runs_dir / args.agent / train_dom / f"seed{seed}" / "agent_model.pth"
                if not model.exists():
                    logger.warning(f"Thiếu {model} — bỏ qua")
                    continue
                base_env = VehicularEdgeEnvironment(create_env_config(cfg))
                agent, _ = build_agent(cfg, base_env, base_env.config)
                agent.load_model(str(model))
            else:
                logger.info(f"=== Train {args.agent} on {train_dom}, seed {seed} ===")
                agent, _, _ = train(cfg)
            for test_dom, te_over in TEST_DOMAINS.items():
                te_cfg = compose_cfg(te_over + common + [f"general.random_seed={seed}"])
                base_env = VehicularEdgeEnvironment(create_env_config(te_cfg))
                _, env = build_agent(te_cfg, base_env, base_env.config)   # MADDPG → env đa agent
                for r in evaluate(agent, env, args.eval_episodes, base_env.config.episode_length, seed):
                    records.append({"agent": args.agent, "seed": seed, "train": train_dom, "test": test_dom, **r})
                pd.DataFrame(records).to_csv(out_dir / f"transfer_raw_{args.agent}.csv", index=False)

    raw = pd.DataFrame(records)
    metrics = [c for c in raw.columns if c not in ("agent", "seed", "train", "test")]
    per_seed = raw.groupby(["train", "test", "seed"])[metrics].mean()
    g = per_seed.groupby(["train", "test"])[metrics]
    summary = g.mean().add_suffix("_mean").join(
        (1.96 * g.std(ddof=1) / np.sqrt(g.count())).fillna(0.0).add_suffix("_ci95"))
    summary.to_csv(out_dir / f"transfer_matrix_{args.agent}.csv")
    logger.info("Transfer matrix (mean over seeds):\n" +
                g.mean()[["objective_cents", "viol_T", "viol_E", "offload_ratio", "aux_share", "block_length_aux"]].to_string())


if __name__ == "__main__":
    main()
