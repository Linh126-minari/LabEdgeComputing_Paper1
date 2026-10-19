"""
Evaluate Heuristic Baseline Policies (All-Local, All-MEC, All-AUX, Greedy, Random)
on both Real (Rio Bus test trace) and Synthetic environments.
Matches exactly the evaluation protocol of scripts/sim2real_study.py.
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig
from src.baselines import (
    AllLocalPolicy,
    AllMECPolicy,
    AllAUXPolicy,
    RandomPolicy,
    GreedyPolicy,
)


def create_env_config(cfg) -> EnvironmentConfig:
    D = EnvironmentConfig()
    env, sec = cfg.env, cfg.get("security", {})

    def g(path: str, field: str):
        node = env
        for key in path.split("."):
            node = node.get(key, None) if node is not None and hasattr(node, "get") else None
        return getattr(D, field) if node is None else node

    enc = sec.get("encryption", {}) if sec else {}
    hp = cfg.agent.get("hyperparameters", {})
    return EnvironmentConfig(
        num_users=g("num_users", "num_users"),
        num_mec_servers=g("num_mec_servers", "num_mec_servers"),
        num_auxiliary_vehicles=g("num_auxiliary_vehicles", "num_auxiliary_vehicles"),
        num_attack_vehicles=g("num_attack_vehicles", "num_attack_vehicles"),
        sim_area_x=g("sim_area_x", "sim_area_x"), sim_area_y=g("sim_area_y", "sim_area_y"),
        task_arrival_rate=g("task_arrival_rate", "task_arrival_rate"),
        task_size_min=g("task_size_min", "task_size_min"), task_size_max=g("task_size_max", "task_size_max"),
        task_cycles_per_bit=g("task_cycles_per_bit", "task_cycles_per_bit"),
        task_deadline=g("task_deadline", "task_deadline"),
        mec_cpu_frequency=g("mec_cpu_frequency", "mec_cpu_frequency"),
        aux_cpu_frequency=g("aux_cpu_frequency", "aux_cpu_frequency"),
        user_cpu_frequency=g("user_cpu_frequency", "user_cpu_frequency"),
        bandwidth_mec=g("channel.bandwidth_mec", "bandwidth_mec"),
        bandwidth_v2v=g("channel.bandwidth_v2v", "bandwidth_v2v"),
        noise_power=g("channel.noise_power", "noise_power"),
        channel_ref_gain=g("channel.channel_gain", "channel_ref_gain"),
        channel_ref_distance=g("channel.ref_distance", "channel_ref_distance"),
        path_loss_exp=g("channel.path_loss_exp", "path_loss_exp"),
        user_tx_power=g("users.tx_power", "user_tx_power"),
        user_idle_power=g("users.idle_power", "user_idle_power"),
        velocity_min=g("users.velocity_min", "velocity_min"), velocity_max=g("users.velocity_max", "velocity_max"),
        velocity_mean=g("users.velocity_mean", "velocity_mean"), velocity_std=g("users.velocity_std", "velocity_std"),
        mec_coverage_radius=g("mec_server.coverage_radius", "mec_coverage_radius"),
        curb_distance=g("mec_server.curb_distance", "curb_distance"),
        v2v_range=g("v2v_range", "v2v_range"),
        mec_bandwidth_pool=g("mec_server.bandwidth_pool", "mec_bandwidth_pool"),
        aux_bandwidth_pool=g("aux_vehicles.bandwidth_pool", "aux_bandwidth_pool"),
        mec_base_price=g("mec_server.base_price", "mec_base_price"),
        aux_base_price=g("aux_vehicles.base_price", "aux_base_price"),
        mec_credibility=g("mec_server.credibility", "mec_credibility"),
        aux_credibility=g("aux_vehicles.credibility", "aux_credibility"),
        price_energy_local=g("users.price_energy_local", "price_energy_local"),
        price_energy_mec=g("mec_server.price_energy_mec", "price_energy_mec"),
        price_energy_av=g("aux_vehicles.price_energy_av", "price_energy_av"),
        leakage_price=g("constraints.leakage_price", "leakage_price"),
        energy_budget_max=g("constraints.energy_budget_max", "energy_budget_max"),
        penalty_mode="fixed",
        deadline_penalty=g("constraints.deadline_penalty", "deadline_penalty"),
        energy_violation_penalty=g("constraints.energy_violation_penalty", "energy_violation_penalty"),
        deadline_floor=g("deadline.floor", "deadline_floor"),
        encryption_enabled=enc.get("enabled", D.encryption_enabled),
        encryption_mode=enc.get("mode", D.encryption_mode),
        he_cycles_per_bit=enc.get("homomorphic", {}).get("cycles_per_bit", D.he_cycles_per_bit),
        block_cipher_cycles_per_bit=enc.get("block_cipher", {}).get("cycles_per_bit", D.block_cipher_cycles_per_bit),
        block_cipher_round_factor=enc.get("block_cipher", {}).get("round_factor", D.block_cipher_round_factor),
        security_level=enc.get("security_level", D.security_level),
        crypto_scheme=enc.get("scheme", D.crypto_scheme),
        attacker_model=g("attack.model", "attacker_model"),
        attacker_density=g("attack.density", "attacker_density"),
        frame_length_bits=g("attack.frame_length_bits", "frame_length_bits"),
        encryption_block_length_bits=g("attack.encryption_block_length_bits", "encryption_block_length_bits"),
        lane_count=g("attack.lane_count", "lane_count"),
        road_max_speed_mps=g("attack.road_max_speed_mps", "road_max_speed_mps"),
        security_range_v2i=g("attack.security_range_v2i", "security_range_v2i"),
        security_range_v2v=g("attack.security_range_v2v", "security_range_v2v"),
        use_4d_action=hp.get("use_4d_action", D.use_4d_action),
        mobility_type=g("mobility.type", "mobility_type"),
        trace_file=g("mobility.trace_file", "trace_file"),
        trace_window_id=g("mobility.trace_window_id", "trace_window_id"),
        background_users_enabled=g("background.users_enabled", "background_users_enabled"),
        background_alpha_l=g("background.alpha_l", "background_alpha_l"),
        background_resource_level=g("background.resource_level", "background_resource_level"),
        background_load_enabled=g("background.load_enabled", "background_load_enabled"),
        alpha_levels=hp.get("alpha_levels", hp.get("action_levels", D.alpha_levels)),
        action_levels=hp.get("action_levels", D.action_levels),
        dt=g("dt", "dt"),
        episode_length=g("episode_length", "episode_length"),
    )

DOMAINS = {
    "real": [
        "env=scenario_riobus",
        "env.mobility.trace_file=data/processed/riobus/riobus_test.parquet",
    ],
    "synthetic": [
        "env=scenario_default",
    ],
}

POLICIES = {
    "all_local": lambda seed: AllLocalPolicy(),
    "all_mec": lambda seed: AllMECPolicy(),
    "all_aux": lambda seed: AllAUXPolicy(),
    "greedy": lambda seed: GreedyPolicy(),
    "random": lambda seed: RandomPolicy(seed=seed),
}


def compose_cfg(overrides):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name="config", overrides=overrides)


def run_episode(policy, env, max_steps):
    obs, info = env.reset()
    for step in range(max_steps):
        action = policy.select_action(obs, env=env, training=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = next_obs
        if terminated or truncated:
            break
    m = env.get_metrics()
    return {
        "objective_cents": m.get("avg_cost", 0.0),
        "cost_energy": m.get("avg_cost_energy", 0.0),
        "cost_service": m.get("avg_cost_service", 0.0),
        "cost_security": m.get("avg_cost_security", 0.0),
        "offload_ratio": m.get("offload_ratio", 0.0),
        "aux_share": m.get("aux_share", 0.0),
        "mec_share": m.get("mec_share", 0.0),
        "block_length_aux": m.get("avg_block_length_aux", 0.0),
        "secure_capacity_bps": m.get("avg_secure_capacity", 0.0),
        "avg_delay_s": m.get("avg_delay", 0.0),
        "viol_T": m.get("deadline_violation_rate", 0.0),
        "viol_E": m.get("energy_violation_rate", 0.0),
        "link_loss": m.get("link_loss_failure_rate", 0.0),
        "avg_tau_eff": m.get("avg_tau_eff", 0.0),
    }


def main():
    seeds = [0, 1, 2]
    n_episodes = 10
    out_dir = ROOT / "train" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ĐÁNH GIÁ 5 HEURISTIC BASELINES TRÊN TEST DATASET")
    print(f"Policies: {list(POLICIES.keys())}")
    print(f"Domains:  {list(DOMAINS.keys())}")
    print(f"Seeds:    {seeds} | Episodes/seed: {n_episodes}")
    print("=" * 70)

    records = []

    for pol_name, pol_fn in POLICIES.items():
        print(f"\n>>> Đang đánh giá chính sách: {pol_name.upper()}...")
        for dom_name, overrides in DOMAINS.items():
            cfg = compose_cfg(overrides + ["agent=dueling_dqn"])
            base_env = VehicularEdgeEnvironment(create_env_config(cfg))
            max_steps = base_env.config.episode_length

            for seed in seeds:
                policy = pol_fn(seed)
                for ep in range(n_episodes):
                    np.random.seed(10_000 + 100 * seed + ep)
                    res = run_episode(policy, base_env, max_steps)
                    records.append({
                        "policy": pol_name,
                        "test_domain": dom_name,
                        "seed": seed,
                        "episode": ep,
                        **res,
                    })

    df_raw = pd.DataFrame(records)
    df_raw.to_csv(out_dir / "heuristic_raw.csv", index=False)

    metrics = [c for c in df_raw.columns if c not in ("policy", "test_domain", "seed", "episode")]
    per_seed = df_raw.groupby(["policy", "test_domain", "seed"])[metrics].mean()
    g = per_seed.groupby(["policy", "test_domain"])[metrics]
    summary = g.mean().add_suffix("_mean").join(
        (1.96 * g.std(ddof=1) / np.sqrt(g.count())).fillna(0.0).add_suffix("_ci95")
    )
    summary.to_csv(out_dir / "heuristic_summary.csv")

    print("\n" + "=" * 90)
    print("TỔNG HỢP HIỆU NĂNG CÁC CHÍNH SÁCH HEURISTIC (MEAN TRÊN SEEDS)")
    print("=" * 90)
    cols = ["objective_cents_mean", "cost_energy_mean", "cost_service_mean", "cost_security_mean", "viol_T_mean", "viol_E_mean"]
    print(summary[cols].to_string())

    # So sánh trực tiếp trên Real Test Set
    print("\n" + "=" * 90)
    print("BẢNG ĐỐI ĐẦU TRỰC TIẾP TRÊN MIỀN REAL (RIO BUS TEST SET):")
    print("=" * 90)
    
    # Nạp kết quả DRL để so sánh
    try:
        lag_df = pd.read_csv(out_dir / "transfer_matrix_lagrangian_dqn.csv")
        due_df = pd.read_csv(out_dir / "transfer_matrix_dueling_dqn.csv")
        lag_rr = lag_df[(lag_df.train == "real") & (lag_df.test == "real")].iloc[0]
        due_rr = due_df[(due_df.train == "real") & (due_df.test == "real")].iloc[0]
        
        print(f"{'Thuật toán / Chính sách':<25} | {'Chi phí C_total (cents)':<23} | {'Vi phạm Deadline':<18} | {'Vi phạm Năng lượng'}")
        print("-" * 90)
        print(f"{'PID-LagrangianDQN (Đề xuất)':<25} | {lag_rr['objective_cents_mean']:.5f} +/- {lag_rr['objective_cents_ci95']:.5f}  | {lag_rr['viol_T_mean']*100:6.2f}%            | {lag_rr['viol_E_mean']*100:6.2f}%")
        print(f"{'DuelingDQN (Baseline [A])':<25} | {due_rr['objective_cents_mean']:.5f} +/- {due_rr['objective_cents_ci95']:.5f}  | {due_rr['viol_T_mean']*100:6.2f}%            | {due_rr['viol_E_mean']*100:6.2f}%")
        
        for pol_name in POLICIES.keys():
            row = summary.loc[(pol_name, "real")]
            print(f"{pol_name.upper():<25} | {row['objective_cents_mean']:.5f} +/- {row['objective_cents_ci95']:.5f}  | {row['viol_T_mean']*100:6.2f}%            | {row['viol_E_mean']*100:6.2f}%")
    except Exception as e:
        print(f"Lỗi nạp DRL results: {e}")

    print("\n✓ Đã xuất kết quả hoàn tất vào:", out_dir / "heuristic_summary.csv")


if __name__ == "__main__":
    main()
