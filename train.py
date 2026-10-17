"""
Training Script for Vehicular Edge Computing DRL Agent
=======================================================
Main training loop using Hydra configuration management.
"""

import sys
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from pathlib import Path
import json

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig
from src.agents.policy import DRLAgent
from src.agents.constrained_policy import LagrangianDRLAgent
from src.agents.maddpg import MADDPGAgent
from src.env.multi_agent import MultiAgentVehicularEdgeEnv
from src.utils.logger import Logger


logger = Logger("TrainingScript")


def create_env_config(cfg: DictConfig) -> EnvironmentConfig:
    """
    EnvironmentConfig từ Hydra config. Giá trị thiếu trong cfg → default của EnvironmentConfig
    (đã đồng bộ với configs/params_master.yaml bằng tests/test_params_master.py) — không hardcode
    số lần thứ hai ở đây.
    """
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
        # Channel (T0.5)
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
        # Coverage / pools
        mec_coverage_radius=g("mec_server.coverage_radius", "mec_coverage_radius"),
        curb_distance=g("mec_server.curb_distance", "curb_distance"),
        v2v_range=g("v2v_range", "v2v_range"),
        mec_bandwidth_pool=g("mec_server.bandwidth_pool", "mec_bandwidth_pool"),
        aux_bandwidth_pool=g("aux_vehicles.bandwidth_pool", "aux_bandwidth_pool"),
        # Pricing ([A] §3.5, Eq(22))
        mec_base_price=g("mec_server.base_price", "mec_base_price"),
        aux_base_price=g("aux_vehicles.base_price", "aux_base_price"),
        mec_credibility=g("mec_server.credibility", "mec_credibility"),
        aux_credibility=g("aux_vehicles.credibility", "aux_credibility"),
        price_energy_local=g("users.price_energy_local", "price_energy_local"),
        price_energy_mec=g("mec_server.price_energy_mec", "price_energy_mec"),
        price_energy_av=g("aux_vehicles.price_energy_av", "price_energy_av"),
        # Objective & constraints
        leakage_price=g("constraints.leakage_price", "leakage_price"),
        energy_budget_max=g("constraints.energy_budget_max", "energy_budget_max"),
        penalty_mode=("external" if cfg.agent.get("name", "") == "LagrangianDQN"
                      else g("constraints.penalty_mode", "penalty_mode")),
        deadline_penalty=g("constraints.deadline_penalty", "deadline_penalty"),
        energy_violation_penalty=g("constraints.energy_violation_penalty", "energy_violation_penalty"),
        deadline_floor=g("deadline.floor", "deadline_floor"),
        # Security — phân vai theo đích
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
        # Mobility (Phase 4)
        mobility_type=g("mobility.type", "mobility_type"),
        trace_file=g("mobility.trace_file", "trace_file"),
        trace_window_id=g("mobility.trace_window_id", "trace_window_id"),
        # Background users (T1.5)
        background_users_enabled=g("background.users_enabled", "background_users_enabled"),
        background_alpha_l=g("background.alpha_l", "background_alpha_l"),
        background_resource_level=g("background.resource_level", "background_resource_level"),
        background_load_enabled=g("background.load_enabled", "background_load_enabled"),
        alpha_levels=hp.get("alpha_levels", hp.get("action_levels", D.alpha_levels)),
        action_levels=hp.get("action_levels", D.action_levels),
        dt=g("dt", "dt"),
        episode_length=g("episode_length", "episode_length"),
    )


def build_agent(cfg: DictConfig, env: VehicularEdgeEnvironment, env_config: EnvironmentConfig):
    """Tạo agent theo cfg.agent.name. Trả (agent, train_env) — MADDPG dùng env đa agent."""
    hp = cfg.agent.hyperparameters
    num_episodes_cfg = cfg.general.get("num_episodes", 100)
    common_kwargs = dict(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.nvec,
        learning_rate=hp.learning_rate,
        gamma=hp.gamma,
        epsilon_start=hp.epsilon_start,
        epsilon_end=hp.epsilon_end,
        epsilon_decay=hp.epsilon_decay,
        buffer_size=hp.buffer_size,
        batch_size=hp.batch_size,
        target_update_freq=hp.target_update_freq,
        # T0.11
        learning_starts=hp.get("learning_starts", 2000),
        use_double_dqn=hp.get("use_double_dqn", True),
        use_soft_update=hp.get("use_soft_update", False),
        tau=hp.get("tau", 0.005),
        epsilon_schedule=hp.get("epsilon_schedule", "linear"),
        exploration_fraction=hp.get("exploration_fraction", 0.3),
        total_training_steps=num_episodes_cfg * env_config.episode_length,
        base_dt=env_config.dt,  # T1.4: semi-MDP discount
        device=cfg.general.device if (cfg.general.device != "cuda" or __import__("torch").cuda.is_available()) else "cpu",
    )
    train_env = env  # env agent tương tác; metrics luôn đọc từ metrics_env
    if cfg.agent.get("name", "") == "MADDPG":
        # MADDPG thật (src/agents/maddpg.py): mọi user là agent trong env đa agent,
        # huấn luyện tập trung / thực thi phân tán. Không còn background user Algorithm 1.
        train_env = MultiAgentVehicularEdgeEnv(env_config, reward_mode=hp.get("reward_mode", "individual"),
                                               xi=hp.get("fairness_xi", 0.5))
        agent = MADDPGAgent(
            obs_dim=env.observation_space.shape[0], action_dim=env.action_space.nvec,
            n_agents=train_env.num_users,
            actor_lr=hp.get("actor_lr", 1e-4), critic_lr=hp.get("critic_lr", 1e-3),
            gamma=hp.gamma, tau=hp.get("tau", 0.01), batch_size=hp.batch_size, buffer_size=hp.buffer_size,
            learning_starts=hp.get("learning_starts", 2000), hidden_size=hp.get("hidden_size", 256),
            gumbel_temperature=hp.get("gumbel_temperature", 1.0),
            epsilon_start=hp.epsilon_start, epsilon_end=hp.epsilon_end,
            exploration_fraction=hp.get("exploration_fraction", 0.3),
            total_training_steps=num_episodes_cfg * env_config.episode_length,
            base_dt=env_config.dt, device=common_kwargs["device"], train_freq=hp.get("train_freq", 1),
        )
    elif cfg.agent.get("name", "") == "LagrangianDQN":
        # T1.3: env đã ở penalty_mode="external" (đặt trong create_env_config)
        agent = LagrangianDRLAgent(
            **common_kwargs,
            lambda_T_init=hp.get("lambda_T_init", 1.0),
            lambda_E_init=hp.get("lambda_E_init", 1.0),
            lambda_lr=hp.get("lambda_lr", 0.05),
            lambda_max=hp.get("lambda_max", 100.0),
            delta_T=hp.get("delta_T", 0.05),
            delta_E=hp.get("delta_E", 0.0),
            multiplier_update_interval=hp.get("multiplier_update_interval", 1),
            use_pid=hp.get("use_pid", True),
            pid_kp=hp.get("pid_kp", 0.1),
            pid_ki=hp.get("pid_ki", None),
            pid_kd=hp.get("pid_kd", 0.02),
        )
    else:
        agent = DRLAgent(**common_kwargs)
    return agent, train_env


def train(cfg: DictConfig):
    """
    Main training function.
    
    Args:
        cfg: Hydra DictConfig with configuration
    """
    logger.info("=" * 80)
    logger.info("Starting training for Vehicular Edge Computing DRL Agent")
    logger.info("=" * 80)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # ==================== Setup ====================
    proj = Path(cfg.general.project_name)
    output_dir = proj if proj.is_absolute() else Path(cfg.paths.logs) / proj
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed (numpy + torch để các seed tái lập được)
    np.random.seed(cfg.general.random_seed)
    __import__("torch").manual_seed(cfg.general.random_seed)
    
    # ==================== Create Environment ====================
    logger.info("Creating environment...")
    env_config = create_env_config(cfg)
    env = VehicularEdgeEnvironment(env_config)

    # T0.10: assert nhất quán state/action dim với công thức trong bài
    J = env_config.num_mec_servers
    N = env_config.num_auxiliary_vehicles
    expected_state_dim = 5 + 2 * (J + N)
    assert env.observation_space.shape[0] == expected_state_dim, (
        f"State dim không khớp công thức trong bài: "
        f"{env.observation_space.shape[0]} != 5 + 2({J}+{N}) = {expected_state_dim}"
    )
    expected_action_dims = 4 if env_config.use_4d_action else 3
    assert len(env.action_space.nvec) == expected_action_dims, (
        f"Action phải có {expected_action_dims} chiều [α_l, target, β(, N_i)], "
        f"nhận được {len(env.action_space.nvec)}"
    )
    logger.info(f"Environment created: state_dim={env.observation_space.shape[0]} "
                f"(formula: 5 + 2(J+N) = 5 + 2({J}+{N})), "
                f"action=MultiDiscrete {env.action_space.nvec.tolist()}")

    # ==================== Create Agent ====================
    logger.info("Creating DRL agent...")
    agent, train_env = build_agent(cfg, env, env_config)
    logger.info("DRL agent created successfully")
    
    # ==================== Training Loop ====================
    num_episodes = cfg.general.get("num_episodes", 100)
    eval_interval = cfg.general.get("eval_interval", 10)
    log_interval = cfg.general.get("log_interval", 5)
    
    training_history = []
    eval_history = []
    component_history = []  # T1.2 bước 5: log riêng từng thành phần reward

    logger.info(f"Starting training for {num_episodes} episodes...")

    for episode in range(1, num_episodes + 1):
        # Train episode
        stats = agent.train_episode(train_env, max_steps=env_config.episode_length)
        training_history.append(stats)

        # T1.2: thành phần reward trung bình của episode (từ env metrics)
        metrics_env = train_env.env if isinstance(train_env, MultiAgentVehicularEdgeEnv) else train_env
        ep_metrics = metrics_env.get_metrics()
        component_history.append({
            "episode": episode,
            "avg_cost_energy": ep_metrics.get("avg_cost_energy", 0.0),
            "avg_cost_service": ep_metrics.get("avg_cost_service", 0.0),
            "avg_cost_security": ep_metrics.get("avg_cost_security", 0.0),
            "avg_secure_capacity": ep_metrics.get("avg_secure_capacity", 0.0),
            "avg_block_length_aux": ep_metrics.get("avg_block_length_aux", 0.0),
            "offload_ratio": ep_metrics.get("offload_ratio", 0.0),
            "deadline_violation_rate": ep_metrics.get("deadline_violation_rate", 0.0),
            "energy_violation_rate": ep_metrics.get("energy_violation_rate", 0.0),
            "link_loss_failure_rate": ep_metrics.get("link_loss_failure_rate", 0.0),
            "avg_tau_eff": ep_metrics.get("avg_tau_eff", 0.0),
            "avg_eps_n": ep_metrics.get("avg_eps_n", 0.0),
            "mec_utilization": ep_metrics.get("mec_utilization", 0.0),
        })
        
        # Log training stats
        if episode % log_interval == 0:
            logger.info(f"Episode {episode:4d} | "
                       f"Reward: {stats['total_reward']:10.4f} | "
                       f"Avg Loss: {stats['avg_loss']:10.6f} | "
                       f"Epsilon: {stats['epsilon']:8.6f}")
        
        # Evaluate periodically
        if episode % eval_interval == 0:
            logger.info(f"Evaluating at episode {episode}...")
            eval_rewards = []
            eval_delays = []
            eval_violations = []
            
            for _ in range(cfg.general.get("eval_episodes", 5)):
                eval_stats = agent.evaluate_episode(train_env, max_steps=env_config.episode_length)
                eval_rewards.append(eval_stats.get("episode_reward", 0.0))
                eval_delays.append(eval_stats.get("avg_delay", 0.0))
                eval_violations.append(eval_stats.get("deadline_violation_rate", 0.0))
            
            eval_result = {
                "episode": episode,
                "avg_reward": np.mean(eval_rewards),
                "avg_delay": np.mean(eval_delays),
                "deadline_violation_rate": np.mean(eval_violations)
            }
            eval_history.append(eval_result)
            
            logger.info(f"Eval Results: Reward={eval_result['avg_reward']:.4f}, "
                       f"Delay={eval_result['avg_delay']:.6f}s, "
                       f"Violation_Rate={eval_result['deadline_violation_rate']:.2%}")
    
    # ==================== Save Results ====================
    logger.info("Saving results...")
    
    # Save model
    model_path = output_dir / "agent_model.pth"
    agent.save_model(str(model_path))
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            "training": training_history,
            "evaluation": eval_history
        }, f, indent=2)

    # T1.2 bước 5: CSV thành phần reward theo episode
    if component_history:
        import csv
        comp_path = output_dir / "reward_components.csv"
        with open(comp_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(component_history[0].keys()))
            writer.writeheader()
            writer.writerows(component_history)
        logger.info(f"Reward components saved to {comp_path}")

    # T1.3 bước 4: lịch sử hội tụ multiplier (hình đẹp cho bài)
    if hasattr(agent, "lambda_history") and agent.lambda_history:
        with open(output_dir / "lambda_history.json", 'w', encoding='utf-8') as f:
            json.dump(agent.lambda_history, f, indent=2)
        logger.info("Lagrangian multiplier history saved")
    
    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
    
    return agent, training_history, eval_history


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point with Hydra config management.
    
    Args:
        cfg: Hydra DictConfig
    """
    train(cfg)


if __name__ == "__main__":
    main()
