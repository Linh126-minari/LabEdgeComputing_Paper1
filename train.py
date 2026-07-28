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
from src.utils.logger import Logger


logger = Logger("TrainingScript")


def create_env_config(cfg: DictConfig) -> EnvironmentConfig:
    """
    Create EnvironmentConfig from Hydra config.
    
    Args:
        cfg: Hydra DictConfig
    
    Returns:
        EnvironmentConfig object
    """
    # QUY TẮC (T0.1): mọi default ở đây khớp configs/params_master.yaml.
    return EnvironmentConfig(
        num_users=cfg.env.get("num_users", 10),
        num_mec_servers=cfg.env.get("num_mec_servers", 1),
        num_auxiliary_vehicles=cfg.env.get("num_auxiliary_vehicles", 3),
        num_attack_vehicles=cfg.env.get("num_attack_vehicles", cfg.env.get("attack", {}).get("count", 2)),
        sim_area_x=cfg.env.get("sim_area_x", 1000.0),
        sim_area_y=cfg.env.get("sim_area_y", 1000.0),
        task_arrival_rate=cfg.env.get("task_arrival_rate", 0.5),
        # T0.4: task size tính bằng BITS (100-1000 kbit theo [A] §6)
        task_size_min=cfg.env.get("task_size_min", 102400.0),
        task_size_max=cfg.env.get("task_size_max", 1024000.0),
        task_cycles_per_bit=cfg.env.get("task_cycles_per_bit", 1000),
        task_deadline=cfg.env.get("task_deadline", 1.0),
        # [A] §6: f_j = 1 GHz, f_n = 0.5 GHz
        mec_cpu_frequency=cfg.env.get("mec_cpu_frequency", 1e9),
        aux_cpu_frequency=cfg.env.get("aux_cpu_frequency", 0.5e9),
        user_cpu_frequency=cfg.env.get("user_cpu_frequency", 2e9),
        # Channel (T0.5)
        bandwidth_mec=cfg.env.get("channel", {}).get("bandwidth_mec", 1.0e6),
        bandwidth_v2v=cfg.env.get("channel", {}).get("bandwidth_v2v", 0.8e6),
        noise_power=cfg.env.get("channel", {}).get("noise_power", 3.0e-13),
        channel_ref_gain=cfg.env.get("channel", {}).get("channel_gain", 4.75e-2),
        channel_ref_distance=cfg.env.get("channel", {}).get("ref_distance", 1.0),
        path_loss_exp=cfg.env.get("channel", {}).get("path_loss_exp", 4.0),
        # Power ([A] §6)
        user_tx_power=cfg.env.get("users", {}).get("tx_power", 0.2),
        user_idle_power=cfg.env.get("users", {}).get("idle_power", 0.05),
        # Coverage / pools
        mec_coverage_radius=cfg.env.get("mec_server", {}).get("coverage_radius", 300.0),
        curb_distance=cfg.env.get("mec_server", {}).get("curb_distance", 20.0),
        v2v_range=cfg.env.get("v2v_range", 200.0),
        mec_bandwidth_pool=cfg.env.get("mec_server", {}).get("bandwidth_pool", 1e7),
        aux_bandwidth_pool=cfg.env.get("aux_vehicles", {}).get("bandwidth_pool", 5e6),
        # Pricing ([A] §3.5)
        mec_base_price=cfg.env.get("mec_server", {}).get("base_price", 1.0),
        aux_base_price=cfg.env.get("aux_vehicles", {}).get("base_price", 0.5),
        mec_credibility=cfg.env.get("mec_server", {}).get("credibility", 0.9),
        aux_credibility=cfg.env.get("aux_vehicles", {}).get("credibility", 0.8),
        # Objective & constraints (T1.2 Phương án 1 + T1.3)
        varpi_1=cfg.general.get("varpi_1", 1e-5),  # [B] Table I (T0.8)
        energy_budget_max=cfg.env.get("constraints", {}).get("energy_budget_max", 0.4),
        penalty_mode=("external" if cfg.agent.get("name", "") == "LagrangianDQN"
                      else cfg.env.get("constraints", {}).get("penalty_mode", "fixed")),
        deadline_penalty=cfg.env.get("constraints", {}).get("deadline_penalty", 5.0),
        energy_violation_penalty=cfg.env.get("constraints", {}).get("energy_violation_penalty", 5.0),
        # Deadline động (T1.1)
        deadline_floor=cfg.env.get("deadline", {}).get("floor", 0.05),
        urgency_cap=cfg.env.get("deadline", {}).get("urgency_cap", 3.0),
        # Security — hệ hai lớp (T0.3)
        encryption_enabled=cfg.security.get("encryption", {}).get("enabled", True),
        encryption_mode=cfg.security.get("encryption", {}).get("mode", "simulation"),
        encryption_overhead_cycles_per_bit=cfg.security.get("encryption", {}).get("overhead_cycles_per_bit", 890.0),
        security_level=cfg.security.get("encryption", {}).get("security_level", 1.0),
        # Attacker model [B] Eq(5)-(9) (T0.2, T0.3)
        attacker_density=cfg.env.get("attack", {}).get("density", 1e-4),
        frame_length_bits=cfg.env.get("attack", {}).get("frame_length_bits", 64.0),
        encryption_block_length_bits=cfg.env.get("attack", {}).get("encryption_block_length_bits", 32.0),
        lane_count=cfg.env.get("attack", {}).get("lane_count", 8.0),
        road_max_speed_mps=cfg.env.get("attack", {}).get("road_max_speed_mps", 25.0),
        security_range_v2i=cfg.env.get("attack", {}).get("security_range_v2i", 200.0),
        security_range_v2v=cfg.env.get("attack", {}).get("security_range_v2v", 150.0),
        # Background users (T1.5)
        background_users_enabled=cfg.env.get("background", {}).get("users_enabled", True),
        background_alpha_l=cfg.env.get("background", {}).get("alpha_l", 0.5),
        background_resource_level=cfg.env.get("background", {}).get("resource_level", 4),
        background_load_enabled=cfg.env.get("background", {}).get("load_enabled", False),
        alpha_levels=cfg.agent.get("hyperparameters", {}).get(
            "alpha_levels",
            cfg.agent.get("hyperparameters", {}).get("action_levels", 10),
        ),
        action_levels=cfg.agent.get("hyperparameters", {}).get("action_levels", 10),
        dt=cfg.env.get("dt", 0.1),
        episode_length=cfg.env.get("episode_length", 1000)
    )


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
    output_dir = Path(cfg.paths.logs) / cfg.general.project_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    np.random.seed(cfg.general.random_seed)
    
    # ==================== Create Environment ====================
    logger.info("Creating environment...")
    env_config = create_env_config(cfg)
    env = VehicularEdgeEnvironment(env_config)

    # T0.10: assert nhất quán state/action dim với công thức trong bài
    J = env_config.num_mec_servers
    N = env_config.num_auxiliary_vehicles
    expected_state_dim = 4 + 2 * (J + N)
    assert env.observation_space.shape[0] == expected_state_dim, (
        f"State dim không khớp công thức trong bài: "
        f"{env.observation_space.shape[0]} != 4 + 2({J}+{N}) = {expected_state_dim}"
    )
    assert len(env.action_space.nvec) == 3, (
        f"Action phải có 3 chiều [alpha_l_level, target, resource_level], "
        f"nhận được {len(env.action_space.nvec)}"
    )
    logger.info(f"Environment created: state_dim={env.observation_space.shape[0]} "
                f"(formula: 4 + 2(J+N) = 4 + 2({J}+{N})), "
                f"action_dim=3 (MultiDiscrete {env.action_space.nvec.tolist()})")
    
    # ==================== Create Agent ====================
    logger.info("Creating DRL agent...")
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
    if cfg.agent.get("name", "") == "LagrangianDQN":
        # T1.3: env đã ở penalty_mode="external" (đặt trong create_env_config)
        agent = LagrangianDRLAgent(
            **common_kwargs,
            lambda_T_init=hp.get("lambda_T_init", 1.0),
            lambda_E_init=hp.get("lambda_E_init", 1.0),
            lambda_lr=hp.get("lambda_lr", 0.05),
            lambda_max=hp.get("lambda_max", 100.0),
            delta_T=hp.get("delta_T", 0.05),
            delta_E=hp.get("delta_E", 0.0),
            multiplier_update_interval=hp.get("multiplier_update_interval", 10),
        )
    else:
        agent = DRLAgent(**common_kwargs)
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
        stats = agent.train_episode(env, max_steps=env_config.episode_length)
        training_history.append(stats)

        # T1.2: thành phần reward trung bình của episode (từ env metrics)
        ep_metrics = env.get_metrics()
        component_history.append({
            "episode": episode,
            "avg_cost_energy": ep_metrics.get("avg_cost_energy", 0.0),
            "avg_cost_service": ep_metrics.get("avg_cost_service", 0.0),
            "avg_secure_capacity": ep_metrics.get("avg_secure_capacity", 0.0),
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
            
            for _ in range(5):  # Evaluate for 5 episodes
                eval_stats = agent.evaluate_episode(env, max_steps=env_config.episode_length)
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
