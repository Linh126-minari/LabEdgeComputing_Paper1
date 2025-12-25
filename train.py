"""
Training Script for Vehicular Edge Computing DRL Agent
=======================================================
Main training loop using Hydra configuration management.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from pathlib import Path
import json

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig
from src.agents.policy import DRLAgent
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
    return EnvironmentConfig(
        num_users=cfg.env.get("num_users", 5),
        num_mec_servers=cfg.env.get("num_mec_servers", 2),
        num_auxiliary_vehicles=cfg.env.get("num_auxiliary_vehicles", 3),
        sim_area_x=cfg.env.get("sim_area_x", 1000.0),
        sim_area_y=cfg.env.get("sim_area_y", 1000.0),
        task_arrival_rate=cfg.env.get("task_arrival_rate", 0.5),
        task_size_range=(
            cfg.env.get("task_size_min", 100),
            cfg.env.get("task_size_max", 500)
        ),
        task_cycles_per_bit=cfg.env.get("task_cycles_per_bit", 1000),
        task_deadline=cfg.env.get("task_deadline", 1.0),
        mec_cpu_frequency=cfg.env.get("mec_cpu_frequency", 1e10),
        aux_cpu_frequency=cfg.env.get("aux_cpu_frequency", 5e9),
        user_cpu_frequency=cfg.env.get("user_cpu_frequency", 2e9),
        weight_delay=cfg.general.get("weight_delay", 1.0),
        weight_energy=cfg.general.get("weight_energy", 0.5),
        weight_cost=cfg.general.get("weight_cost", 0.3),
        weight_penalty=cfg.general.get("weight_penalty", 100.0),
        encryption_enabled=cfg.security.get("encryption", {}).get("enabled", True),
        encryption_mode=cfg.security.get("encryption", {}).get("mode", "simulation"),
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
    logger.info(f"Environment created: state_dim={env.observation_space.shape[0]}, "
                f"action_dim={env.action_space.shape}")
    
    # ==================== Create Agent ====================
    logger.info("Creating DRL agent...")
    agent = DRLAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.nvec,
        learning_rate=cfg.agent.hyperparameters.learning_rate,
        gamma=cfg.agent.hyperparameters.gamma,
        epsilon_start=cfg.agent.hyperparameters.epsilon_start,
        epsilon_end=cfg.agent.hyperparameters.epsilon_end,
        epsilon_decay=cfg.agent.hyperparameters.epsilon_decay,
        buffer_size=cfg.agent.hyperparameters.buffer_size,
        batch_size=cfg.agent.hyperparameters.batch_size,
        target_update_freq=cfg.agent.hyperparameters.target_update_freq,
        device=cfg.general.device
    )
    logger.info("DRL agent created successfully")
    
    # ==================== Training Loop ====================
    num_episodes = cfg.general.get("num_episodes", 100)
    eval_interval = cfg.general.get("eval_interval", 10)
    log_interval = cfg.general.get("log_interval", 5)
    
    training_history = []
    eval_history = []
    
    logger.info(f"Starting training for {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        # Train episode
        stats = agent.train_episode(env, max_steps=env_config.episode_length)
        training_history.append(stats)
        
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
    with open(history_path, 'w') as f:
        json.dump({
            "training": training_history,
            "evaluation": eval_history
        }, f, indent=2)
    
    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
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
