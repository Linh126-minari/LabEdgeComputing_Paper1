"""
Multi-Agent Vehicular Edge Environment Wrapper (Phase 3 - Task T3.3).

Provides PettingZoo ParallelEnv style multi-agent interface for multi-user VEC.
Supports individual rewards, Max-Min fairness rewards (Paper [B] headline metric),
and mixed cooperation/competition trade-off rewards.
"""
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig


class MultiAgentVehicularEdgeEnv:
    """Multi-Agent VEC Environment for MARL (IDQN / QMIX)."""

    def __init__(self, config: EnvironmentConfig, reward_mode: str = "combined", xi: float = 0.5):
        self.config = config
        self.env = VehicularEdgeEnvironment(config)
        self.num_users = len(self.env.users)
        self.agents = [f"user_{i}" for i in range(self.num_users)]
        self.reward_mode = reward_mode  # "individual", "max_min", "combined"
        self.xi = float(np.clip(xi, 0.0, 1.0))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)
        obs_dict = {f"user_{i}": self._get_user_state(i) for i in range(self.num_users)}
        info_dict = {f"user_{i}": {"action_mask": self.env.action_mask(self.env.users[i])}
                     for i in range(self.num_users)}
        return obs_dict, info_dict

    def _get_user_state(self, user_idx: int) -> np.ndarray:
        user = self.env.users[user_idx]
        state_vector = []

        if self.env.task_queue[user.id]:
            task = self.env.task_queue[user.id][0]
            tau_eff, _ = self.env.effective_deadline(user, task)
            task_size_norm = min(task.size / self.config.task_size_max, 1.0)
            deadline_norm = min(tau_eff / max(self.config.task_deadline, 1e-6), 1.0)
            omega = 1.0 / max(tau_eff, 0.1)
            urgency_norm = min(omega / 10.0, 1.0)
        else:
            task_size_norm, deadline_norm, urgency_norm = 0.0, 0.0, 0.0

        state_vector.extend([task_size_norm, deadline_norm, urgency_norm])

        max_dist = np.sqrt(self.config.sim_area_x ** 2 + self.config.sim_area_y ** 2)
        for mec in self.env.mec_servers:
            state_vector.append(min(user.get_distance(mec) / max_dist, 1.0))
        for aux in self.env.auxiliary_vehicles:
            state_vector.append(min(user.get_distance(aux) / max_dist, 1.0))

        for i in range(len(self.env.mec_servers)):
            util = 1.0 - self.env.mec_cpu_available[i] / self.config.mec_cpu_frequency
            state_vector.append(float(np.clip(util, 0.0, 1.0)))
        for i in range(len(self.env.auxiliary_vehicles)):
            util = 1.0 - self.env.aux_cpu_available[i] / self.config.aux_cpu_frequency
            state_vector.append(float(np.clip(util, 0.0, 1.0)))

        if self.env.mec_servers and self.env.task_queue[user.id]:
            dist = user.get_distance(self.env.mec_servers[0])
            rate = self.env.channel_model.get_uplink_rate_mec(dist, user.tx_power)
            state_vector.append(min(rate / 1e7, 1.0))
        else:
            state_vector.append(0.0)

        return np.array(state_vector, dtype=np.float32)

    def step(self, actions_dict: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]
    ]:
        user_rewards = {}
        user_metrics = {}

        # Execute actions for all users
        for i, user in enumerate(self.env.users):
            agent_id = f"user_{i}"
            action = actions_dict.get(agent_id, np.array([9, 0, 0, 2], dtype=np.int64))
            
            alpha_l, alpha_j, alpha_n, res_level, block_len = self.env._decode_action_to_alphas(action)
            if self.env.task_queue[user.id]:
                task = self.env.task_queue[user.id].pop(0)
                tau_eff, link_limited = self.env.effective_deadline(user, task)
                m = self.env._execute_task(user, task, alpha_l, alpha_j, alpha_n, res_level, tau_eff, block_len)
                r, _ = self.env._compute_reward(m)
                user_rewards[agent_id] = r
                user_metrics[agent_id] = m
            else:
                user_rewards[agent_id] = 0.0
                user_metrics[agent_id] = {}

        # Compute fairness-adjusted rewards
        raw_rewards_list = list(user_rewards.values())
        min_reward = min(raw_rewards_list) if raw_rewards_list else 0.0

        final_rewards = {}
        for agent_id, raw_r in user_rewards.items():
            if self.reward_mode == "individual":
                final_rewards[agent_id] = raw_r
            elif self.reward_mode == "max_min":
                final_rewards[agent_id] = min_reward
            elif self.reward_mode == "combined":
                final_rewards[agent_id] = (1.0 - self.xi) * raw_r + self.xi * min_reward
            else:
                final_rewards[agent_id] = raw_r

        self.env._advance_time()
        self.env.step_count += 1

        terminated_dict = {a: False for a in self.agents}
        truncated_dict = {a: self.env.step_count >= self.config.episode_length for a in self.agents}
        terminated_dict["__all__"] = False
        truncated_dict["__all__"] = any(truncated_dict.values())

        obs_dict = {f"user_{i}": self._get_user_state(i) for i in range(self.num_users)}
        info_dict = {f"user_{i}": {
            "action_mask": self.env.action_mask(self.env.users[i]),
            "raw_reward": user_rewards[f"user_{i}"],
            "metrics": user_metrics[f"user_{i}"]
        } for i in range(self.num_users)}

        return obs_dict, final_rewards, terminated_dict, truncated_dict, info_dict
