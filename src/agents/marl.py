"""
Parameter-Shared Independent DQN (IDQN) for Multi-Agent VEC (Phase 3 - Task T3.3).

All user vehicles share a central Q-network weights while maintaining their
local observations and action selection. Supports action masking and multi-agent coordination.
"""
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.networks import DuelingNetwork, StandardDQNNetwork
from src.agents.memory import ReplayBuffer
from src.utils.logger import Logger


class ParameterSharedIDQNAgent:
    """Independent Deep Q-Network Agent with Parameter Sharing for MARL."""

    def __init__(self,
                 state_dim: int,
                 action_dim: List[int],
                 learning_rate: float = 0.001,
                 gamma: float = 0.9,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 100,
                 use_dueling: bool = True,
                 device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = np.array(action_dim, dtype=np.int64)
        self.total_actions = int(np.prod(self.action_dim))
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
        self.logger = Logger("IDQNAgent")

        net_cls = DuelingNetwork if use_dueling else StandardDQNNetwork
        self.q_network = net_cls(state_dim, self.total_actions).to(self.device)
        self.target_network = net_cls(state_dim, self.total_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(buffer_size, state_dim)
        self.train_step_count = 0

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray = None, training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy over valid action mask."""
        if training and np.random.rand() < self.epsilon:
            if action_mask is not None:
                feasible_indices = np.where(action_mask)[0]
                if len(feasible_indices) > 0:
                    idx = np.random.choice(feasible_indices)
                    return np.array(np.unravel_index(idx, self.action_dim), dtype=np.int64)
            return np.array([np.random.randint(0, dim) for dim in self.action_dim], dtype=np.int64)

        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t).squeeze(0).cpu().numpy()

        if action_mask is not None:
            q_values[~action_mask] = -1e9

        best_idx = int(np.argmax(q_values))
        return np.array(np.unravel_index(best_idx, self.action_dim), dtype=np.int64)

    def update(self) -> float:
        """Train Q-network on a mini-batch from replay buffer."""
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones, elapsed_times, next_masks = self.memory.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q(s, a)
        action_indices_list = []
        for act in actions:
            if isinstance(act, (int, np.integer)):
                action_indices_list.append(int(act))
            else:
                action_indices_list.append(int(np.ravel_multi_index(act, self.action_dim)))

        action_indices = torch.tensor(action_indices_list, dtype=torch.long).unsqueeze(1).to(self.device)
        q_values = self.q_network(states_t).gather(1, action_indices)

        # Compute Target Q
        with torch.no_grad():
            max_next_q = self.target_network(next_states_t).max(1, keepdim=True)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * 0.995)
