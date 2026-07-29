"""
Deep Reinforcement Learning Agent Policy
=========================================
Implements DRL agent using Dueling DQN for task offloading optimization.

Reference: Paper's DDRP algorithm based on Dueling DQN.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List
from collections import deque

from src.agents.networks import DuelingNetwork, StandardDQNNetwork
from src.agents.memory import ReplayBuffer
from src.utils.logger import Logger


class DRLAgent:
    """
    Deep Reinforcement Learning Agent using Dueling + Double DQN.

    LƯU Ý thuật ngữ (T0.11): [A] gọi phương pháp là Dueling DQN (DDRP) — chỉ
    dueling. Việc thêm Double DQN là MỞ RỘNG của dự án so với [A]; DDQN thuần
    vẫn phải giữ làm baseline riêng (T2.5) để ablation có nghĩa.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim,
                 learning_rate: float = 0.001,
                 gamma: float = 0.9,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 100,
                 learning_starts: int = 2000,
                 use_dueling: bool = True,
                 use_double_dqn: bool = True,
                 use_soft_update: bool = False,
                 tau: float = 0.005,
                 epsilon_schedule: str = "linear",
                 exploration_fraction: float = 0.3,
                 total_training_steps: int = 500000,
                 base_dt: float = 0.1,
                 device: str = "cpu"):
        """
        Initialize DRL agent.

        Args:
            state_dim: Dimension of state space
            action_dim: MultiDiscrete action dimensions (e.g., env.action_space.nvec)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Final epsilon for exploration
            epsilon_decay: Decay rate (chỉ dùng khi epsilon_schedule="exponential")
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency to update target network (hard update)
            learning_starts: Số sample tối thiểu trong buffer trước khi train
            use_dueling: Dùng Dueling Network architecture (T2.5)
            use_double_dqn: Dùng Double DQN target (T0.11)
            use_soft_update: Soft update target (τ mỗi step) thay hard copy
            tau: Hệ số soft update
            epsilon_schedule: "linear" hoặc "exponential"
            exploration_fraction: Tỉ lệ số step huấn luyện dành cho decay ε
            total_training_steps: Tổng số step dự kiến (cho linear decay)
            base_dt: Bước thời gian cơ sở của env
            device: Device to use (cpu/cuda)
        """
        self.state_dim = state_dim
        self.action_dim = np.array(action_dim, dtype=np.int64)
        self.total_actions = int(np.prod(self.action_dim))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = max(int(learning_starts), batch_size)
        self.use_dueling = use_dueling
        self.use_double_dqn = use_double_dqn
        self.use_soft_update = use_soft_update
        self.tau = tau
        self.epsilon_schedule = epsilon_schedule
        self.exploration_fraction = exploration_fraction
        self.total_training_steps = max(int(total_training_steps), 1)
        self.base_dt = float(base_dt)
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.logger = Logger("DRLAgent")
        
        # ==================== Networks ====================
        net_cls = DuelingNetwork if use_dueling else StandardDQNNetwork
        self.q_network = net_cls(state_dim, self.total_actions).to(self.device)
        self.target_network = net_cls(state_dim, self.total_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # No gradient for target network
        
        # ==================== Optimizer & Loss ====================
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        # T0.11: Huber loss (SmoothL1) — ổn định hơn MSE với thang reward lớn
        self.loss_fn = nn.SmoothL1Loss()
        
        # ==================== Memory ====================
        self.replay_buffer = ReplayBuffer(capacity=buffer_size,
                                          total_actions=self.total_actions)
        
        # ==================== Tracking ====================
        self.step_count = 0
        self.episode_count = 0
        self.training_losses = deque(maxlen=100)
        
        self.logger.info(
            f"DRL Agent initialized: state_dim={state_dim}, action_dim={self.action_dim.tolist()}, "
            f"total_actions={self.total_actions}"
        )

    def _flatten_action(self, action: np.ndarray) -> int:
        """Convert MultiDiscrete action vector to flat action index."""
        action_arr = np.asarray(action, dtype=np.int64)
        return int(np.ravel_multi_index(action_arr, self.action_dim))

    def _unflatten_action(self, action_idx: int) -> np.ndarray:
        """Convert flat action index to MultiDiscrete action vector."""
        action_tuple = np.unravel_index(int(action_idx), tuple(self.action_dim.tolist()))
        return np.array(action_tuple, dtype=np.int64)
    
    def select_action(self, state: np.ndarray, training: bool = True,
                      mask: np.ndarray = None) -> np.ndarray:
        """
        Select action using epsilon-greedy strategy with optional action masking.

        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            mask: Bool array (total_actions,) — True = khả thi (T1.6).
                None → mọi action khả thi.

        Returns:
            Selected action (MultiDiscrete vector)
        """
        if training and np.random.random() < self.epsilon:
            # Explore: sample CHỈ trong tập khả thi (T1.6)
            if mask is not None and mask.any():
                action_idx = int(np.random.choice(np.flatnonzero(mask)))
                return self._unflatten_action(action_idx)
            return np.array([np.random.randint(0, int(dim)) for dim in self.action_dim],
                            dtype=np.int64)

        # Exploit: greedy trên Q đã mask
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0)
            if mask is not None and mask.any():
                q_values = q_values.masked_fill(
                    ~torch.as_tensor(mask, device=self.device), float("-inf"))
            action_idx = int(q_values.argmax().item())
            return self._unflatten_action(action_idx)
    
    def remember(self, state: np.ndarray, action: np.ndarray, reward: float,
                 next_state: np.ndarray, done: bool,
                 elapsed_time: float = None, next_mask: np.ndarray = None):
        """
        Store experience in replay buffer.

        Args:
            elapsed_time: thời gian THỰC của transition (giây) — T1.4 semi-MDP.
                None → một slot dt cơ sở.
            next_mask: action mask của next_state — T1.6.
        """
        action_int = self._flatten_action(action)
        elapsed_ratio = 1.0 if elapsed_time is None else max(elapsed_time / self.base_dt, 1e-6)
        self.replay_buffer.add(state, action_int, reward, next_state, done,
                               elapsed_time=elapsed_ratio, next_mask=next_mask)
    
    def train_step(self) -> float:
        """
        Perform one training step on batch from replay buffer.
        
        Returns:
            Loss value
        """
        # T0.11: warm-up — chờ đủ learning_starts sample (không phải chỉ 1 batch)
        if not self.replay_buffer.is_ready(self.learning_starts):
            return 0.0

        # Sample batch (T1.4/T1.6: kèm elapsed_time và next_mask)
        states, actions, rewards, next_states, dones, elapsed, next_masks = \
            self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        elapsed_tensor = torch.FloatTensor(elapsed).to(self.device)

        # ==================== Compute Q-values ====================
        # Current Q-values from policy network
        q_values = self.q_network(states_tensor)
        q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Target Q-values (T0.11 Double DQN + T1.6 masking + T1.4 semi-MDP discount)
        with torch.no_grad():
            neg_inf = torch.tensor(float("-inf"), device=self.device)
            mask_tensor = None
            if next_masks is not None:
                mask_tensor = torch.as_tensor(next_masks, device=self.device)
                # phòng hờ: hàng toàn False → cho phép tất cả (tránh -inf lan)
                all_false = ~mask_tensor.any(dim=1)
                if all_false.any():
                    mask_tensor[all_false] = True

            if self.use_double_dqn:
                # Double DQN: chọn action bằng ONLINE net, đánh giá bằng TARGET net
                online_next_q = self.q_network(next_states_tensor)
                if mask_tensor is not None:
                    online_next_q = online_next_q.masked_fill(~mask_tensor, neg_inf)
                next_actions = online_next_q.argmax(dim=1, keepdim=True)
                max_next_q_values = self.target_network(next_states_tensor).gather(
                    1, next_actions).squeeze(1)
            else:
                # DQN thuần (giữ làm baseline)
                next_q_values = self.target_network(next_states_tensor)
                if mask_tensor is not None:
                    next_q_values = next_q_values.masked_fill(~mask_tensor, neg_inf)
                max_next_q_values = next_q_values.max(dim=1)[0]

            # T1.4 (semi-MDP): discount lũy theo THỜI GIAN thực,
            # γ_eff = γ^(elapsed_time / dt) — hai transition cách nhau 0.1 s và
            # 5 s không còn bị discount như nhau.
            gamma_eff = torch.pow(torch.tensor(self.gamma, device=self.device),
                                  elapsed_tensor)
            target_q_values = rewards_tensor + gamma_eff * max_next_q_values * (1 - dones_tensor)
        
        # ==================== Backward Pass ====================
        loss = self.loss_fn(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.training_losses.append(loss.item())
        
        return loss.item()
    
    def update_target_network(self):
        """
        Hard update: copy toàn bộ trọng số online net sang target net.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update_target_network(self):
        """
        Soft update (T0.11, tùy chọn): θ_target ← τ·θ_online + (1−τ)·θ_target
        mỗi step, thay vì hard copy mỗi target_update_freq step.
        """
        with torch.no_grad():
            for tp, op in zip(self.target_network.parameters(), self.q_network.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * op.data)

    def update_epsilon(self):
        """
        Decay epsilon cho epsilon-greedy (T0.11).

        - "linear" (khuyến nghị): decay tuyến tính theo TỔNG SỐ STEP, đạt
          epsilon_end sau exploration_fraction · total_training_steps step.
          Gọi mỗi step — an toàn vì phụ thuộc step_count, không tích lũy.
        - "exponential": ε ← max(ε_end, ε·decay). CHỈ gọi mỗi EPISODE —
          nếu gọi mỗi step với decay=0.995, ε sụp về 0.01 sau ~920 step.
        """
        if self.epsilon_schedule == "linear":
            decay_steps = self.exploration_fraction * self.total_training_steps
            frac = min(self.step_count / max(decay_steps, 1.0), 1.0)
            self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
        else:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _shape_reward(self, reward: float, info: dict) -> float:
        """
        Hook cho subclass (T1.3 — Lagrangian agent cộng −λ_T·viol_T − λ_E·viol_E).
        Agent chuẩn trả nguyên reward của env.
        """
        _ = info
        return reward

    def _on_episode_end(self, episode_stats: dict):
        """Hook cho subclass (T1.3 — dual ascent cập nhật multiplier)."""
        _ = episode_stats

    def train_episode(self, env, max_steps: int = 1000) -> dict:
        """
        Train for one episode (Gymnasium API + semi-MDP + masking, T1.4/T1.6).
        """
        state, info = env.reset()
        mask = info.get("action_mask")
        episode_reward = 0.0
        episode_loss = 0.0
        num_updates = 0
        num_viol_T = 0
        viol_E_sum = 0.0
        step = 0

        for step in range(max_steps):
            action = self.select_action(state, training=True, mask=mask)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_mask = info.get("action_mask")

            # T1.3 hook: reward định hình (Lagrangian) — agent chuẩn giữ nguyên
            shaped_reward = self._shape_reward(reward, info)
            num_viol_T += int(info.get("deadline_violated", False))
            viol_E_sum += float(info.get("energy_violation", 0.0))

            self.remember(state, action, shaped_reward, next_state, done,
                          elapsed_time=info.get("elapsed_time"),
                          next_mask=next_mask)

            loss = self.train_step()
            episode_loss += loss
            num_updates += 1

            # Update target network (T0.11: soft hoặc hard qua config)
            if self.use_soft_update:
                self.soft_update_target_network()
            elif self.step_count % self.target_update_freq == 0:
                self.update_target_network()

            episode_reward += reward  # log reward GỐC (không gồm shaping)
            self.step_count += 1

            # T0.11: linear decay gọi mỗi STEP (phụ thuộc step_count, không tích lũy)
            if self.epsilon_schedule == "linear":
                self.update_epsilon()

            state = next_state
            mask = next_mask

            if done:
                break

        # T0.11: exponential decay chỉ gọi mỗi EPISODE
        if self.epsilon_schedule != "linear":
            self.update_epsilon()

        self.episode_count += 1

        n_steps = step + 1
        stats = {
            "episode": self.episode_count,
            "total_reward": episode_reward,
            "avg_loss": episode_loss / max(num_updates, 1),
            "epsilon": self.epsilon,
            "steps": n_steps,
            "viol_rate_T": num_viol_T / max(n_steps, 1),
            "avg_viol_E": viol_E_sum / max(n_steps, 1),
        }
        self._on_episode_end(stats)  # T1.3 hook
        return stats

    def evaluate_episode(self, env, max_steps: int = 1000) -> dict:
        """
        Evaluate agent performance for one episode (no training, no exploration).
        """
        state, info = env.reset()
        mask = info.get("action_mask")
        episode_reward = 0.0
        step = 0

        for step in range(max_steps):
            action = self.select_action(state, training=False, mask=mask)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            mask = info.get("action_mask")
            if terminated or truncated:
                break

        metrics = env.get_metrics()
        metrics.update({
            "episode_reward": episode_reward,
            "steps": step + 1
        })
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        torch.save(self.q_network.state_dict(), filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from file.
        
        Args:
            filepath: Path to load model
        """
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_training_stats(self) -> dict:
        """
        Get current training statistics.
        
        Returns:
            Dictionary with training metrics
        """
        avg_loss = np.mean(list(self.training_losses)) if self.training_losses else 0.0
        
        return {
            "episode": self.episode_count,
            "step": self.step_count,
            "epsilon": self.epsilon,
            "avg_loss": avg_loss,
            "buffer_size": len(self.replay_buffer)
        }
