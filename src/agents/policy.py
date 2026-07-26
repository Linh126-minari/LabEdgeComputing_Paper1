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

from src.agents.networks import DuelingNetwork
from src.agents.memory import ReplayBuffer
from src.utils.logger import Logger


class DRLAgent:
    """
    Deep Reinforcement Learning Agent using Dueling DQN.
    
    Implements the DDRP (Dueling DQN Resource Partitioning) algorithm
    from the paper for task offloading decisions in vehicular edge computing.
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
            epsilon_decay: Decay rate for epsilon
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency to update target network
            device: Device to use (cpu/cuda)
        """
        self.state_dim = state_dim
        self.action_dim = np.array(action_dim, dtype=np.int64)
        self.total_actions = int(np.prod(self.action_dim))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        
        self.logger = Logger("DRLAgent")
        
        # ==================== Networks ====================
        # Main Q-network (policy network)
        self.q_network = DuelingNetwork(state_dim, self.total_actions).to(self.device)
        
        # Target Q-network (for stability)
        self.target_network = DuelingNetwork(state_dim, self.total_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # No gradient for target network
        
        # ==================== Optimizer & Loss ====================
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # ==================== Memory ====================
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
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
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using epsilon-greedy strategy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
        
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.array([
                np.random.randint(0, int(dim)) for dim in self.action_dim
            ], dtype=np.int64)
        else:
            # Exploit: greedy action from Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)

                action_idx = int(q_values.argmax(dim=1).item())
                action = self._unflatten_action(action_idx)
        
        return action
    
    def remember(self, state: np.ndarray, action: np.ndarray, reward: float,
                next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode terminated
        """
        # Convert MultiDiscrete action to single index for DQN update.
        action_int = self._flatten_action(action)
        self.replay_buffer.add(state, action_int, reward, next_state, done)
    
    def train_step(self) -> float:
        """
        Perform one training step on batch from replay buffer.
        
        Returns:
            Loss value
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # ==================== Compute Q-values ====================
        # Current Q-values from policy network
        q_values = self.q_network(states_tensor)
        q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Target Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards_tensor + self.gamma * max_next_q_values * (1 - dones_tensor)
        
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
        Update target network weights with policy network weights.
        
        Called periodically to stabilize training.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        """
        Decay epsilon for epsilon-greedy strategy.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train_episode(self, env, max_steps: int = 1000) -> dict:
        """
        Train for one episode.
        
        Args:
            env: Environment to train in
            max_steps: Maximum steps per episode
        
        Returns:
            Episode statistics dictionary
        """
        state = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        num_updates = 0
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(state, training=True)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            self.remember(state, action, reward, next_state, done)
            
            # Train on batch
            loss = self.train_step()
            episode_loss += loss
            num_updates += 1
            
            # Update target network periodically
            if self.step_count % self.target_update_freq == 0:
                self.update_target_network()
            
            episode_reward += reward
            self.step_count += 1
            
            state = next_state
            
            if done:
                break
        
        # Decay epsilon
        self.update_epsilon()
        
        self.episode_count += 1
        
        return {
            "episode": self.episode_count,
            "total_reward": episode_reward,
            "avg_loss": episode_loss / max(num_updates, 1),
            "epsilon": self.epsilon,
            "steps": step + 1
        }
    
    def evaluate_episode(self, env, max_steps: int = 1000) -> dict:
        """
        Evaluate agent performance for one episode (no training).
        
        Args:
            env: Environment to evaluate in
            max_steps: Maximum steps per episode
        
        Returns:
            Episode statistics dictionary
        """
        state = env.reset()
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Select action (no exploration)
            action = self.select_action(state, training=False)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            state = next_state
            
            if done:
                break
        
        # Get environment metrics
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
