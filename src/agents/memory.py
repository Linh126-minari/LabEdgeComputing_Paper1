"""
Experience Replay Buffer for Deep Reinforcement Learning
=========================================================
Stores and samples experiences for training DRL agents.
"""

import numpy as np
from collections import deque
from typing import Tuple
import random


class ReplayBuffer:
    """
    Experience Replay Buffer for DRL training.
    
    Stores (state, action, reward, next_state, done) tuples and samples
    random batches for training to break correlation in data.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                     np.ndarray, np.ndarray]:
        """
        Sample random batch from buffer.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            (states, actions, rewards, next_states, dones) as numpy arrays
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)
    
    def is_ready(self, min_size: int = 1000) -> bool:
        """Check if buffer has enough samples for training"""
        return len(self.buffer) >= min_size
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
