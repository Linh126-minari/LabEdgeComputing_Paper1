"""
Experience Replay Buffer for Deep Reinforcement Learning
=========================================================
Stores and samples experiences for training DRL agents.

Phase 1 (T1.4, T1.6): mỗi transition lưu thêm
- elapsed_time: thời gian thực đã trôi (semi-MDP — discount γ^(Δt/dt));
- next_mask: action mask của state kế tiếp (masking target Q).
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple
import random


class ReplayBuffer:
    """
    Experience Replay Buffer for DRL training.

    Stores (state, action, reward, next_state, done, elapsed_time, next_mask)
    tuples and samples random batches for training.
    """

    def __init__(self, capacity: int = 100000, total_actions: Optional[int] = None):
        """
        Args:
            capacity: Maximum number of experiences to store
            total_actions: Kích thước action space phẳng (cho next_mask mặc định)
        """
        self.capacity = capacity
        self.total_actions = total_actions
        self.buffer = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool,
            elapsed_time: float = 1.0, next_mask: Optional[np.ndarray] = None):
        """
        Add experience to buffer.

        Args:
            elapsed_time: thời gian thực của transition (đơn vị: bội của dt cơ sở,
                tức elapsed/dt) — T1.4. Mặc định 1.0 = một slot.
            next_mask: bool mask của action khả thi ở next_state — T1.6.
                None → toàn True (không masking).
        """
        if next_mask is None and self.total_actions is not None:
            next_mask = np.ones(self.total_actions, dtype=bool)
        self.buffer.append((state, action, reward, next_state, done,
                            float(elapsed_time), next_mask))

    def sample(self, batch_size: int = 64) -> Tuple[np.ndarray, ...]:
        """
        Sample random batch from buffer.

        Returns:
            (states, actions, rewards, next_states, dones, elapsed_times, next_masks)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        batch = random.sample(self.buffer, batch_size)

        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        elapsed = np.array([exp[5] for exp in batch], dtype=np.float32)
        if batch[0][6] is None:
            next_masks = None
        else:
            next_masks = np.array([exp[6] for exp in batch], dtype=bool)

        return states, actions, rewards, next_states, dones, elapsed, next_masks

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int = 1000) -> bool:
        """Check if buffer has enough samples for training"""
        return len(self.buffer) >= min_size

    def clear(self):
        self.buffer.clear()
