"""
Base Policy Abstract Class for Baselines and Agents (Phase 2 - T2.1).
"""
from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy as np


class BasePolicy(ABC):
    """Abstract base class for all offloading decision policies."""

    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        env: Optional[Any] = None,
        training: bool = False
    ) -> np.ndarray:
        """
        Select an action given state vector and optional env context.

        Args:
            state: Normalized environment state vector.
            env: Optional VehicularEdgeEnvironment instance for state queries.
            training: Whether policy is in training mode.

        Returns:
            Action array, e.g. MultiDiscrete [alpha_l_level, target, resource_level].
        """
        pass
