"""
Algorithm 1 from Paper [A] — User Selection Policy (Phase 2 - T2.2).

Ref: [A] Section 3.5 & Algorithm 1.
Selects MEC server if covered; otherwise selects AUX vehicle with min dynamic price g_n.
"""
from typing import Optional, Any
import numpy as np
from src.baselines.base import BasePolicy


class PaperAAlgorithm1Policy(BasePolicy):
    """
    User Selection algorithm based on [A] Algorithm 1.

    Choice logic:
    1. If user is inside MEC coverage radius -> Select MEC (target = 0).
    2. Else if idle AUX vehicles exist within V2V range -> Select AUX with min dynamic price g_n (target = 1).
    3. Else -> Fallback to Local execution (alpha_l = 1.0).
    """

    def __init__(self, default_alpha_l_level: int = 4, resource_level: int = 4):
        """
        Args:
            default_alpha_l_level: Default offload ratio level when offloading (default 4 -> alpha_l = 4/9 ~ 0.44).
            resource_level: Default resource level (default 4 -> 50% pool).
        """
        self.default_alpha_l_level = default_alpha_l_level
        self.resource_level = resource_level

    def select_action(
        self,
        state: np.ndarray,
        env: Optional[Any] = None,
        training: bool = False
    ) -> np.ndarray:
        if env is None:
            return np.array([9, 0, 0], dtype=np.int64)

        user = env.users[0]

        # 1. If in MEC coverage -> Select MEC
        if env._mec_feasible(user):
            return np.array([self.default_alpha_l_level, 0, self.resource_level], dtype=np.int64)

        # 2. Otherwise check candidate AUX vehicles
        aux_ok, best_aux_idx = env._aux_feasible(user)
        if aux_ok and best_aux_idx is not None:
            return np.array([self.default_alpha_l_level, 1, self.resource_level], dtype=np.int64)

        # 3. Fallback to Local
        return np.array([9, 0, 0], dtype=np.int64)
