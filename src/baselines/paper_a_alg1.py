"""
Algorithm 1 của [A] — chọn đích offload (Phase 2 - T2.2).

1. Trong vùng phủ MEC → MEC.  2. Ngược lại AUX rảnh trong tầm có giá g_n nhỏ nhất.  3. Không có → local.
α_l và β cố định (Algorithm 1 chỉ chọn đích; [A] tối ưu α, β bằng DDRP/CMSO).
"""
from typing import Optional, Any
import numpy as np
from src.baselines.base import BasePolicy


class PaperAAlgorithm1Policy(BasePolicy):
    def __init__(self, default_alpha_l_level: int = 4, resource_level: int = 4):
        self.default_alpha_l_level = default_alpha_l_level
        self.resource_level = resource_level

    def select_action(self, state: np.ndarray, env: Optional[Any] = None, training: bool = False) -> np.ndarray:
        user = env.users[0]
        if env._mec_feasible(user):
            return env.make_action(self.default_alpha_l_level, 0, self.resource_level)
        if env._aux_feasible(user)[0]:
            return env.make_action(self.default_alpha_l_level, 1, self.resource_level)
        return env.make_action(int(env.action_space.nvec[0]) - 1, 0, 0)
