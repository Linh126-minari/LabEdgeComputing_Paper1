"""
Heuristic Baseline Policies (Phase 2 - T2.1).

1. AllLocalPolicy  (α_l = 1)
2. AllMECPolicy    (offload hết lên MEC nếu trong vùng phủ, ngược lại local)
3. AllAUXPolicy    (offload hết sang AUX rảnh trong tầm, ngược lại local)
4. RandomPolicy    (ngẫu nhiên đều trên action khả thi)
5. GreedyPolicy    (thử MỌI action khả thi bằng env.evaluate_action — cùng mô hình chi phí với env,
                    chọn min C_total, ưu tiên action thỏa ràng buộc deadline/năng lượng)
"""
from typing import Optional, Any
import numpy as np
from src.baselines.base import BasePolicy


def _local(env) -> np.ndarray:
    return env.make_action(int(env.action_space.nvec[0]) - 1, 0, 0)


class AllLocalPolicy(BasePolicy):
    def select_action(self, state: np.ndarray, env: Optional[Any] = None, training: bool = False) -> np.ndarray:
        return _local(env)


class AllMECPolicy(BasePolicy):
    def __init__(self, resource_level: int = 4):
        self.resource_level = resource_level

    def select_action(self, state: np.ndarray, env: Optional[Any] = None, training: bool = False) -> np.ndarray:
        if env._mec_feasible(env.users[0]):
            return env.make_action(0, 0, self.resource_level)
        return _local(env)


class AllAUXPolicy(BasePolicy):
    """Offload hết sang AUX; N_i theo nghiệm dạng đóng [B] Eq(39) nếu action có N_i."""

    def __init__(self, resource_level: int = 4):
        self.resource_level = resource_level

    def select_action(self, state: np.ndarray, env: Optional[Any] = None, training: bool = False) -> np.ndarray:
        if env._aux_feasible(env.users[0])[0]:
            return env.make_action(0, 1, self.resource_level, _gbd_block_index(env))
        return _local(env)


def _gbd_block_index(env) -> Optional[int]:
    """N_i* từ [B] Eq(39) → chỉ số gần nhất trong block_length_options."""
    if len(env.action_space.nvec) < 4:
        return None
    from src.baselines.gbd import GBDClosedFormSecurityBaseline
    cfg = env.config
    n_e = env.security_overhead_model.expected_attackers(cfg.security_range_v2v, cfg.attacker_density,
                                                         cfg.lane_count, cfg.road_max_speed_mps)
    n_star = GBDClosedFormSecurityBaseline(int(cfg.frame_length_bits)).compute_optimal_block_length(n_e)
    return int(np.argmin([abs(o - n_star) for o in cfg.block_length_options]))


class RandomPolicy(BasePolicy):
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def select_action(self, state: np.ndarray, env: Optional[Any] = None, training: bool = False) -> np.ndarray:
        nvec = tuple(int(n) for n in env.action_space.nvec)
        idx = self.rng.choice(np.where(env.action_mask())[0])
        return np.array(np.unravel_index(idx, nvec), dtype=np.int64)


class GreedyPolicy(BasePolicy):
    """Myopic: min chi phí tức thời trên các action khả thi (thỏa ràng buộc nếu có thể)."""

    def select_action(self, state: np.ndarray, env: Optional[Any] = None, training: bool = False) -> np.ndarray:
        nvec = tuple(int(n) for n in env.action_space.nvec)
        best, best_key = _local(env), (True, float("inf"))
        for idx in np.where(env.action_mask())[0]:
            action = np.array(np.unravel_index(idx, nvec), dtype=np.int64)
            m = env.evaluate_action(action)
            key = (bool(m["viol_T"] or m["viol_E"] > 0), m["cost_total"])
            if key < best_key:
                best, best_key = action, key
        return best
