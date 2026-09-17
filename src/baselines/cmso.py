"""
CMSO — Algorithm 3 của [A] (Phase 2 - T2.3), bản rời rạc hóa.

[A] Alg. 3: (1) chọn AUX theo Algorithm 1; (2) cận α_l từ Eq(32)-(33): α_l ≤ t·f/(s·c) và
α_l ≥ 1 − t·R/s; (3) f* = α_l·s·c/t (Eq 34 — trong env là DVFS); (4) min Cⁿ(α_l) (lồi, Eq 36-37).
Ở đây bước (4) được giải CHÍNH XÁC trên lưới α của env bằng env.evaluate_action, nên chi phí dùng
đúng mô hình của env (λⁿ, giá g_n, rủi ro rò rỉ) thay vì một công thức chép lại. Nhánh MEC không
thuộc CMSO ([A] giải bằng DDRP) → dùng α_l giữa như Algorithm 1.
"""
from typing import Optional, Any
import numpy as np
from src.baselines.base import BasePolicy
from src.baselines.heuristic import _gbd_block_index


class CMSOPolicy(BasePolicy):
    def __init__(self, resource_level: int = 4):
        self.resource_level = resource_level

    def select_action(self, state: np.ndarray, env: Optional[Any] = None, training: bool = False) -> np.ndarray:
        user = env.users[0]
        L = int(env.action_space.nvec[0])
        local = env.make_action(L - 1, 0, 0)
        aux_ok, aux_idx = env._aux_feasible(user)
        if not aux_ok:
            return env.make_action(L // 2, 0, self.resource_level) if env._mec_feasible(user) else local

        task = env.task_queue[user.id][0]
        tau, _ = env.effective_deadline(user, task)
        rate = env.channel_model.get_uplink_rate_v2v(user.get_distance(env.auxiliary_vehicles[aux_idx]), user.tx_power)
        cycles = task.size * task.cycles_per_bit
        lb = max(0.0, 1.0 - tau * rate / max(task.size, 1.0))            # [A] Eq(33)
        ub = min(1.0, tau * user.cpu_frequency / max(cycles, 1.0))       # [A] Eq(32)
        if lb > ub:
            return local

        blk = _gbd_block_index(env)
        best, best_key = local, None
        for lvl in range(L):
            a_l = lvl / max(L - 1, 1)
            if not (lb - 1e-9 <= a_l <= ub + 1e-9):
                continue
            action = env.make_action(lvl, 1, self.resource_level, blk)
            m = env.evaluate_action(action)
            key = (bool(m["viol_T"] or m["viol_E"] > 0), m["cost_total"])
            if best_key is None or key < best_key:
                best, best_key = action, key
        return best
