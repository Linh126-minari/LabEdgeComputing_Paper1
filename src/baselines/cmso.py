"""
CMSO Algorithm 3 from Paper [A] using cvxpy (Phase 2 - T2.3).

Ref: [A] Section 4.2, Eq(32)-(37).
Solves the convex offloading proportion problem for AUX offloading branch.
"""
from typing import Optional, Any
import numpy as np
from src.baselines.base import BasePolicy

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


class CMSOPolicy(BasePolicy):
    """
    CMSO (Computation Offloading Scheme) from Paper [A].

    Optimizes local computation ratio alpha_l in [alpha_l_min, alpha_l_max]
    for AUX offloading using convex optimization (cvxpy).
    """

    def __init__(self, resource_level: int = 4):
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
        if hasattr(env, "task_queue") and env.task_queue.get(user.id):
            task = env.task_queue[user.id][0]
        else:
            from src.env.entities import Task
            task = Task(
                id=0, user_id=user.id,
                size=env.config.task_size_min,
                cycles=env.config.task_size_min * env.config.task_cycles_per_bit,
                max_delay=env.config.task_deadline_app
            )
        aux_ok, aux_idx = env._aux_feasible(user)

        if not aux_ok or aux_idx is None:
            # If AUX is not feasible, try MEC or fallback to Local
            if env._mec_feasible(user):
                return np.array([4, 0, self.resource_level], dtype=np.int64)
            return np.array([9, 0, 0], dtype=np.int64)

        aux = env.auxiliary_vehicles[aux_idx]
        tx_rate = env.channel_model.calc_rate(
            user.get_distance(aux), user.tx_power, env.config.bandwidth_v2v
        )

        cycles_total = task.size * task.cycles_per_bit
        # Bounds from [A] Eq(32)-(33)
        # alpha_l_ub = (tau * f_u) / (s * c)
        tau = max(task.max_delay, env.config.deadline_floor)
        alpha_l_ub = (tau * user.cpu_frequency) / max(cycles_total, 1.0)
        # alpha_l_lb = 1 - (tau * R) / s
        alpha_l_lb = 1.0 - (tau * tx_rate) / max(task.size, 1.0)

        alpha_min = max(0.0, float(alpha_l_lb))
        alpha_max = min(1.0, float(alpha_l_ub))

        if alpha_min > alpha_max:
            # Infeasible bounds fallback to local
            return np.array([9, 0, 0], dtype=np.int64)

        # Solve 1D convex problem over alpha_l in [alpha_min, alpha_max]
        optimal_alpha = self._solve_convex(env, user, aux, task, tx_rate, alpha_min, alpha_max)

        # Map continuous optimal_alpha in [0, 1] to discrete level based on alpha_levels
        alpha_levels = getattr(env.config, "alpha_levels", 10)
        alpha_level = int(np.clip(round(optimal_alpha * (alpha_levels - 1)), 0, alpha_levels - 1))

        return np.array([alpha_level, 1, self.resource_level], dtype=np.int64)

    def _solve_convex(
        self,
        env: Any,
        user: Any,
        aux: Any,
        task: Any,
        tx_rate: float,
        alpha_min: float,
        alpha_max: float
    ) -> float:
        """Solve convex optimization problem or analytical 1D grid search as fallback."""
        cycles_total = task.size * task.cycles_per_bit
        kappa_val = getattr(env.config, "kappa", 1e-27)
        if CVXPY_AVAILABLE:
            try:
                alpha = cp.Variable()
                
                f_u = user.cpu_frequency
                f_n = aux.cpu_frequency * ((self.resource_level + 1) / 10.0)
                
                lambda_l = getattr(env.config, "price_energy_local", 1.0)
                lambda_n = getattr(env.config, "price_energy_av", 0.8)

                objective = cp.Minimize(
                    lambda_l * (kappa_val * cycles_total * f_u**2 * alpha) +
                    lambda_n * (kappa_val * (1.0 - alpha) * cycles_total * f_n**2)
                )
                constraints = [alpha >= alpha_min, alpha <= alpha_max]
                prob = cp.Problem(objective, constraints)
                prob.solve(verbose=False)

                if alpha.value is not None:
                    return float(alpha.value)
            except Exception:
                pass

        # Fallback 1D grid search over [alpha_min, alpha_max]
        alphas = np.linspace(alpha_min, alpha_max, 50)
        best_alpha = alpha_min
        min_val = float("inf")
        
        f_u = user.cpu_frequency
        f_n = aux.cpu_frequency * ((self.resource_level + 1) / 10.0)
        lambda_l = getattr(env.config, "price_energy_local", 1.0)
        lambda_n = getattr(env.config, "price_energy_av", 0.8)

        for a_val in alphas:
            val = lambda_l * (kappa_val * cycles_total * (f_u ** 2) * a_val) + lambda_n * (kappa_val * (1.0 - a_val) * cycles_total * (f_n ** 2))
            if val < min_val:
                min_val = val
                best_alpha = float(a_val)

        return best_alpha
