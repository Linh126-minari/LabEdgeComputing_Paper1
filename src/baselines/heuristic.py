"""
Heuristic Baseline Policies (Phase 2 - T2.1).

Contains:
1. AllLocalPolicy (α_l = 1.0)
2. AllMECPolicy (Offload to MEC if covered, else Local)
3. AllAUXPolicy (Offload to AUX if available, else Local)
4. RandomPolicy (Uniform random sampling over feasible action mask)
5. GreedyPolicy (Brute-force evaluation over valid actions to minimize immediate cost)
"""
from typing import Optional, Any
import numpy as np
from src.baselines.base import BasePolicy


class AllLocalPolicy(BasePolicy):
    """Always executes tasks locally (alpha_l = 1.0)."""

    def select_action(
        self,
        state: np.ndarray,
        env: Optional[Any] = None,
        training: bool = False
    ) -> np.ndarray:
        # alpha_l_level = 9 (100% local), target = 0, resource_level = 0
        return np.array([9, 0, 0], dtype=np.int64)


class AllMECPolicy(BasePolicy):
    """Offloads completely to MEC if within coverage; otherwise falls back to local."""

    def __init__(self, resource_level: int = 4):
        self.resource_level = resource_level

    def select_action(
        self,
        state: np.ndarray,
        env: Optional[Any] = None,
        training: bool = False
    ) -> np.ndarray:
        if env is not None and hasattr(env, "_mec_feasible"):
            user = env.users[0]
            if env._mec_feasible(user):
                return np.array([0, 0, self.resource_level], dtype=np.int64)
        return np.array([9, 0, 0], dtype=np.int64)


class AllAUXPolicy(BasePolicy):
    """Offloads completely to AUX if an idle AUX vehicle is in range; otherwise falls back to local."""

    def __init__(self, resource_level: int = 4):
        self.resource_level = resource_level

    def select_action(
        self,
        state: np.ndarray,
        env: Optional[Any] = None,
        training: bool = False
    ) -> np.ndarray:
        if env is not None and hasattr(env, "_aux_feasible"):
            user = env.users[0]
            aux_ok, _ = env._aux_feasible(user)
            if aux_ok:
                return np.array([0, 1, self.resource_level], dtype=np.int64)
        return np.array([9, 0, 0], dtype=np.int64)


class RandomPolicy(BasePolicy):
    """Uniform random sampling over feasible action mask."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def select_action(
        self,
        state: np.ndarray,
        env: Optional[Any] = None,
        training: bool = False
    ) -> np.ndarray:
        if env is not None and hasattr(env, "action_mask"):
            mask = env.action_mask()
            feasible_indices = np.where(mask)[0]
            if len(feasible_indices) > 0:
                idx = self.rng.choice(feasible_indices)
                return np.array(np.unravel_index(idx, (10, 2, 10)), dtype=np.int64)
        
        # Fallback to random action over space
        alpha_l = self.rng.randint(0, 10)
        target = self.rng.randint(0, 2)
        res = self.rng.randint(0, 10)
        return np.array([alpha_l, target, res], dtype=np.int64)


class GreedyPolicy(BasePolicy):
    """Evaluates all valid actions in action_mask to minimize immediate cost (C_u - varpi_1 * U)."""

    def select_action(
        self,
        state: np.ndarray,
        env: Optional[Any] = None,
        training: bool = False
    ) -> np.ndarray:
        if env is None or not hasattr(env, "action_mask"):
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
        mask = env.action_mask(user)
        feasible_indices = np.where(mask)[0]

        best_action = np.array([9, 0, 0], dtype=np.int64)
        min_cost = float("inf")

        for idx in feasible_indices:
            action_tuple = np.unravel_index(idx, (10, 2, 10))
            alpha_l_level, target, res_level = action_tuple
            
            alpha_l = alpha_l_level / 9.0
            resource_frac = (res_level + 1) / 10.0

            cycles_total = task.size * task.cycles_per_bit
            kappa_val = getattr(env.config, "kappa", 1e-27)
            idle_power_val = getattr(env.config, "user_idle_power", 0.05)
            # Compute local cost & energy
            E_local = (kappa_val * cycles_total * (user.cpu_frequency ** 2) * alpha_l)
            C_local = self.config_val(env, "price_energy_local", 1.0) * E_local

            # Offloading branch evaluation
            if alpha_l >= 1.0:
                cost = C_local
            elif target == 0:  # MEC
                mec = env.mec_servers[0]
                tx_rate = env.channel_model.calc_rate(
                    user.get_distance(mec), user.tx_power, env.config.bandwidth_mec
                )
                t_tx = task.size / max(tx_rate, 1.0)
                E_tx = user.tx_power * t_tx
                c_en = env.security_config.encryption_total_cycles_per_bit if hasattr(env, "security_config") else 890
                E_enc = (kappa_val * (task.size * c_en) * (user.cpu_frequency ** 2))
                
                f_j = mec.cpu_frequency * resource_frac
                t_exe = ((1.0 - alpha_l) * cycles_total) / max(f_j, 1.0)
                E_wait = idle_power_val * max(0.0, t_exe - (cycles_total * alpha_l / user.cpu_frequency))
                
                urgency = min(1.0 / max(task.max_delay, env.config.deadline_floor), env.config.urgency_cap)
                g_j = (mec.base_price * np.exp(urgency) * mec.credibility *
                       (((1.0 - alpha_l) * cycles_total) / env.config.cost_normalization_divisor) *
                       (1.0 + 0.1 * res_level))
                
                sec_cap = env.security_model.secure_information_capacity(
                    tx_rate, env.config.block_cipher_block_length,
                    env.security_overhead["v2i"]["p_att"]
                ) if hasattr(env, "security_model") else 0.0

                C_u = self.config_val(env, "price_energy_local", 1.0) * (E_local + E_tx + E_enc + E_wait) + g_j
                cost = C_u - env.config.varpi_1 * sec_cap
            else:  # AUX
                aux_ok, aux_idx = env._aux_feasible(user)
                if not aux_ok or aux_idx is None:
                    cost = float("inf")
                else:
                    aux = env.auxiliary_vehicles[aux_idx]
                    tx_rate = env.channel_model.calc_rate(
                        user.get_distance(aux), user.tx_power, env.config.bandwidth_v2v
                    )
                    t_tx = task.size / max(tx_rate, 1.0)
                    E_tx = user.tx_power * t_tx
                    c_en = env.security_config.encryption_total_cycles_per_bit if hasattr(env, "security_config") else 890
                    E_enc = (kappa_val * (task.size * c_en) * (user.cpu_frequency ** 2))
                    
                    f_n = aux.cpu_frequency * resource_frac
                    t_exe = ((1.0 - alpha_l) * cycles_total) / max(f_n, 1.0)
                    E_wait = idle_power_val * max(0.0, t_exe - (cycles_total * alpha_l / user.cpu_frequency))
                    
                    urgency = min(1.0 / max(task.max_delay, env.config.deadline_floor), env.config.urgency_cap)
                    load = aux.current_load + 1
                    g_n = (aux.base_price * np.log(1.0 + load) * urgency * aux.credibility *
                           (((1.0 - alpha_l) * cycles_total) / env.config.cost_normalization_divisor) *
                           (1.0 + 0.1 * res_level))
                    
                    sec_cap = env.security_model.secure_information_capacity(
                        tx_rate, env.config.block_cipher_block_length,
                        env.security_overhead["v2v"]["p_att"]
                    ) if hasattr(env, "security_model") else 0.0

                    E_av = (kappa_val * ((1.0 - alpha_l) * cycles_total) * (aux.cpu_frequency ** 2))
                    C_u = (self.config_val(env, "price_energy_local", 1.0) * (E_local + E_tx + E_enc + E_wait) +
                           self.config_val(env, "price_energy_av", 0.8) * E_av + g_n)
                    cost = C_u - env.config.varpi_1 * sec_cap

            if cost < min_cost:
                min_cost = cost
                best_action = np.array(action_tuple, dtype=np.int64)

        return best_action

    @staticmethod
    def config_val(env: Any, key: str, default: float) -> float:
        return getattr(env.config, key, default)
