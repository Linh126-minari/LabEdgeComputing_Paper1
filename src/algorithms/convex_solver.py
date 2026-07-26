import cvxpy as cp
import numpy as np


class CMSOSolver:
    """
    Convex sub-problem solver for local/offloading split.

    This version integrates:
    - attacker vehicles via aggregated attack probability,
    - security-level dependent encryption overhead,
    - effective transmission-rate and transmission-energy penalties.
    """

    def _get_param(self, source, key, idx=None, default=None):
        """Read parameter from dict-like/object/tuple for backward compatibility."""
        if isinstance(source, dict):
            return source.get(key, default)
        if hasattr(source, key):
            return getattr(source, key)
        if idx is not None and isinstance(source, (list, tuple)) and idx < len(source):
            return source[idx]
        return default

    def _attack_probability(self, security_params):
        """
        Paper attacker model (Eq. 6-8):
        - n_i^e = pi * d_i^2 * rho * (N_l / (3 * v_max))
        - S_i = (N_i / N_f)^(n_i^e)
        - p_att = 1 - S_i
        """
        n_f = max(float(self._get_param(security_params, "frame_length_bits", default=8192.0)), 1.0)
        n_i = float(self._get_param(security_params, "block_length_bits", default=2048.0))
        n_i = float(np.clip(n_i, 1.0, n_f))

        d_i = max(float(self._get_param(security_params, "coverage_radius", default=250.0)), 0.0)
        rho = max(float(self._get_param(security_params, "attacker_density", default=2e-6)), 0.0)
        n_l = max(float(self._get_param(security_params, "lane_count", default=3.0)), 1e-6)
        v_max = max(float(self._get_param(security_params, "road_max_speed_mps", default=25.0)), 1e-6)

        gamma = n_l / (3.0 * v_max)
        n_attackers = float(np.pi * (d_i ** 2) * rho * gamma)

        num_attackers = self._get_param(security_params, "num_attackers", default=None)
        if num_attackers is not None:
            n_attackers = min(n_attackers, max(float(num_attackers), 0.0))

        security_strength = float(np.clip((n_i / n_f) ** max(n_attackers, 0.0), 0.0, 1.0))
        return float(np.clip(1.0 - security_strength, 0.0, 1.0))

    def solve(self, task_params, user_params, av_params, prices=None, security_params=None):
        """
        Solve convex split-ratio problem for one user-task pair.

        Expected parameter keys (dict/object) or legacy tuple order:
        - task_params: size_bits, cycles_per_bit, deadline_s, enc_cycles_per_bit
        - user_params: cpu_freq, lambda_local, lambda_offload, tx_power, wait_power, kappa
        - av_params: cpu_freq, uplink_rate, service_price
                - security_params: security_level, attack_rate_penalty, attack_energy_penalty,
                    frame_length_bits, block_length_bits, coverage_radius, attacker_density,
                    lane_count, road_max_speed_mps, num_attackers, security_risk_weight,
                    secure_capacity_weight
        """
        _ = prices

        # Decision variable: local processing ratio. Offloading ratio = 1 - alpha_l.
        alpha_l = cp.Variable(name="alpha_l")
        alpha_n = 1.0 - alpha_l

        # Task parameters
        s_m = float(self._get_param(task_params, "size_bits", idx=0, default=1.0))
        c_m = float(self._get_param(task_params, "cycles_per_bit", idx=1, default=1.0))
        t_m = max(float(self._get_param(task_params, "deadline_s", idx=2, default=1.0)), 1e-6)
        c_en = float(self._get_param(task_params, "enc_cycles_per_bit", idx=3, default=800.0))

        # User parameters
        f_m = max(float(self._get_param(user_params, "cpu_freq", idx=0, default=2e9)), 1.0)
        lambda_l = float(self._get_param(user_params, "lambda_local", idx=1, default=1.0))
        lambda_n = float(self._get_param(user_params, "lambda_offload", idx=2, default=1.0))
        p_m = float(self._get_param(user_params, "tx_power", idx=3, default=0.5))
        p_w = float(self._get_param(user_params, "wait_power", idx=4, default=0.2))
        kappa = float(self._get_param(user_params, "kappa", default=1e-27))

        # AV/MEC node parameters
        f_n = max(float(self._get_param(av_params, "cpu_freq", idx=0, default=10e9)), 1.0)
        r_up_nom = max(float(self._get_param(av_params, "uplink_rate", idx=1, default=1e6)), 1.0)
        g_n = float(self._get_param(av_params, "service_price", idx=2, default=0.0))

        # Security and attacker parameters
        if security_params is None:
            security_params = {}
        sec_level = max(float(self._get_param(security_params, "security_level", default=1.0)), 1e-6)
        eta_r = float(self._get_param(security_params, "attack_rate_penalty", default=0.35))
        eta_e = float(self._get_param(security_params, "attack_energy_penalty", default=0.45))
        rho_sec = float(self._get_param(security_params, "security_risk_weight", default=0.0))
        w_secure_capacity = float(self._get_param(security_params, "secure_capacity_weight", default=0.0))

        n_f = max(float(self._get_param(security_params, "frame_length_bits", default=8192.0)), 1.0)
        n_i = float(np.clip(self._get_param(security_params, "block_length_bits", default=2048.0), 1.0, n_f))

        p_att = self._attack_probability(security_params)
        r_eff = max(r_up_nom * (1.0 - eta_r * p_att), 1.0)
        security_strength = float(np.clip(1.0 - p_att, 0.0, 1.0))
        secure_capacity = float((r_up_nom / (8.0 * n_i)) * np.log2(8.0 * n_i) * security_strength)

        # Local-energy term: convex cubic form in alpha_l.
        e_local = kappa * cp.power(alpha_l * s_m * c_m, 3) / (t_m ** 2)

        # Security-aware offloading terms (all affine in alpha_n).
        e_enc = (kappa * (f_m ** 2) * alpha_n * s_m * c_en) / sec_level
        t_tx = alpha_n * s_m / r_eff
        e_tx = p_m * t_tx * (1.0 + eta_e * p_att)
        e_wait = p_w * alpha_n * s_m * c_m / f_n
        service_cost = g_n * alpha_n * s_m * c_m
        security_risk_cost = rho_sec * p_att * alpha_n
        secure_capacity_benefit = w_secure_capacity * secure_capacity * alpha_n

        objective = cp.Minimize(
            lambda_l * e_local
            + lambda_n * (e_enc + e_tx + e_wait)
            + service_cost
            + security_risk_cost
            - secure_capacity_benefit
        )

        # Delay constraints with security overhead.
        t_local = alpha_l * s_m * c_m / f_m
        t_offload = t_tx + (alpha_n * s_m * c_m / f_n) + (alpha_n * s_m * c_en / (f_m * sec_level))

        constraints = [
            alpha_l >= 0.0,
            alpha_l <= 1.0,
            t_local <= t_m,
            t_offload <= t_m,
            t_local + t_offload <= t_m,
        ]

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve()
            if alpha_l.value is None:
                return 0.5
            return float(np.clip(alpha_l.value, 0.0, 1.0))
        except cp.SolverError:
            return 0.5