"""
Generalized Benders Decomposition (GBD) Closed-Form Baseline from Paper [B] (Phase 2 - T2.4).

Ref: [B] Eq(39). Computes optimal block length N_i* given expected attacker count n^e.
"""
import numpy as np


class GBDClosedFormSecurityBaseline:
    """
    Closed-form block length decision from [B] Eq(39).

    Eq(39): N_i* = round(2^(1 / ((1 - n^e) * ln 2) - 3)) for 0 <= n^e < 1,
            N_i* = N_f for n^e >= 1.
    """

    def __init__(self, frame_length_bits: int = 64):
        self.frame_length_bits = frame_length_bits

    def compute_optimal_block_length(self, expected_attackers: float) -> int:
        """
        Compute optimal block length N_i* from expected attacker count n^e.

        Args:
            expected_attackers: n^e

        Returns:
            Optimal block length N_i* clipped to [1, N_f].
        """
        n_e = max(0.0, float(expected_attackers))
        if n_e >= 1.0:
            return self.frame_length_bits

        exponent = 1.0 / max((1.0 - n_e) * np.log(2.0), 1e-6) - 3.0
        n_i_star = int(round(2.0 ** exponent))
        return int(np.clip(n_i_star, 1, self.frame_length_bits))
