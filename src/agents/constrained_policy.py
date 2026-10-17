"""
Constrained RL — PID-Lagrangian DQN (T1.3 & Stooke et al., ICML 2020).

Kế thừa khung lý thuyết CMDP và Reward-Constrained Policy Optimization (RCPO, Tessler et al., ICLR 2019):
    L(θ, λ_T, λ_E) = E[Σ r_t] − λ_T·(E[viol_T] − δ_T) − λ_E·(E[viol_E] − δ_E)

Khắc phục hiện tượng trễ pha và dao động (overshoot/oscillation) của Dual Ascent cổ điển bằng
bộ điều khiển phản hồi PID (Stooke et al., ICML 2020):
    Δ_T = viol_rate_T − δ_T
    Δ_E = avg_viol_E − δ_E

Cập nhật tích phân (I):
    I_T ← clip(I_T + K_I · Δ_T, 0, λ_max)
    I_E ← clip(I_E + K_I · Δ_E, 0, λ_max)

Nhân tử PID phản ứng tức thời (P) và giảm chấn vi phân (D):
    λ_T = clip(I_T + K_P · Δ_T + K_D · (Δ_T − Δ_T_prev), 0, λ_max)
    λ_E = clip(I_E + K_P · Δ_E + K_D · (Δ_E − Δ_E_prev), 0, λ_max)

Reward truyền vào Q-learning: r_t − λ_T·viol_T − λ_E·viol_E.
Yêu cầu: env đặt penalty_mode="external" (cờ vi phạm trả về qua info).
"""
from typing import List, Optional

from src.agents.policy import DRLAgent


class LagrangianDRLAgent(DRLAgent):
    """Dueling + Double DQN với ràng buộc xử lý bằng PID-Lagrangian (ICML 2020) & RCPO (ICLR 2019)."""

    def __init__(self, *args,
                 lambda_T_init: float = 1.0,
                 lambda_E_init: float = 1.0,
                 lambda_lr: float = 0.05,
                 lambda_max: float = 100.0,
                 delta_T: float = 0.05,
                 delta_E: float = 0.0,
                 multiplier_update_interval: int = 1,
                 use_pid: bool = True,
                 pid_kp: float = 0.1,
                 pid_ki: Optional[float] = None,
                 pid_kd: float = 0.02,
                 **kwargs):
        """
        Args:
            lambda_T_init/lambda_E_init: giá trị khởi tạo của multiplier
            lambda_lr: learning rate của dual ascent (dùng làm K_I mặc định)
            lambda_max: chặn trên để tránh phát nổ khi ràng buộc bất khả thi
            delta_T: ngưỡng vi phạm deadline cho phép (0.05 = chấp nhận 5%)
            delta_E: ngưỡng vi phạm năng lượng trung bình cho phép (J)
            multiplier_update_interval: K — chu kỳ cập nhật multiplier (với PID khuyến nghị K=1)
            use_pid: True = bật PID-Lagrangian (Stooke et al., ICML 2020); False = RCPO thuần
            pid_kp: Hệ số tỉ lệ K_P (phản ứng tức thời với vi phạm)
            pid_ki: Hệ số tích phân K_I (hội tụ triệt tiêu sai số xác lập)
            pid_kd: Hệ số vi phân K_D (giảm chấn, chống vọt lố và dập tắt dao động)
        """
        super().__init__(*args, **kwargs)
        self.lambda_T = float(lambda_T_init)
        self.lambda_E = float(lambda_E_init)
        self.lambda_lr = float(lambda_lr)
        self.lambda_max = float(lambda_max)
        self.delta_T = float(delta_T)
        self.delta_E = float(delta_E)
        self.multiplier_update_interval = int(multiplier_update_interval)

        # PID parameters (Stooke et al., ICML 2020)
        self.use_pid = bool(use_pid)
        self.pid_ki = float(lambda_lr if pid_ki is None else pid_ki)
        self.pid_kp = float(pid_kp if self.use_pid else 0.0)
        self.pid_kd = float(pid_kd if self.use_pid else 0.0)

        # Trạng thái tích phân (Integral accumulator)
        self.integral_T = float(lambda_T_init)
        self.integral_E = float(lambda_E_init)

        # Lỗi ở bước trước (cho thành phần vi phân D)
        self.prev_error_T = 0.0
        self.prev_error_E = 0.0

        # Tích lũy vi phạm giữa hai lần cập nhật multiplier
        self._viol_T_rates: List[float] = []
        self._viol_E_avgs: List[float] = []
        # Lịch sử để vẽ hình hội tụ multiplier (T1.3 bước 4)
        self.lambda_history: List[dict] = []

        mode_str = f"PID (Kp={self.pid_kp}, Ki={self.pid_ki}, Kd={self.pid_kd})" if self.use_pid else "Classical RCPO (I-only)"
        self.logger.info(
            f"Lagrangian agent [{mode_str}]: λ_T={self.lambda_T}, λ_E={self.lambda_E}, "
            f"δ_T={self.delta_T}, δ_E={self.delta_E}, K={self.multiplier_update_interval}")

    # ------------------------------------------------------------------
    # Hooks từ DRLAgent.train_episode
    # ------------------------------------------------------------------
    def _shape_reward(self, reward: float, info: dict) -> float:
        """r_shaped = r − λ_T·viol_T − λ_E·viol_E (λ tự điều chỉnh, thay hằng 100)."""
        viol_T = 1.0 if info.get("deadline_violated", False) else 0.0
        viol_E = float(info.get("energy_violation", 0.0))
        return reward - self.lambda_T * viol_T - self.lambda_E * viol_E

    def _on_episode_end(self, episode_stats: dict):
        """Cập nhật nhân tử đối ngẫu sau mỗi K episode (mặc định K=1 cho PID)."""
        self._viol_T_rates.append(episode_stats.get("viol_rate_T", 0.0))
        self._viol_E_avgs.append(episode_stats.get("avg_viol_E", 0.0))

        if self.episode_count % self.multiplier_update_interval != 0:
            return

        window = self.multiplier_update_interval
        viol_rate_T = sum(self._viol_T_rates[-window:]) / window
        avg_viol_E = sum(self._viol_E_avgs[-window:]) / window

        err_T = viol_rate_T - self.delta_T
        err_E = avg_viol_E - self.delta_E

        if self.use_pid:
            # 1. Cập nhật tích phân I: I_k = clip(I_{k-1} + K_I * e_k, 0, lambda_max)
            self.integral_T = float(min(self.lambda_max, max(0.0, self.integral_T + self.pid_ki * err_T)))
            self.integral_E = float(min(self.lambda_max, max(0.0, self.integral_E + self.pid_ki * err_E)))

            # 2. Thành phần vi phân D: d_err = e_k - e_{k-1}
            d_err_T = err_T - self.prev_error_T
            d_err_E = err_E - self.prev_error_E
            self.prev_error_T = err_T
            self.prev_error_E = err_E

            # 3. Nhân tử Lagrangian tức thời kết hợp P, I, D
            self.lambda_T = float(min(self.lambda_max, max(
                0.0, self.integral_T + self.pid_kp * err_T + self.pid_kd * d_err_T)))
            self.lambda_E = float(min(self.lambda_max, max(
                0.0, self.integral_E + self.pid_kp * err_E + self.pid_kd * d_err_E)))
        else:
            # RCPO thuần (I-only Dual Ascent cổ điển)
            self.lambda_T = float(min(self.lambda_max, max(0.0, self.lambda_T + self.lambda_lr * err_T)))
            self.lambda_E = float(min(self.lambda_max, max(0.0, self.lambda_E + self.lambda_lr * err_E)))
            self.integral_T = self.lambda_T
            self.integral_E = self.lambda_E

        # Log cho hình hội tụ multiplier (T1.3 bước 4)
        self.lambda_history.append({
            "episode": self.episode_count,
            "lambda_T": self.lambda_T,
            "lambda_E": self.lambda_E,
            "integral_T": self.integral_T,
            "integral_E": self.integral_E,
            "viol_rate_T": viol_rate_T,
            "avg_viol_E": avg_viol_E,
        })
        self.logger.info(
            f"[Lagrangian PID] ep={self.episode_count}: viol_T={viol_rate_T:.3f} "
            f"(δ={self.delta_T}) → λ_T={self.lambda_T:.3f} (I_T={self.integral_T:.3f}); "
            f"viol_E={avg_viol_E:.4f} → λ_E={self.lambda_E:.3f} (I_E={self.integral_E:.3f})")

    def get_training_stats(self) -> dict:
        stats = super().get_training_stats()
        stats.update({
            "lambda_T": self.lambda_T,
            "lambda_E": self.lambda_E,
            "integral_T": self.integral_T,
            "integral_E": self.integral_E,
        })
        return stats

