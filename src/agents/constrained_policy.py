"""
Constrained RL — Lagrangian-DQN (T1.3).

Thay thế penalty tay λ_p = 100 bằng multiplier HỌC ĐƯỢC (kiểu RCPO):

    L(θ, λ_T, λ_E) = E[Σ r_t] − λ_T·(E[viol_T] − δ_T) − λ_E·(E[viol_E] − δ_E)

Ánh xạ ràng buộc:
    viol_T = 1[T_u > τ_u]        — [A] (23e): deadline là ràng buộc CỨNG
    viol_E = max(0, E_u − E_max) — [B] (C3): E_max = 0.4 J

Reward truyền vào Q-learning: r_t − λ_T·viol_T − λ_E·viol_E, với λ tự điều
chỉnh bằng dual ascent (chậm hơn policy — cập nhật mỗi K episode, lr_λ << lr_θ):

    λ_T ← max(0, λ_T + lr_λ·(viol_rate_T − δ_T))
    λ_E ← max(0, λ_E + lr_λ·(avg_viol_E − δ_E))

Yêu cầu: env đặt penalty_mode="external" (env KHÔNG tự trừ penalty; cờ vi phạm
trả về qua info["deadline_violated"], info["energy_violation"]).

Baseline so sánh bắt buộc (T1.3 bước 5): cùng agent với λ_p cố định ∈ {1, 10, 100}
→ dùng DRLAgent chuẩn + env penalty_mode="fixed", deadline_penalty ∈ {1, 10, 100}.
"""
from typing import List

from src.agents.policy import DRLAgent


class LagrangianDRLAgent(DRLAgent):
    """Dueling + Double DQN với ràng buộc xử lý bằng Lagrangian relaxation (T1.3)."""

    def __init__(self, *args,
                 lambda_T_init: float = 1.0,
                 lambda_E_init: float = 1.0,
                 lambda_lr: float = 0.05,
                 lambda_max: float = 100.0,
                 delta_T: float = 0.05,
                 delta_E: float = 0.0,
                 multiplier_update_interval: int = 10,
                 **kwargs):
        """
        Args:
            lambda_T_init/lambda_E_init: giá trị khởi tạo của multiplier
            lambda_lr: learning rate của dual ascent (lr_λ << lr_policy)
            lambda_max: chặn trên để tránh phát nổ khi ràng buộc bất khả thi
            delta_T: ngưỡng vi phạm deadline cho phép (0.05 = chấp nhận 5%)
            delta_E: ngưỡng vi phạm năng lượng trung bình cho phép (J)
            multiplier_update_interval: K — cập nhật multiplier mỗi K episode
        """
        super().__init__(*args, **kwargs)
        self.lambda_T = float(lambda_T_init)
        self.lambda_E = float(lambda_E_init)
        self.lambda_lr = float(lambda_lr)
        self.lambda_max = float(lambda_max)
        self.delta_T = float(delta_T)
        self.delta_E = float(delta_E)
        self.multiplier_update_interval = int(multiplier_update_interval)

        # Tích lũy vi phạm giữa hai lần cập nhật multiplier
        self._viol_T_rates: List[float] = []
        self._viol_E_avgs: List[float] = []
        # Lịch sử để vẽ hình hội tụ multiplier (T1.3 bước 4)
        self.lambda_history: List[dict] = []

        self.logger.info(
            f"Lagrangian agent: λ_T={self.lambda_T}, λ_E={self.lambda_E}, "
            f"lr_λ={self.lambda_lr}, δ_T={self.delta_T}, δ_E={self.delta_E}, "
            f"K={self.multiplier_update_interval}")

    # ------------------------------------------------------------------
    # Hooks từ DRLAgent.train_episode
    # ------------------------------------------------------------------
    def _shape_reward(self, reward: float, info: dict) -> float:
        """r_shaped = r − λ_T·viol_T − λ_E·viol_E (λ tự điều chỉnh, thay hằng 100)."""
        viol_T = 1.0 if info.get("deadline_violated", False) else 0.0
        viol_E = float(info.get("energy_violation", 0.0))
        return reward - self.lambda_T * viol_T - self.lambda_E * viol_E

    def _on_episode_end(self, episode_stats: dict):
        """Dual ascent — chậm hơn policy: cập nhật mỗi K episode (T1.3 bước 2)."""
        self._viol_T_rates.append(episode_stats.get("viol_rate_T", 0.0))
        self._viol_E_avgs.append(episode_stats.get("avg_viol_E", 0.0))

        if self.episode_count % self.multiplier_update_interval != 0:
            return

        window = self.multiplier_update_interval
        viol_rate_T = sum(self._viol_T_rates[-window:]) / window
        avg_viol_E = sum(self._viol_E_avgs[-window:]) / window

        self.lambda_T = float(min(self.lambda_max, max(
            0.0, self.lambda_T + self.lambda_lr * (viol_rate_T - self.delta_T))))
        self.lambda_E = float(min(self.lambda_max, max(
            0.0, self.lambda_E + self.lambda_lr * (avg_viol_E - self.delta_E))))

        # Log cho hình hội tụ multiplier (T1.3 bước 4)
        self.lambda_history.append({
            "episode": self.episode_count,
            "lambda_T": self.lambda_T,
            "lambda_E": self.lambda_E,
            "viol_rate_T": viol_rate_T,
            "avg_viol_E": avg_viol_E,
        })
        self.logger.info(
            f"[Lagrangian] ep={self.episode_count}: viol_T={viol_rate_T:.3f} "
            f"(δ={self.delta_T}) → λ_T={self.lambda_T:.3f}; "
            f"viol_E={avg_viol_E:.4f} → λ_E={self.lambda_E:.3f}")

    def get_training_stats(self) -> dict:
        stats = super().get_training_stats()
        stats.update({"lambda_T": self.lambda_T, "lambda_E": self.lambda_E})
        return stats
