"""
Federated Learning Aggregation Module.
Ref:
- [P3] Xing Chen et al., IEEE TMC 2025, Sec IV-D (FedAvg for Multi-Agent DQN).
- [P1] Si Shen et al., IEEE IoT-J 2024, Sec IV (Asynchronous Staleness Weighting).
"""
import copy
from typing import List, Dict, Optional
import torch
import numpy as np

from src.utils.logger import Logger


class FedAvgAggregator:
    """
    Federated Averaging (McMahan et al., 2017) cho các DQN / Constrained DQN Agents:
    θ_global = sum_{i=1}^M (D_i / sum_j D_j) * θ_i

    D_i là số lượng mẫu (kinh nghiệm tích lũy) trong Replay Buffer của Agent i.
    """

    def __init__(self, global_model: torch.nn.Module,
                 aggregation_interval: int = 10,
                 min_participants: int = 2):
        """
        Args:
            global_model: Mô hình PyTorch toàn cục (q_network)
            aggregation_interval: Chu kỳ K episodes giữa 2 lần tổng hợp
            min_participants: Số lượng Agent tối thiểu tham gia để tiến hành aggregate
        """
        self.logger = Logger("FedAvgAggregator")
        self.global_state_dict = copy.deepcopy(global_model.state_dict())
        self.aggregation_interval = max(1, int(aggregation_interval))
        self.min_participants = max(1, int(min_participants))
        self.round_count = 0
        self.history: List[Dict] = []

    def should_aggregate(self, episode: int) -> bool:
        """Kiểm tra xem episode hiện tại có thỏa mãn chu kỳ K tổng hợp không."""
        return episode > 0 and (episode % self.aggregation_interval == 0)

    def aggregate(self, agent_models: List[torch.nn.Module],
                  weights: Optional[List[float]] = None) -> dict:
        """
        Tổng hợp trọng số từ danh sách các agent_models cục bộ.

        Args:
            agent_models: Danh sách mô hình q_network của các agent
            weights: Trọng số tỷ lệ đóng góp (None -> trọng số đều)

        Returns:
            global_state_dict sau khi được cập nhật
        """
        n = len(agent_models)
        if n < self.min_participants:
            self.logger.warning(
                f"Chỉ có {n} agents tham gia, ít hơn ngưỡng tối thiểu {self.min_participants}. Bỏ qua vòng này."
            )
            return self.global_state_dict

        if weights is None:
            norm_weights = [1.0 / n] * n
        else:
            total_w = sum(weights)
            if total_w <= 1e-12:
                norm_weights = [1.0 / n] * n
            else:
                norm_weights = [float(w) / total_w for w in weights]

        # Khởi tạo state_dict rỗng có cùng tensor shapes
        new_state = {}
        for key in self.global_state_dict:
            new_state[key] = torch.zeros_like(self.global_state_dict[key], dtype=torch.float32)
            for i, model in enumerate(agent_models):
                model_state = model.state_dict()
                new_state[key] += norm_weights[i] * model_state[key].to(new_state[key].device).float()

            # Giữ nguyên dtype ban đầu nếu là integer (ví dụ các bộ đếm nếu có)
            if self.global_state_dict[key].dtype != torch.float32:
                new_state[key] = new_state[key].to(self.global_state_dict[key].dtype)

        self.global_state_dict = new_state
        self.round_count += 1

        self.history.append({
            "round": self.round_count,
            "num_participants": n,
            "weights": norm_weights,
        })
        self.logger.info(
            f"[FedAvg] Round {self.round_count}: {n} agents aggregated. Weights: {[round(w, 3) for w in norm_weights]}"
        )
        return self.global_state_dict

    def broadcast(self, agents: list):
        """Phát tán mô hình toàn cục (global_state_dict) tới tất cả các Agent tham gia."""
        for agent in agents:
            if hasattr(agent, "q_network"):
                agent.q_network.load_state_dict(copy.deepcopy(self.global_state_dict))
            if hasattr(agent, "target_network"):
                agent.target_network.load_state_dict(copy.deepcopy(self.global_state_dict))


class AsyncFedAggregator(FedAvgAggregator):
    """
    Asynchronous Federated Aggregation với cơ chế phạt độ trễ (Staleness Weighting).
    Ref: [P1] Si Shen et al., IEEE IoT-J 2024, Sec IV-B.

    Trọng số cập nhật:
    w_i^{(r)} = (D_i * exp(-alpha * tau_i^{(r)})) / sum_j (D_j * exp(-alpha * tau_j^{(r)}))
    trong đó tau_i^{(r)} = r - r_i^{last} là độ trễ (staleness) tính theo số vòng kể từ lần
    gửi mô hình gần nhất của xe i.
    """

    def __init__(self, *args, staleness_alpha: float = 0.5, **kwargs):
        """
        Args:
            staleness_alpha: Hệ số phạt suy giảm hàm mũ alpha (mặc định 0.5 theo IoT-J 2024).
        """
        super().__init__(*args, **kwargs)
        self.staleness_alpha = float(staleness_alpha)
        self.last_update_round: Dict[int, int] = {}  # agent_id -> round index cuối cùng tham gia

    def compute_staleness_weights(self, agent_ids: List[int], data_sizes: List[int]) -> List[float]:
        """
        Tính toán trọng số bất đồng bộ có trọng số suy giảm theo độ cũ của mô hình (staleness).
        """
        raw_weights = []
        for i, aid in enumerate(agent_ids):
            # Nếu chưa từng cập nhật, độ trễ bằng toàn bộ số vòng đã diễn ra
            last_round = self.last_update_round.get(aid, 0)
            staleness = max(0, self.round_count - last_round)
            decay = float(np.exp(-self.staleness_alpha * staleness))
            d_size = max(1, data_sizes[i])
            raw_weights.append(d_size * decay)

        total = sum(raw_weights)
        if total <= 1e-12:
            return [1.0 / len(raw_weights)] * len(raw_weights)
        return [float(w) / total for w in raw_weights]

    def async_aggregate(self, participating_agent_ids: List[int],
                        agent_models: List[torch.nn.Module],
                        data_sizes: List[int]) -> dict:
        """
        Thực hiện tổng hợp bất đồng bộ chỉ từ tập hợp các Agent thực sự gửi mô hình (ví dụ: nằm trong vùng RSU).
        """
        if len(participating_agent_ids) < self.min_participants:
            self.logger.warning(
                f"[AsyncFed] Chỉ có {len(participating_agent_ids)} agents tham gia (< {self.min_participants}). Bỏ qua."
            )
            return self.global_state_dict

        weights = self.compute_staleness_weights(participating_agent_ids, data_sizes)
        # Sử dụng aggregate của lớp cha với weights đã nhân hệ số suy giảm
        res = self.aggregate(agent_models, weights=weights)

        # Cập nhật round index mới nhất cho các agent vừa tham gia
        for aid in participating_agent_ids:
            self.last_update_round[aid] = self.round_count

        return res

