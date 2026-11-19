"""
Unit tests for Phase 4: Federated Learning Module (FedAvg)
Ref:
- [P3] Xing Chen et al., IEEE TMC 2025, Sec IV-D (FedAvg for Multi-Agent DQN).
- [P1] Si Shen et al., IEEE IoT-J 2024, Sec IV.
"""
import sys
from pathlib import Path
import copy
import torch
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithms.federated import FedAvgAggregator
from src.agents.networks import DuelingNetwork
from src.agents.policy import DRLAgent


def test_fedavg_weighted_average_calculation():
    """Kiểm tra trọng số trung bình tham số giữa các mô hình theo công thức FedAvg."""
    state_dim = 10
    total_actions = 6
    global_model = DuelingNetwork(state_dim, total_actions)

    model1 = DuelingNetwork(state_dim, total_actions)
    model2 = DuelingNetwork(state_dim, total_actions)

    # Đặt trọng số cố định: model1 = 1.0, model2 = 3.0
    with torch.no_grad():
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p1.fill_(1.0)
            p2.fill_(3.0)

    aggregator = FedAvgAggregator(global_model, aggregation_interval=5, min_participants=2)

    # Trường hợp 1: Trọng số bằng nhau [0.5, 0.5] -> kết quả = 2.0
    res = aggregator.aggregate([model1, model2], weights=[100, 100])
    for val in res.values():
        assert torch.allclose(val, torch.full_like(val, 2.0))
    assert aggregator.round_count == 1

    # Trường hợp 2: Trọng số lệch 1:3 -> kết quả = 0.25*1 + 0.75*3 = 2.5
    res2 = aggregator.aggregate([model1, model2], weights=[50, 150])
    for val in res2.values():
        assert torch.allclose(val, torch.full_like(val, 2.5))
    assert aggregator.round_count == 2


def test_fedavg_min_participants_check():
    """Nếu số lượng agent tham gia ít hơn min_participants thì bỏ qua vòng aggregate."""
    state_dim = 8
    total_actions = 4
    model = DuelingNetwork(state_dim, total_actions)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(5.0)

    aggregator = FedAvgAggregator(model, aggregation_interval=5, min_participants=3)
    # Chỉ gửi 2 models (cần >= 3)
    res = aggregator.aggregate([model, model], weights=[10, 10])

    # round_count không tăng
    assert aggregator.round_count == 0
    # global_state_dict giữ nguyên giá trị 5.0 ban đầu
    for val in res.values():
        assert torch.allclose(val, torch.full_like(val, 5.0))


def test_fedavg_broadcast_synchronization():
    """Sau khi broadcast, tất cả các agents đều nhận được cùng một bộ trọng số của global model."""
    action_dims = [4, 2, 4]
    state_dim = 10
    agent1 = DRLAgent(state_dim=state_dim, action_dim=action_dims)
    agent2 = DRLAgent(state_dim=state_dim, action_dim=action_dims)
    agent3 = DRLAgent(state_dim=state_dim, action_dim=action_dims)

    aggregator = FedAvgAggregator(agent1.q_network, aggregation_interval=1)

    # Thay đổi global_state_dict
    with torch.no_grad():
        for key in aggregator.global_state_dict:
            aggregator.global_state_dict[key].fill_(7.77)

    aggregator.broadcast([agent1, agent2, agent3])

    for agent in [agent1, agent2, agent3]:
        for p in agent.q_network.parameters():
            assert torch.allclose(p, torch.full_like(p, 7.77))
        for p in agent.target_network.parameters():
            assert torch.allclose(p, torch.full_like(p, 7.77))


def test_fedavg_should_aggregate_interval():
    """Kiểm tra hàm should_aggregate đúng theo chu kỳ K."""
    model = DuelingNetwork(6, 2)
    agg = FedAvgAggregator(model, aggregation_interval=10)

    assert agg.should_aggregate(0) is False
    assert agg.should_aggregate(5) is False
    assert agg.should_aggregate(10) is True
    assert agg.should_aggregate(15) is False
    assert agg.should_aggregate(20) is True
