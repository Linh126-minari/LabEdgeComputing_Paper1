"""
Unit tests for Phase 5: Asynchronous Staleness Weighting
Ref: [P1] Si Shen et al., IEEE IoT-J 2024, Sec IV-B & Sec V-C (Staleness factor exp(-alpha * tau))
"""
import sys
from pathlib import Path
import copy
import torch
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithms.federated import AsyncFedAggregator
from src.agents.networks import DuelingNetwork


def test_staleness_exponential_decay():
    """Kiểm tra trọng số suy giảm theo hàm mũ exp(-alpha * staleness) khi thời gian gửi bị trễ."""
    state_dim = 8
    total_actions = 4
    model = DuelingNetwork(state_dim, total_actions)

    agg = AsyncFedAggregator(model, aggregation_interval=1, staleness_alpha=0.5, min_participants=2)
    # Giả lập round_count = 10
    agg.round_count = 10
    # Agent 0 vừa gửi ở round 10 -> staleness = 0
    # Agent 1 gửi ở round 8 -> staleness = 2
    # Agent 2 gửi ở round 6 -> staleness = 4
    agg.last_update_round = {0: 10, 1: 8, 2: 6}

    # Giả sử kích thước buffer của 3 xe bằng nhau = 100
    agent_ids = [0, 1, 2]
    data_sizes = [100, 100, 100]

    weights = agg.compute_staleness_weights(agent_ids, data_sizes)

    # w_0 > w_1 > w_2 vì staleness càng lớn thì trọng số càng nhỏ
    assert weights[0] > weights[1] > weights[2]
    # Tỷ lệ lý thuyết: w_0 / w_1 = exp(-0.5 * 0) / exp(-0.5 * 2) = exp(1.0) approx 2.718
    assert np.isclose(weights[0] / weights[1], np.exp(1.0), rtol=1e-3)
    # Tổng trọng số phải được chuẩn hóa về 1.0
    assert np.isclose(sum(weights), 1.0)


def test_async_aggregate_partial_participants():
    """Chỉ tập hợp các Agent tham gia (trong vùng phủ RSU) được tổng hợp và cập nhật last_update_round."""
    state_dim = 6
    total_actions = 3
    global_model = DuelingNetwork(state_dim, total_actions)

    # 3 models
    m0 = DuelingNetwork(state_dim, total_actions)
    m1 = DuelingNetwork(state_dim, total_actions)
    m2 = DuelingNetwork(state_dim, total_actions)

    with torch.no_grad():
        for p0, p1, p2 in zip(m0.parameters(), m1.parameters(), m2.parameters()):
            p0.fill_(1.0)
            p1.fill_(2.0)
            p2.fill_(9.0)  # m2 không tham gia

    agg = AsyncFedAggregator(global_model, aggregation_interval=1, staleness_alpha=0.5, min_participants=2)

    # Chỉ có agent 0 và 1 tham gia
    participating_ids = [0, 1]
    participating_models = [m0, m1]
    data_sizes = [100, 100]

    res = agg.async_aggregate(participating_ids, participating_models, data_sizes)

    assert agg.round_count == 1
    # last_update_round của agent 0 và 1 được cập nhật lên 1, agent 2 không đổi
    assert agg.last_update_round[0] == 1
    assert agg.last_update_round[1] == 1
    assert 2 not in agg.last_update_round

    # Kết quả tổng hợp chỉ tính trên m0 và m1 (1.0 và 2.0 -> 1.5)
    for key, val in res.items():
        assert torch.allclose(val, torch.full_like(val, 1.5))


def test_async_aggregate_skips_when_under_min_participants():
    """Nếu số lượng xe trong vùng phủ sóng không đủ min_participants thì bỏ qua."""
    state_dim = 4
    total_actions = 2
    model = DuelingNetwork(state_dim, total_actions)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(4.0)

    agg = AsyncFedAggregator(model, aggregation_interval=1, min_participants=3)

    # Chỉ có 2 xe tham gia (< 3)
    res = agg.async_aggregate([0, 1], [model, model], [50, 50])

    assert agg.round_count == 0
    for val in res.values():
        assert torch.allclose(val, torch.full_like(val, 4.0))
