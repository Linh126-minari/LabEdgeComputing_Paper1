"""
Unit tests for Phase 3: Erlang-C Multi-core Queueing Model
Ref: [P3] Xing Chen et al., IEEE TMC 2025, Sec III-C, Eq.(8)-(12)
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env.queueing import erlang_c, mm_n_queue_delay
from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig


def test_erlang_c_bounds_and_limits():
    """Kiểm tra xác suất Erlang-C luôn thuộc [0, 1] và xử lý đúng các trường hợp biên."""
    # Khi tải bằng 0 -> xác suất chờ = 0
    assert erlang_c(n_cores=4, offered_load=0.0) == 0.0

    # Khi tải thấp (rho = 0.25) -> xác suất chờ rất nhỏ
    p_low = erlang_c(n_cores=4, offered_load=1.0)
    assert 0.0 < p_low < 0.1

    # Khi tải cao (rho = 0.9) -> xác suất chờ lớn
    p_high = erlang_c(n_cores=4, offered_load=3.6)
    assert 0.5 < p_high < 1.0

    # Khi tải rho >= 1.0 -> xác suất chờ = 1.0
    assert erlang_c(n_cores=4, offered_load=4.0) == 1.0
    assert erlang_c(n_cores=4, offered_load=5.0) == 1.0


def test_mm_n_queue_delay_calculation():
    """Kiểm tra tính toán thời gian xếp hàng trung bình M/M/n."""
    # arrival_rate = 0 -> delay = 0
    assert mm_n_queue_delay(n_cores=4, arrival_rate=0.0, service_rate_per_core=1.0) == 0.0

    # rho < 1 -> delay hữu hạn dương
    delay = mm_n_queue_delay(n_cores=4, arrival_rate=2.0, service_rate_per_core=1.0)
    assert 0.0 < delay < float('inf')

    # rho >= 1 -> delay = inf (quá tải)
    assert mm_n_queue_delay(n_cores=4, arrival_rate=4.0, service_rate_per_core=1.0) == float('inf')
    assert mm_n_queue_delay(n_cores=4, arrival_rate=5.0, service_rate_per_core=1.0) == float('inf')


def test_queueing_model_backward_compatible():
    """Mặc định queueing_model_enabled=False giữ nguyên hành vi gốc."""
    cfg = EnvironmentConfig(queueing_model_enabled=False, num_users=5, num_mec_servers=1)
    env = VehicularEdgeEnvironment(cfg)
    state, info = env.reset(seed=42)
    action = env.action_space.sample()
    next_state, reward, term, trunc, info = env.step(action)
    assert "task_delay" in info
    assert info["task_delay"] > 0


def test_queueing_model_enabled_adds_queue_delay():
    """Khi queueing_model_enabled=True, delay khi offload lên MEC bao gồm cả trễ xếp hàng."""
    cfg_no_q = EnvironmentConfig(queueing_model_enabled=False, num_users=10, num_mec_servers=1)
    cfg_q = EnvironmentConfig(queueing_model_enabled=True, mec_num_cores=2, num_users=10, num_mec_servers=1)

    env_no_q = VehicularEdgeEnvironment(cfg_no_q)
    env_q = VehicularEdgeEnvironment(cfg_q)

    # Đưa cùng seed và cùng action offload toàn phần lên MEC
    env_no_q.reset(seed=123)
    env_q.reset(seed=123)

    # action: [alpha_l=0, target=0 (MEC), resource_level=9, block_length=0]
    action_mec = np.array([0, 0, 9, 0], dtype=np.int64)
    _, _, _, _, info_no_q = env_no_q.step(action_mec)
    _, _, _, _, info_q = env_q.step(action_mec)

    if info_no_q["offload_location"].startswith("MEC") and info_q["offload_location"].startswith("MEC"):
        assert info_q["task_delay"] >= info_no_q["task_delay"]
