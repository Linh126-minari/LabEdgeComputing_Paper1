"""
Unit & Integration Tests for PID-Lagrangian Constrained DRL (Stooke et al., ICML 2020 & Tessler et al., ICLR 2019).
"""
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.constrained_policy import LagrangianDRLAgent


def test_pid_lagrangian_responsiveness():
    """Kiểm tra tính phản ứng nhanh (Proportional response) khi có vi phạm đột biến."""
    # 1. Agent RCPO thuần (I-only)
    agent_rcpo = LagrangianDRLAgent(
        state_dim=13, action_dim=[10, 2, 10, 4], lambda_T_init=1.0,
        lambda_lr=0.05, delta_T=0.05, multiplier_update_interval=1,
        use_pid=False, device="cpu"
    )

    # 2. Agent PID-Lagrangian (có P và D)
    agent_pid = LagrangianDRLAgent(
        state_dim=13, action_dim=[10, 2, 10, 4], lambda_T_init=1.0,
        lambda_lr=0.05, delta_T=0.05, multiplier_update_interval=1,
        use_pid=True, pid_kp=0.2, pid_ki=0.05, pid_kd=0.05, device="cpu"
    )

    # Episode 1: Vi phạm đột biến 40% (ngưỡng cho phép 5%)
    agent_rcpo.episode_count = 1
    agent_rcpo._on_episode_end({"viol_rate_T": 0.40, "avg_viol_E": 0.0})

    agent_pid.episode_count = 1
    agent_pid._on_episode_end({"viol_rate_T": 0.40, "avg_viol_E": 0.0})

    # Nhờ thành phần P và D, PID phản ứng mạnh mẽ hơn để chặn đứng vi phạm
    assert agent_pid.lambda_T > agent_rcpo.lambda_T > 1.0


def test_pid_lagrangian_damping_prevents_overshoot():
    """Kiểm tra thành phần vi phân D (Derivative damping) giảm phạt khi vi phạm đang hạ nhiệt."""
    agent = LagrangianDRLAgent(
        state_dim=13, action_dim=[10, 2, 10, 4], lambda_T_init=1.0,
        lambda_lr=0.05, delta_T=0.05, multiplier_update_interval=1,
        use_pid=True, pid_kp=0.1, pid_ki=0.05, pid_kd=0.1, device="cpu"
    )

    # Episode 1: Vi phạm 50%
    agent.episode_count = 1
    agent._on_episode_end({"viol_rate_T": 0.50, "avg_viol_E": 0.0})
    lam_ep1 = agent.lambda_T

    # Episode 2: Vi phạm giảm mạnh xuống 10% (vẫn lớn hơn delta=5%, nhưng đang giảm nhanh)
    agent.episode_count = 2
    agent._on_episode_end({"viol_rate_T": 0.10, "avg_viol_E": 0.0})
    lam_ep2 = agent.lambda_T

    # Damping D < 0 vì error giảm từ 0.45 xuống 0.05 -> lam_ep2 phải hạ nhiệt so với ep1
    assert lam_ep2 < lam_ep1
    assert agent.integral_T > 1.0  # Tích phân I vẫn tích lũy đúng


def test_pid_lagrangian_backward_compatibility():
    """Kiểm tra chế độ tắt PID quay lại chính xác RCPO cổ điển."""
    agent = LagrangianDRLAgent(
        state_dim=13, action_dim=[10, 2, 10, 4], lambda_T_init=1.0,
        lambda_lr=0.05, delta_T=0.05, multiplier_update_interval=1,
        use_pid=False, device="cpu"
    )
    agent.episode_count = 1
    agent._on_episode_end({"viol_rate_T": 0.25, "avg_viol_E": 0.0})
    # Trong RCPO: lambda = 1.0 + 0.05 * (0.25 - 0.05) = 1.01
    assert agent.lambda_T == pytest.approx(1.0 + 0.05 * 0.20)
