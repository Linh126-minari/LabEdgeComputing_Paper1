"""
Phase 1 — Regression tests cho T1.1–T1.7.

Mỗi task có ít nhất một test bám đúng DoD trong kế hoạch.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig
from src.env.entities import MECServer, Task


def make_env(**overrides) -> VehicularEdgeEnvironment:
    np.random.seed(42)
    kwargs = {'episode_length': 500, 'task_arrival_rate': 5.0}
    kwargs.update(overrides)
    cfg = EnvironmentConfig(**kwargs)
    env = VehicularEdgeEnvironment(cfg)
    env.reset(seed=42)
    return env


# =========================================================================
# T1.1 — Deadline sinh từ thời gian lưu trú
# =========================================================================
def test_road_segment_length_paper_A():
    """[A] Eq(6): x = 2√(y² − z²). y=300, z=20 → x = 2√(89600) = 598.66."""
    mec = MECServer(id=0, position=np.zeros(2), cpu_frequency=1e9,
                    coverage_radius=300.0, curb_distance=20.0)
    assert abs(mec.road_segment_length() - 2 * np.sqrt(300**2 - 20**2)) < 1e-9


def test_residence_time_geometry():
    """Xe ở tâm vùng phủ, chạy thẳng v=25 → t^w = R/v = 12 s; ngoài vùng → inf."""
    env = make_env()
    mec = env.mec_servers[0]
    user = env.users[0]

    user.position = np.array(mec.position, dtype=float)
    user.velocity = 25.0
    t_w = env.residence_time_mec(user, mec)
    assert abs(t_w - mec.coverage_radius / 25.0) < 1e-6

    user.position = mec.position + np.array([mec.coverage_radius + 50.0, 0.0])
    assert env.residence_time_mec(user, mec) == np.inf


def test_residence_time_decreases_with_velocity():
    """DoD T1.1: τ tương quan ÂM với vận tốc."""
    env = make_env()
    mec = env.mec_servers[0]
    user = env.users[0]
    user.position = np.array(mec.position, dtype=float)

    times = []
    for v in [15.0, 20.0, 25.0]:
        user.velocity = v
        times.append(env.residence_time_mec(user, mec))
    assert times[0] > times[1] > times[2]


def test_tau_eff_and_omega_have_variance():
    """DoD T1.1: histogram τ_eff có phương sai khác 0; ω là feature thật."""
    env = make_env()
    taus, omegas = [], []
    for _ in range(300):
        action = env.action_space.sample()
        state, _, terminated, truncated, info = env.step(action)
        taus.append(info["tau_eff"])
        omegas.append(state[2])  # urgency feature
        if terminated or truncated:
            env.reset()
    assert np.var(taus) > 0.0, "τ_eff là hằng số — deadline động không hoạt động"
    assert np.var(omegas) > 0.0, "ω là chiều state chết"
    assert all(env.config.deadline_floor <= t <= env.config.task_deadline for t in taus)


def test_link_loss_failure_metric_exists():
    """DoD T1.1: có metric link-loss failure rate, phân biệt với deadline."""
    env = make_env()
    for _ in range(100):
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        assert info["failure_cause"] in ("none", "link_loss", "deadline")
        if terminated or truncated:
            env.reset()
    metrics = env.get_metrics()
    assert "link_loss_failure_rate" in metrics
    assert 0.0 <= metrics["link_loss_failure_rate"] <= 1.0


# =========================================================================
# T1.2 — Hàm mục tiêu duy nhất (Phương án 1)
# =========================================================================
def test_reward_single_unit_and_components():
    """DoD T1.2: reward một đơn vị (cents); log đủ thành phần; tổng khớp."""
    env = make_env()
    for _ in range(50):
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        comps = info["reward_components"]
        expected = -(comps["cost_energy"] + comps["cost_service"] + comps["cost_security"]) - comps["penalty"]
        assert abs(reward - expected) < 1e-9
        # cost = năng lượng quy tiền + giá dịch vụ + rủi ro rò rỉ (một đơn vị: cents)
        assert abs(info["task_cost"] - (info["task_cost_energy"] + info["task_cost_service"]
                                        + info["task_cost_security"])) < 1e-9
        if terminated or truncated:
            env.reset()


def test_energy_priced_via_lambda():
    """[A] Eq(22): năng lượng quy tiền qua λ^l = 1 cents/J (T1.2)."""
    env = make_env(background_users_enabled=False)
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(env.make_action(9, 0, 0))  # all-local
        if info["offload_location"] == "local":
            # all-local: không phí dịch vụ, cost = λ^l · E_user
            assert info["task_cost_service"] == pytest.approx(0.0)
            assert info["task_cost"] == pytest.approx(
                env.config.price_energy_local * info["task_energy"])
            return
        if terminated or truncated:
            env.reset()
    pytest.fail("Không có bước all-local nào")


def test_external_penalty_mode_returns_raw_reward():
    """T1.3: penalty_mode=external — env KHÔNG trừ penalty, trả cờ vi phạm."""
    env = make_env(penalty_mode="external")
    for _ in range(100):
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        assert info["reward_components"]["penalty"] == 0.0
        assert "deadline_violated" in info and "energy_violation" in info
        if terminated or truncated:
            env.reset()


# =========================================================================
# T1.3 — Lagrangian constrained RL
# =========================================================================
def test_lagrangian_multiplier_dual_ascent():
    """Multiplier tăng khi vi phạm > δ, giảm (không âm) khi vi phạm < δ."""
    pytest.importorskip("torch")
    from src.agents.constrained_policy import LagrangianDRLAgent

    agent = LagrangianDRLAgent(
        state_dim=12, action_dim=[10, 2, 10], lambda_T_init=1.0,
        lambda_lr=0.1, delta_T=0.05, multiplier_update_interval=1, device="cpu")

    lam0 = agent.lambda_T
    agent.episode_count = 1
    agent._on_episode_end({"viol_rate_T": 0.5, "avg_viol_E": 0.0})  # vi phạm 50% >> 5%
    assert agent.lambda_T > lam0

    lam1 = agent.lambda_T
    agent.episode_count = 2
    agent._on_episode_end({"viol_rate_T": 0.0, "avg_viol_E": 0.0})  # không vi phạm
    assert agent.lambda_T < lam1
    assert agent.lambda_T >= 0.0
    assert len(agent.lambda_history) == 2  # có lịch sử để vẽ hình hội tụ


def test_lagrangian_shaped_reward():
    """r_shaped = r − λ_T·viol_T − λ_E·viol_E."""
    pytest.importorskip("torch")
    from src.agents.constrained_policy import LagrangianDRLAgent

    agent = LagrangianDRLAgent(state_dim=12, action_dim=[10, 2, 10],
                               lambda_T_init=2.0, lambda_E_init=3.0, device="cpu")
    info = {"deadline_violated": True, "energy_violation": 0.5}
    assert agent._shape_reward(-1.0, info) == pytest.approx(-1.0 - 2.0 - 3.0 * 0.5)
    info = {"deadline_violated": False, "energy_violation": 0.0}
    assert agent._shape_reward(-1.0, info) == pytest.approx(-1.0)


# =========================================================================
# T1.4 — Semi-MDP
# =========================================================================
def test_semi_mdp_elapsed_time_and_decision_count():
    """DoD T1.4: elapsed_time ≥ dt; step_count đếm QUYẾT ĐỊNH; không info lỗi."""
    env = make_env(task_arrival_rate=0.5)  # λ thấp → có chờ thật
    decisions = 0
    for _ in range(30):
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        decisions += 1
        assert "error" not in info
        assert info["elapsed_time"] >= env.config.dt - 1e-12
        # elapsed là bội của dt
        ratio = info["elapsed_time"] / env.config.dt
        assert abs(ratio - round(ratio)) < 1e-6
        if terminated or truncated:
            break
    assert env.step_count == decisions


def test_semi_mdp_gamma_effective_discount():
    """T1.4: γ_eff = γ^(Δt/dt) — transition chờ lâu bị discount mạnh hơn."""
    pytest.importorskip("torch")
    import torch
    gamma = 0.95
    for elapsed_ratio, expected in [(1.0, 0.95), (10.0, 0.95 ** 10)]:
        gamma_eff = torch.pow(torch.tensor(gamma), torch.tensor(elapsed_ratio))
        assert abs(float(gamma_eff) - expected) < 1e-6


def test_gymnasium_api():
    """T1.4 bước 5: reset → (obs, info); step → 5-tuple terminated/truncated."""
    env = make_env()
    out = env.reset(seed=3)
    assert isinstance(out, tuple) and len(out) == 2
    state, info = out
    assert "action_mask" in info

    result = env.step(env.action_space.sample())
    assert len(result) == 5
    _, _, terminated, truncated, _ = result
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_episode_truncates_at_decision_budget():
    env = make_env()
    env.config.episode_length = 5
    for i in range(5):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    assert truncated and not terminated


# =========================================================================
# T1.5 — Đa user chiếm tài nguyên
# =========================================================================
def test_background_users_occupy_resources():
    """DoD T1.5: ε_n và U_j(t) có phương sai khác 0 khi bật background users."""
    env = make_env(background_users_enabled=True)
    eps_samples, util_samples = [], []
    for _ in range(300):
        state, _, terminated, truncated, _ = env.step(env.action_space.sample())
        eps_samples.append(sum(a.current_load for a in env.auxiliary_vehicles))
        # utilization của MEC là feature state (vị trí 3 + J + N)
        util_samples.append(1.0 - env.mec_cpu_available[0] / env.config.mec_cpu_frequency)
        if terminated or truncated:
            env.reset()
    assert np.var(eps_samples) > 0.0, "ε_n là hằng số — background users không chiếm tài nguyên"
    assert np.var(util_samples) > 0.0, "U_j(t) là hằng số"
    assert max(eps_samples) > 0
    assert env.background_task_count > 0


def test_aux_serves_at_most_one_user():
    """DoD T1.5: [A] §3.4 — mỗi AUX phục vụ tối đa 1 user tại một thời điểm."""
    env = make_env(background_users_enabled=True)
    for _ in range(500):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        for aux in env.auxiliary_vehicles:
            assert aux.current_load <= 1, (
                f"AUX {aux.id} phục vụ {aux.current_load} user — vi phạm [A] §3.4")
        if terminated or truncated:
            env.reset()


def test_resource_allocation_never_exceeds_pool():
    """DoD T1.5: Σβ ≤ 1 ([A] 23j) — available không âm, không vượt pool."""
    env = make_env(background_users_enabled=True)
    for _ in range(500):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert 0.0 <= env.mec_cpu_available[0] <= env.config.mec_cpu_frequency + 1e-6
        for i in range(len(env.auxiliary_vehicles)):
            assert 0.0 <= env.aux_cpu_available[i] <= env.config.aux_cpu_frequency + 1e-6
        if terminated or truncated:
            env.reset()


def test_eps_n_increases_with_num_users():
    """DoD T1.5: ε_n tăng theo num_users."""
    def avg_eps(num_users: int) -> float:
        env = make_env(num_users=num_users, background_users_enabled=True)
        samples = []
        for _ in range(200):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            samples.append(sum(a.current_load for a in env.auxiliary_vehicles))
            if terminated or truncated:
                env.reset()
        return float(np.mean(samples))

    assert avg_eps(20) > avg_eps(2)


# =========================================================================
# T1.6 — Action masking
# =========================================================================
def test_action_mask_out_of_range():
    """Ngoài tầm cả MEC lẫn AUX → chỉ α_l = 1 khả thi."""
    env = make_env(background_users_enabled=False)
    user = env.users[0]
    mec = env.mec_servers[0]
    # Đưa user ra góc xa mọi node
    user.position = np.array([0.0, 0.0])
    mec.position = np.array([900.0, 900.0])
    for aux in env.auxiliary_vehicles:
        aux.position = np.array([900.0, 900.0])

    mask = env.action_mask().reshape(tuple(env.action_space.nvec))
    assert mask[-1].any(), "α_l = 1 phải luôn khả thi"
    assert not mask[:-1].any(), "Nhánh remote phải bị mask khi ngoài tầm"


def test_action_mask_mec_in_coverage():
    env = make_env(background_users_enabled=False)
    user = env.users[0]
    user.position = np.array(env.mec_servers[0].position, dtype=float)
    for aux in env.auxiliary_vehicles:
        aux.position = user.position + np.array([500.0, 500.0])  # AUX ngoài tầm

    mask = env.action_mask().reshape(tuple(env.action_space.nvec))
    assert mask[:, 0, :].any(axis=-1).all(), "MEC trong vùng phủ phải khả thi (mọi α, β)"
    assert not mask[:5, 1, :].any(), "AUX ngoài tầm phải bị mask"
    # MEC dùng Paillier → N_i không có nghĩa → chỉ một N_i (mặc định) được mở
    assert (mask[:-1, 0, :].sum(axis=-1) == 1).all()


def test_masked_agent_never_selects_infeasible():
    """DoD T1.6: agent không bao giờ chọn nhánh remote khi bị mask."""
    pytest.importorskip("torch")
    from src.agents.policy import DRLAgent

    agent = DRLAgent(state_dim=12, action_dim=[10, 2, 10], device="cpu")
    # mask: chỉ α_l=1 khả thi
    mask = np.zeros((10, 2, 10), dtype=bool)
    mask[9, :, :] = True
    flat = mask.flatten()

    for _ in range(100):
        action = agent.select_action(np.random.rand(12).astype(np.float32),
                                     training=True, mask=flat)
        assert action[0] == 9, f"Agent chọn action bị mask: {action}"


def test_mask_flatten_order_matches_agent():
    """Thứ tự flatten của env.action_mask khớp np.ravel_multi_index của agent."""
    env = make_env()
    nvec = tuple(env.action_space.nvec)
    maskNd = env.action_mask().reshape(nvec)
    flat = env.action_mask()
    for _ in range(50):
        a = tuple(np.random.randint(n) for n in nvec)
        idx = np.ravel_multi_index(a, nvec)
        assert flat[idx] == maskNd[a]


# =========================================================================
# T1.7 — Hiệu chỉnh chế độ bảo mật
# =========================================================================
def test_eq39_degenerate_regions():
    """Nghiệm đóng [B] Eq(39): suy biến khi n^e nhỏ, bão hòa khi n^e ≥ 0.85."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from calibrate_security import n_i_star_eq39

    assert n_i_star_eq39(0.1) <= 1        # suy biến — không mã hóa
    assert n_i_star_eq39(0.7) == pytest.approx(4, abs=1)  # vùng có nghĩa
    assert n_i_star_eq39(0.85) == 64      # bão hòa
    assert n_i_star_eq39(1.4) == 64       # n^e ≥ 1 → N_f


def test_operating_point_in_meaningful_regime():
    """DoD T1.7: bộ tham số vận hành cho n^e ∈ [0.4, 1.4] và p_att TB ∈ [0.1, 0.6]."""
    from src.security.overhead_models import SecurityOverheadConfig, SecurityOverheadModel
    cfg = EnvironmentConfig()
    model = SecurityOverheadModel(SecurityOverheadConfig())

    p_atts = []
    for d in [cfg.security_range_v2i, cfg.security_range_v2v]:
        n_e = model.expected_attackers(d, cfg.attacker_density,
                                       cfg.lane_count, cfg.road_max_speed_mps)
        assert 0.4 <= n_e <= 1.4, f"n^e({d}) = {n_e} ngoài vùng có nghĩa"
        p_atts.append(1 - model.security_strength(
            cfg.encryption_block_length_bits, cfg.frame_length_bits, n_e))
    assert 0.1 <= float(np.mean(p_atts)) <= 0.6
