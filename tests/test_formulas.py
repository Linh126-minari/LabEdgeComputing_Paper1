"""
T0.12 — Regression test cho từng công thức, so với giá trị TÍNH TAY từ [A]/[B].

Mục đích: đóng băng các công thức đã sửa trong Phase 0 để không tái phát lỗi.
Cố ý đưa lại tham số sai (vd. rho = 2e-6) sẽ làm test đỏ.

Nguồn:
  [A] Xue, Yao, Wang — Ad Hoc Networks 161 (2024) 103545
  [B] Xiao et al. — IEEE TWC 22(12), 2023
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.channel import ChannelModel
from src.env.entities import AuxVehicle
from src.security.overhead_models import SecurityOverheadConfig, SecurityOverheadModel


@pytest.fixture
def model_paper_b():
    """SecurityOverheadModel với đúng bộ tham số của [B] Table I."""
    return SecurityOverheadModel(SecurityOverheadConfig(
        attacker_density=1e-4,   # rho, [B] Table I
        lane_count=8,            # N_l, [B] Table I
        max_speed_mps=24,        # v_max, [B] Table I
        frame_length_bits=64,    # N_f, [B] Table I
    ))


# ---------------------------------------------------------------------------
# [B] Eq(8) — expected attackers (T0.2)
# ---------------------------------------------------------------------------
def test_expected_attackers_paper_B(model_paper_b):
    """[B] Eq(8): n^e = pi d^2 rho gamma, gamma = N_l/(3 v_max).

    Tính tay: gamma = 8/72 = 0.11111
    n^e(200) = pi * 200^2 * 1e-4 * 0.11111 = 1.396
    """
    n_e = model_paper_b.expected_attackers(200.0)
    assert abs(n_e - 1.396) < 0.01
    assert 1.3 <= n_e <= 1.5  # DoD T0.2


def test_expected_attackers_at_300m(model_paper_b):
    """n^e(300) = pi * 300^2 * 1e-4 * 0.11111 = 3.1416 (~ pi)."""
    assert abs(model_paper_b.expected_attackers(300.0) - 3.1416) < 0.01


def test_attacker_density_is_ratio_not_spatial():
    """T0.2 DoD: mật độ mặc định phải là 1e-4 (tỉ lệ), KHÔNG phải 2e-6 (1/m²)."""
    cfg = SecurityOverheadConfig()
    assert cfg.attacker_density == pytest.approx(1e-4)
    assert cfg.attacker_density != pytest.approx(2e-6)


# ---------------------------------------------------------------------------
# [B] Eq(6)-(7) — frame vulnerability & security strength (T0.3)
# ---------------------------------------------------------------------------
def test_frame_vulnerability_paper_B(model_paper_b):
    """[B] Eq(6): Phi = (N_f - N_i)/N_f. N_i=32, N_f=64 -> Phi = 0.5."""
    assert abs(model_paper_b.frame_vulnerability(32, 64) - 0.5) < 1e-9


def test_security_strength_paper_B(model_paper_b):
    """[B] Eq(7): S = (N_i/N_f)^(n^e). N_i=32, N_f=64, n^e=1 -> S = 0.5."""
    assert abs(model_paper_b.security_strength(32, 64, 1.0) - 0.5) < 1e-6


def test_block_length_defaults_decoupled_from_paillier():
    """T0.3 DoD: N_f=64 (block cipher [B]), KHÔNG phải 8192; N_i=32, KHÔNG phải 2048."""
    cfg = SecurityOverheadConfig()
    assert cfg.frame_length_bits == 64.0
    assert cfg.default_block_length_bits == 32.0


# ---------------------------------------------------------------------------
# [B] Eq(9) — secure information capacity (T0.3, T0.8)
# ---------------------------------------------------------------------------
def test_secure_capacity_paper_B(model_paper_b):
    """[B] Eq(9): U = R/(8N)·log2(8N)·S.

    Tính tay: R=1e7, N=32, S=1 -> 1e7/256 · log2(256) = 39062.5 · 8 = 312500.
    """
    u = model_paper_b.secure_information_capacity(1e7, 32, 1.0)
    assert abs(u - 312500.0) < 1.0


def test_secure_capacity_scale_is_1e5_not_1e3(model_paper_b):
    """T0.3 DoD: với R=1e7, N_i=32, S=1, U ở bậc 1e5 (KHÔNG phải 1e3 như khi
    N_i=2048)."""
    u = model_paper_b.secure_information_capacity(1e7, 32, 1.0)
    assert 1e5 <= u < 1e6


def test_security_cost_zero_for_local_and_he_and_monotone_in_block_length():
    """Phân vai an ninh: local & MEC-Paillier không rủi ro; nhánh AUX rủi ro giảm khi N_i tăng."""
    from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig

    env = VehicularEdgeEnvironment(EnvironmentConfig(episode_length=10, attacker_model="expected",
                                                     background_users_enabled=False))
    env.reset(seed=0)
    user = env.users[0]
    user.position = np.array(env.mec_servers[0].position, dtype=float)
    env.auxiliary_vehicles[0].position = user.position + np.array([50.0, 0.0])
    L, R = env.config.alpha_levels, env.config.action_levels
    assert env.evaluate_action(env.make_action(L - 1, 0, 0))["cost_security"] == 0.0
    mec = env.evaluate_action(env.make_action(0, 0, 4))
    assert mec["offload_location"] == "MEC_0" and mec["cost_security"] == 0.0 and mec["bits_at_risk"] == 0.0
    risks = [env.evaluate_action(env.make_action(0, 1, 4, k))["bits_at_risk"]
             for k in range(len(env.config.block_length_options))]
    assert all(r1 >= r2 for r1, r2 in zip(risks, risks[1:])) and risks[0] > risks[-1] == 0.0

def test_aux_price_nonzero_when_idle():
    """[A] §3.5 + MỞ RỘNG T0.7: eps_n >= 1 nên g_n > 0 ngay cả khi AUX rảnh."""
    aux = AuxVehicle(id=0, position=np.array([0.0, 0.0]), cpu_frequency=0.5e9,
                     base_price=0.5, credibility=0.8, current_load=0)
    price = aux.get_dynamic_price(task_urgency=1.0)
    # b_n·ln(1+1)·1·g_n0 = 0.8·ln2·0.5 = 0.27726
    assert price > 0
    assert abs(price - 0.8 * np.log(2.0) * 0.5) < 1e-9


def test_mec_price_exponential_in_urgency():
    """[A] §3.5: g_j ∝ exp(ω), ω = 1/(1+τ_eff) ∈ (0,1)."""
    from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig
    from src.env.entities import Task

    env = VehicularEdgeEnvironment(EnvironmentConfig(episode_length=10))
    mec = env.mec_servers[0]
    task = Task(size=1e5, cycles_per_bit=1000, max_delay=1.0)
    c1 = env._compute_mec_cost(mec, task, resource_level=0, tau_eff=1.0)   # ω = 0.5
    c2 = env._compute_mec_cost(mec, task, resource_level=0, tau_eff=0.25)  # ω = 0.8
    assert abs(c2 / c1 - np.exp(0.3)) < 1e-6


def test_urgency_in_open_unit_interval():
    """[A] §3.5 yêu cầu ω ∈ (0,1); τ = 1 s → ω = 0.5 (giá trị [A] dùng trong mô phỏng)."""
    from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig

    env = VehicularEdgeEnvironment(EnvironmentConfig(episode_length=10))
    assert env._price_urgency(1.0) == pytest.approx(0.5)
    for tau in (0.0, 0.05, 0.3, 1.0, 100.0):
        assert 0.0 < env._price_urgency(tau) < 1.0


# ---------------------------------------------------------------------------
# Channel model (T0.5)
# ---------------------------------------------------------------------------
def test_channel_rate_at_100m_in_range():
    """T0.5 DoD: rate tại d=100 m với B=1 MHz thuộc [1, 20] Mbps."""
    ch = ChannelModel()
    rate = ch.calc_rate(100.0, tx_power=0.2, bandwidth=1e6)
    assert 1e6 <= rate <= 20e6


def test_channel_rate_monotone_decreasing():
    """T0.5 DoD: rate đơn điệu giảm theo khoảng cách."""
    ch = ChannelModel()
    dists = [10, 50, 100, 200, 300, 500]
    rates = [ch.calc_rate(d, 0.2, 1e6) for d in dists]
    assert all(r1 > r2 for r1, r2 in zip(rates, rates[1:]))


def test_channel_snr_calibration_targets():
    """T0.5: SNR(100m) ~ 25 dB; SNR(300m) trong [3, 12] dB."""
    ch = ChannelModel()
    snr_100_db = 10 * np.log10(ch.calc_snr(100.0, 0.2))
    snr_300_db = 10 * np.log10(ch.calc_snr(300.0, 0.2))
    assert 20.0 <= snr_100_db <= 32.0
    assert 3.0 <= snr_300_db <= 12.0


def test_channel_no_dkm_hack():
    """T0.5 DoD: không còn hack d_km trong channel.py."""
    src = (Path(__file__).resolve().parents[1] / "src" / "env" / "channel.py").read_text(
        encoding="utf-8")
    assert "d_km = " not in src


# ---------------------------------------------------------------------------
# Energy (T0.6)
# ---------------------------------------------------------------------------
def test_idle_power_is_used():
    """T0.6 DoD: idle_power được SỬ DỤNG — khi offload hết (alpha_l=0),
    tổng năng lượng với p_w=0.05 phải LỚN HƠN khi p_w=0 (cùng seed)."""
    from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig

    def total_energy(idle_power: float) -> float:
        np.random.seed(123)
        env = VehicularEdgeEnvironment(EnvironmentConfig(
            episode_length=50, user_idle_power=idle_power,
            task_arrival_rate=5.0,  # đảm bảo luôn có task
            background_users_enabled=False,  # tất định giữa hai lần chạy
        ))
        env.reset(seed=123)
        total = 0.0
        for _ in range(20):
            # Đặt user 0 ngay tại MEC 0 để chắc chắn trong vùng phủ (tất định)
            env.users[0].position = np.array(env.mec_servers[0].position, dtype=float)
            # action: alpha_l_level=0 (offload hết), target MEC, resource cao
            _, _, terminated, truncated, info = env.step(np.array([0, 0, 9]))
            total += info.get("task_energy", 0.0)
            if terminated or truncated:
                break
        return total

    assert total_energy(0.05) > total_energy(0.0)


# ---------------------------------------------------------------------------
# Task generation (T0.4)
# ---------------------------------------------------------------------------
def test_task_sizes_in_paper_A_range():
    """T0.4 DoD: 1000 task sinh ra đều nằm trong [102400, 1024000] bits."""
    from src.env.traffic import TaskGenerator, TrafficConfig

    gen = TaskGenerator(TrafficConfig())
    for _ in range(1000):
        task = gen.generate_task()
        assert 102400 <= task.size <= 1024000


def test_task_generator_rejects_wrong_units():
    """T0.4: assertion bắt lỗi đơn vị (KB thay vì bits)."""
    from src.env.traffic import TaskGenerator, TrafficConfig

    with pytest.raises(AssertionError):
        TaskGenerator(TrafficConfig(task_size_min=100, task_size_max=1000))  # KB!


def test_user_has_no_generate_task():
    """T0.4 DoD: User.generate_task() đã bị xóa — chỉ TaskGenerator sinh task."""
    from src.env.entities import User
    assert not hasattr(User, "generate_task")


# ---------------------------------------------------------------------------
# Double DQN (T0.11)
# ---------------------------------------------------------------------------
def test_double_dqn_target_differs_from_plain_dqn():
    """T0.11 DoD: target Q của Double DQN khác target của DQN thuần
    (chọn action bằng online net vs max của target net)."""
    torch = pytest.importorskip("torch")
    from src.agents.policy import DRLAgent

    def make_agent(use_double):
        np.random.seed(7)
        torch.manual_seed(7)
        return DRLAgent(state_dim=12, action_dim=[10, 2, 10],
                        learning_starts=32, batch_size=8,
                        use_double_dqn=use_double, device="cpu")

    def fill_and_get_loss(agent):
        np.random.seed(11)
        for _ in range(64):
            s = np.random.rand(12).astype(np.float32)
            a = np.array([np.random.randint(10), np.random.randint(2), np.random.randint(10)])
            agent.remember(s, a, np.random.randn(), np.random.rand(12).astype(np.float32), False)
        # Làm online net khác target net để hai target khác nhau
        with torch.no_grad():
            for p in agent.q_network.parameters():
                p.add_(0.5 * torch.randn_like(p))
        torch.manual_seed(3)
        return agent.train_step()

    loss_double = fill_and_get_loss(make_agent(True))
    loss_plain = fill_and_get_loss(make_agent(False))
    assert loss_double != pytest.approx(loss_plain)


def test_warmup_blocks_training():
    """T0.11: không train khi buffer < learning_starts."""
    pytest.importorskip("torch")
    from src.agents.policy import DRLAgent

    agent = DRLAgent(state_dim=12, action_dim=[10, 2, 10],
                     learning_starts=2000, batch_size=8, device="cpu")
    for _ in range(100):  # 100 < 2000
        s = np.random.rand(12).astype(np.float32)
        agent.remember(s, np.array([0, 0, 0]), 0.0, s, False)
    assert agent.train_step() == 0.0
