"""
Môi trường Vehicular Edge Computing — semi-MDP, deadline động, đa user.

Trạng thái sau Phase 1:
- T1.1: deadline hiệu dụng τ_u = min(τ_app, t^w_MEC, t^w_V2V) sinh từ thời gian
  lưu trú ([A] Eq(6)-(8), [B] Eq(2)); metric link-loss failure.
- T1.2: hàm mục tiêu duy nhất (Phương án 1) — chi phí kinh tế của [A] Eq(22)
  (một đơn vị: cents) trừ ϖ₁·U của [B]; deadline và energy là RÀNG BUỘC.
- T1.4: semi-MDP — step() = một QUYẾT ĐỊNH; thời gian chờ task được trả về qua
  info["elapsed_time"] để agent discount γ^(Δt/dt); Gymnasium API
  (terminated/truncated).
- T1.5: users[1:] chạy Algorithm 1 của [A] và THỰC SỰ chiếm tài nguyên;
  ràng buộc Σβ ≤ 1 ([A] 23j) và mỗi AUX ≤ 1 user ([A] §3.4) được thực thi.
- T1.6: action_mask() — chặn hành động không khả thi.
- T1.7: d_i (bán kính vùng phát của người gửi, [B] Eq(8)) hiệu chỉnh riêng cho
  V2I/V2V để n^e nằm trong vùng không suy biến.

Nguồn:
  [A] Xue, Yao, Wang — Ad Hoc Networks 161 (2024) 103545
  [B] Xiao et al. — IEEE TWC 22(12), 2023
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List
from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces

from src.env.entities import User, MECServer, AuxVehicle, AttackVehicle, Task, ComputingNode
from src.env.channel import ChannelModel
from src.env.mobility import LinearMobility, TraceMobility
from src.security.crypto_engine import SecurityModule
from src.security.overhead_models import SecurityOverheadConfig, SecurityOverheadModel
from src.utils.logger import Logger


@dataclass
class EnvironmentConfig:
    """
    Cấu hình môi trường.

    QUY TẮC (T0.1): giá trị mặc định ở đây phải khớp configs/params_master.yaml
    (single source of truth). Khi chạy qua Hydra, mọi giá trị được nạp từ đó.
    """
    # Topology — Ref: [A] Fig.3 (U/J/N = 10/1/3)
    num_users: int = 10
    num_mec_servers: int = 1
    num_auxiliary_vehicles: int = 3
    num_attack_vehicles: int = 11  # attacker vật lý: 11 xe/1 km² → E[nᵉ] trong 150 m ≈ 0.785 = kỳ vọng [B] Eq(8)

    # Khu vực mô phỏng — MỞ RỘNG (không có trong [A]/[B])
    sim_area_x: float = 1000.0  # meters
    sim_area_y: float = 1000.0  # meters

    # Task generation — Ref: [A] Section 6
    task_arrival_rate: float = 0.5  # Poisson λ (tasks/second)
    task_size_min: float = 102400.0   # bits (=100 kbit), [A] Section 6 (T0.4)
    task_size_max: float = 1024000.0  # bits (=1000 kbit), [A] Section 6 (T0.4)
    task_cycles_per_bit: float = 1000  # cycles/bit, [A] Section 6
    task_deadline: float = 1.0  # τ_app (giới hạn trên; deadline hiệu dụng — T1.1)

    # Deadline động (T1.1)
    deadline_floor: float = 0.05  # s — MỞ RỘNG: sàn của τ_eff để tránh chia 0

    # CPU & Resources — Ref: [A] Section 6 (f_j = 1 GHz, f_n = 0.5 GHz)
    # Hiệu chỉnh (scripts/calibrate_energy.py; DANH_GIA_KIEN_TRUC_THEO_BAI_BAO.md §3.3): edge phải
    # mạnh hơn xe ([A]: 2×, [B]: F^e/F^loc với q^e ≪ q^c) thì đánh đổi offload mới tồn tại.
    mec_cpu_frequency: float = 3e9   # F^e = 3 GHz [B] Table I
    aux_cpu_frequency: float = 2e9   # hiệu chỉnh
    user_cpu_frequency: float = 1e9  # F^loc tối đa của xe (DVFS) — hiệu chỉnh; [A] 0.5, [B] 2 GHz
    kappa: float = 5e-28  # E = κ·c·s·f² — hiệu chỉnh; [B] dùng 1e-27

    # Channel — Ref: [A] Section 6; δ₀/d₀ tái hiệu chỉnh (T0.5)
    bandwidth_mec: float = 1.0e6   # B = 1 MHz [A]
    bandwidth_v2v: float = 0.8e6   # B_V2V = 0.8 MHz [A]
    noise_power: float = 3.0e-13   # W0 [A]
    channel_ref_gain: float = 4.75e-2  # δ₀ tại d₀ — MỞ RỘNG (hiệu chỉnh, xem T0.5)
    channel_ref_distance: float = 1.0  # d₀ = 1 m — MỞ RỘNG (T0.5)
    path_loss_exp: float = 4.0     # θ = 4 [A]

    # Power — Ref: [A] Section 6
    user_tx_power: float = 0.2   # p_m = 0.2 W [A]
    user_idle_power: float = 0.05  # p_w = 0.05 W [A] — dùng cho E^w (T0.6)

    # Coverage
    mec_coverage_radius: float = 300.0  # L = 300 m [A]
    curb_distance: float = 20.0  # z trong [A] Eq(6) — MỞ RỘNG ([A] không cho giá trị)
    v2v_range: float = 200.0  # MỞ RỘNG

    # Bán kính vùng phát của NGƯỜI GỬI d_i trong [B] Eq(8) — hiệu chỉnh T1.7
    # sao cho n^e nằm trong vùng không suy biến [0.4, 1.2]:
    #   V2I: n^e(200) = 1.34 → p_att = 0.61;  V2V: n^e(150) = 0.75 → p_att = 0.41
    security_range_v2i: float = 200.0  # m — MỞ RỘNG (hiệu chỉnh T1.7)
    security_range_v2v: float = 150.0  # m — MỞ RỘNG (hiệu chỉnh T1.7)

    # Bandwidth pools — MỞ RỘNG
    mec_bandwidth_pool: float = 1e7  # bit/s
    aux_bandwidth_pool: float = 5e6  # bit/s
    resource_floor_fraction: float = 0.01  # MỞ RỘNG (T1.5): dưới mức này coi như hết
                                           # tài nguyên → từ chối (Σβ ≤ 1, [A] 23j)

    # Pricing — Ref: [A] Section 3.5, Section 6
    mec_base_price: float = 1.0    # g_j0 — MỞ RỘNG (giá trị chuẩn hóa)
    aux_base_price: float = 0.5    # g_n0 — MỞ RỘNG
    mec_credibility: float = 0.9   # b_j — MỞ RỘNG
    aux_credibility: float = 0.8   # b_n — MỞ RỘNG
    cost_normalization_divisor: float = 1e9  # MỞ RỘNG: chuẩn hóa workload → giga-cycles (T0.7)
    resource_multiplier_step: float = 0.1    # MỞ RỘNG: λ_res = 1 + 0.1·level (T0.7)
    price_energy_local: float = 1.0  # λ^l = 1 cents/J [A] §6 — năng lượng tính local
    price_energy_mec: float = 1.0    # λ^j — [A] không cho giá trị → giả định = λ^l
    price_energy_av: float = 0.8     # λ^n = 0.8 cents/J [A] §6 — năng lượng USER ở nhánh AUX ([A] Eq(20),(22))

    # Objective & ràng buộc
    # Chi phí rủi ro rò rỉ: C_sec = leakage_price · (bit offload) · P_leak, P_leak = 1 − S ([B] Eq(7)).
    # Local và MEC-Paillier có P_leak = 0 → không thiên lệch về offload (thay thưởng ϖ₁·U cũ).
    leakage_price: float = 1e-7  # cents/bit-at-risk — hiệu chỉnh (scripts/calibrate_energy.py)
    energy_budget_max: float = 0.4  # E_max (J) — [B] Table I (C3)
    # DVFS ([B] C4: 0 ≤ f^c, f^en ≤ F^loc): chọn tần số NHỎ NHẤT còn kịp deadline
    # (năng lượng κ·C·f² tăng theo f). False → luôn chạy F^loc = user_cpu_frequency.
    dvfs_enabled: bool = True
    dvfs_min_fraction: float = 0.1  # f_min = 0.1·F^loc (CPU thật có tần số sàn)
    # Chế độ xử lý ràng buộc:
    #   "fixed"    — env tự trừ penalty cố định (baseline / T1.2 bước 4, λ_p ∈ [2,10])
    #   "external" — env trả reward gốc + cờ vi phạm trong info; agent Lagrangian
    #                (T1.3) tự cộng −λ_T·viol_T − λ_E·viol_E với λ học được
    penalty_mode: str = "fixed"
    deadline_penalty: float = 5.0        # λ_p — MỞ RỘNG (tạm, thay bằng T1.3)
    energy_violation_penalty: float = 5.0  # cents/J vượt E_max — MỞ RỘNG (tạm, T1.3)

    # Security — PHÂN VAI THEO ĐÍCH (DANH_GIA_KIEN_TRUC_THEO_BAI_BAO.md §3.1):
    #   MEC (server không tin cậy): Paillier HE [A] — chống cả MEC tò mò lẫn nghe lén → P_leak = 0
    #   AUX (kênh V2V):            mã khối độ dài N_i [B] — P_leak = 1 − (N_i/N_f)^{n^e}
    encryption_enabled: bool = True
    encryption_mode: str = "simulation"
    he_cycles_per_bit: float = 800.0          # c^en Paillier [A] §6
    block_cipher_cycles_per_bit: float = 90.0  # q^en [B] Table I
    block_cipher_round_factor: float = 0.1    # MỞ RỘNG: chi phí mã khối ∝ 1 + κ_N·log2(N_i/8) (số vòng tăng theo độ dài khóa)
    security_level: float = 1.0

    # Attacker model & Crypto Scheme (Phase 3 - T3.2, T3.4)
    attacker_model: str = "realized"   # "realized" (đếm attacker thật, nᵉ thay đổi theo state) | "expected" [B] Eq(8)
    crypto_scheme: str = "paillier"   # "paillier" (MEC: HE, AUX: mã khối) | "aes" (cả hai mã khối, MEC giải mã) | "none"
    ciphertext_expansion: float = 2.0  # Paillier: ciphertext mod h² → 2× khi upload MEC (T3.4)
    use_4d_action: bool = True         # MultiDiscrete[α, target, β, N_i] (T3.1)
    block_length_options: Tuple[float, ...] = (8.0, 16.0, 32.0, 64.0)

    # Mobility (Phase 4 - T4.2)
    mobility_type: str = "linear"  # "linear", "random_waypoint", or "trace"
    trace_file: str = "data/processed/riobus_corridor.csv"
    # File nhiều cửa sổ (cột window_id, vd. data/processed/riobus/riobus_train.parquet):
    # None → mỗi reset() chọn ngẫu nhiên một cửa sổ; đặt id để cố định.
    trace_window_id: Optional[str] = None

    # Attacker model — Ref: [B] Table I, Eq(5)-(9)
    attacker_density: float = 1e-4  # ρ (vehicles/vehicle) [B] Table I (T0.2)
    frame_length_bits: float = 64.0  # N_f [B] Table I (T0.3)
    encryption_block_length_bits: float = 32.0  # N_i — biến quyết định (T3.1)
    lane_count: float = 8.0        # N_l = 8 [B] Table I
    road_max_speed_mps: float = 24.0  # v_max [B] Table I

    # Vận tốc synthetic — [B] Eq(1), Table I: Gauss cắt cụt μ=13, σ=5 trên [2, 24] m/s
    velocity_mean: float = 13.0
    velocity_std: float = 5.0
    velocity_min: float = 2.0
    velocity_max: float = 24.0

    # Background users (T1.5 Mức 2) — users[1:] chạy Algorithm 1 của [A]
    background_users_enabled: bool = True
    background_alpha_l: float = 0.5      # α_l cố định cho background (T2.2 bước 3)
    background_resource_level: int = 4   # mức tài nguyên giữa
    # (T1.5 Mức 1 — tùy chọn) background load Markov-modulated khi không bật Mức 2
    background_load_enabled: bool = False
    background_load_arrival: float = 0.3   # kỳ vọng Poisson đến mỗi slot
    background_load_departure: float = 0.3  # xác suất rời đi mỗi slot

    # Action discretization
    alpha_levels: int = 10
    action_levels: int = 10

    # Semi-MDP (T1.4)
    dt: float = 0.1  # bước thời gian cơ sở (s)
    episode_length: int = 1000  # số QUYẾT ĐỊNH tối đa mỗi episode (không đếm chờ)
    max_wait_iterations: int = 100000  # chốt an toàn chống treo khi λ quá nhỏ


class VehicularEdgeEnvironment(gym.Env):
    """
    Môi trường semi-MDP cho Vehicular Edge Computing (Gymnasium API).

    State (dim = 4 + 2(J+N)): task (size, τ_eff, ω), khoảng cách, utilization, rate.
    Action (3 chiều): [alpha_l_level, target_selection, resource_level]
        - target_selection: 0 = MEC, 1 = AUX (ràng buộc [A] 23b-23d: α_j·α_n = 0)
    Action: [α_l level, đích (0 MEC / 1 AUX), β level, (N_i index nếu use_4d_action)].
    Reward: r = −C [cents], C = [A] Eq(22) + chi phí rủi ro rò rỉ (xem _execute_task).
        Ràng buộc T ≤ τ_u ([A] 23e) và E ≤ E_max ([B] C3) xử lý qua penalty_mode.
    step() trả (obs, reward, terminated, truncated, info);
        info["elapsed_time"] = thời gian thực đã trôi (semi-MDP, T1.4);
        info["action_mask"] = mask cho state MỚI (T1.6).
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, config: EnvironmentConfig):
        super().__init__()
        self.config = config
        self.logger = Logger("VehicularEdgeEnv")

        # Track active resource allocations: (release_time, node_type, node_idx, cpu, bw)
        self.active_allocations: List[Tuple[float, str, int, float, float]] = []

        # ==================== Initialize Entities ====================
        self.users: List[User] = []
        self._init_users()
        self.mec_servers: List[MECServer] = []
        self._init_mec_servers()
        self.auxiliary_vehicles: List[AuxVehicle] = []
        self._init_auxiliary_vehicles()
        self.attack_vehicles: List[AttackVehicle] = []
        self._init_attack_vehicles()

        # ==================== Initialize Models ====================
        self.channel_model = ChannelModel(
            bandwidth_mec=config.bandwidth_mec,
            bandwidth_v2v=config.bandwidth_v2v,
            noise_power=config.noise_power,
            channel_gain=config.channel_ref_gain,
            ref_distance=config.channel_ref_distance,
            path_loss_exp=config.path_loss_exp,
        )
        self.security_module = SecurityModule(key_length=1024) if config.encryption_enabled else None
        self.security_overhead_model = SecurityOverheadModel(
            SecurityOverheadConfig(
                overhead_cycles_per_bit=config.block_cipher_cycles_per_bit,
                user_cpu_frequency=config.user_cpu_frequency,
                security_level=config.security_level,
                frame_length_bits=config.frame_length_bits,
                default_block_length_bits=config.encryption_block_length_bits,
                attacker_density=config.attacker_density,
                lane_count=config.lane_count,
                max_speed_mps=config.road_max_speed_mps,
            )
        )

        # ==================== Simulation State ====================
        self.current_time: float = 0.0
        self.step_count: int = 0        # số QUYẾT ĐỊNH (T1.4), không đếm slot chờ
        self.episode_count: int = 0

        self.task_queue: Dict[int, List[Task]] = defaultdict(list)
        self.task_history: Dict[int, Dict[str, Any]] = {}
        self.background_history: List[Dict[str, Any]] = []
        self.next_task_id: int = 0

        # Resource tracking
        self.mec_cpu_available: List[float] = [config.mec_cpu_frequency] * len(self.mec_servers)
        self.mec_bw_available: List[float] = [config.mec_bandwidth_pool] * len(self.mec_servers)
        self.aux_cpu_available: List[float] = [config.aux_cpu_frequency] * len(self.auxiliary_vehicles)
        self.aux_bw_available: List[float] = [config.aux_bandwidth_pool] * len(self.auxiliary_vehicles)

        # Metrics (T1.5)
        self.rejected_resource_count: int = 0
        self.background_task_count: int = 0

        # ==================== Gym Spaces ====================
        state_dim = self._get_state_dim()
        self.observation_space = spaces.Box(low=0, high=1, shape=(state_dim,), dtype=np.float32)
        nvec = [config.alpha_levels, 2, config.action_levels]   # α_l, đích (0 MEC / 1 AUX), β
        if config.use_4d_action:
            nvec.append(len(config.block_length_options))          # N_i (T3.1)
        self.action_space = spaces.MultiDiscrete(nvec)

        self.logger.info(f"Environment initialized with {len(self.users)} users, "
                         f"{len(self.mec_servers)} MEC, {len(self.auxiliary_vehicles)} aux vehicles")

    # ==================== Initialization Methods ====================

    def _load_trace_file(self):
        """Đọc file trace một lần (cache)."""
        if hasattr(self, "_trace_full_df"):
            return self._trace_full_df
        from pathlib import Path
        import pandas as pd
        t_path = Path(self.config.trace_file)
        df = None
        if t_path.exists():
            try:
                # chỉ đọc cột mobility cần; id dạng category → ít RAM (chạy song song nhiều run)
                cols = ["window_id", "vehicle_id", "t_rel", "x_m", "y_m", "v_mps", "active"]
                if t_path.suffix == ".parquet":
                    import pyarrow.parquet as pq
                    df = pd.read_parquet(t_path, columns=[c for c in cols if c in pq.read_schema(t_path).names])
                else:
                    df = pd.read_csv(t_path)
                    df = df[[c for c in cols if c in df.columns]]
                for c in ("window_id", "vehicle_id"):
                    if c in df.columns:
                        df[c] = df[c].astype("category")
            except Exception as e:
                self.logger.warning(f"Không đọc được trace {t_path}: {e}")
        else:
            self.logger.warning(f"Trace file {t_path} không tồn tại — dùng LinearMobility")
        self._trace_full_df = df
        return df

    def _select_trace_window(self):
        """Chọn cửa sổ thời gian (mọi xe trong cửa sổ chung một đồng hồ thật)."""
        import pandas as pd
        df = self._load_trace_file()
        if df is None or df.empty or "window_id" not in df.columns:
            self._cached_trace_df = df
            return
        if self.config.trace_window_id is not None:
            wid = self.config.trace_window_id
        else:
            wids = sorted(map(str, df["window_id"].unique()))
            wid = wids[np.random.randint(len(wids))]
        self.trace_window_id = wid
        self._cached_trace_df = df[df["window_id"] == wid].copy()
        for c in ("window_id", "vehicle_id"):
            if isinstance(self._cached_trace_df[c].dtype, pd.CategoricalDtype):
                self._cached_trace_df[c] = self._cached_trace_df[c].astype(str)

    def _get_trace_df(self):
        """Helper to load trace dataset for TraceMobility (cửa sổ hiện tại)."""
        if not hasattr(self, "_cached_trace_df"):
            self._select_trace_window()
        return self._cached_trace_df

    def _reset_trace_mobility(self):
        """reset() ở chế độ trace: chọn lại cửa sổ, gán lại quỹ đạo cho mọi thực thể."""
        from src.env.mobility import TraceMobility
        self._select_trace_window()
        trace_df = self._cached_trace_df
        if trace_df is None or trace_df.empty:
            return False
        # Slot phủ dữ liệu tốt nhất cho users (users[0] là agent), rồi AUX, rồi attacker
        if "active" in trace_df.columns:
            cov = trace_df.groupby("vehicle_id")["active"].mean()
            v_ids = cov.sort_values(ascending=False, kind="stable").index.tolist()
        else:
            v_ids = trace_df["vehicle_id"].unique().tolist()
        entities = [*self.users, *self.auxiliary_vehicles]   # attacker luôn synthetic
        for i, entity in enumerate(entities):
            mob = TraceMobility(trace_df, vehicle_id=str(v_ids[i % len(v_ids)]))
            entity.mobility_model = mob
            entity.position, v0 = mob.generate_initial_state()
            if hasattr(entity, "velocity"):
                entity.velocity = v0
        return True

    def _sample_velocity(self) -> float:
        """[B] Eq(1): Gauss cắt cụt N(μ, σ²) trên [v_min, v_max] (lấy mẫu loại bỏ)."""
        cfg = self.config
        for _ in range(1000):
            v = np.random.normal(cfg.velocity_mean, cfg.velocity_std)
            if cfg.velocity_min <= v <= cfg.velocity_max:
                return float(v)
        return float(np.clip(cfg.velocity_mean, cfg.velocity_min, cfg.velocity_max))

    def _init_users(self):
        trace_df = self._get_trace_df() if self.config.mobility_type == "trace" else None
        v_ids = trace_df["vehicle_id"].unique().tolist() if trace_df is not None and not trace_df.empty else []

        for i in range(self.config.num_users):
            if trace_df is not None and v_ids:
                veh_id = str(v_ids[i % len(v_ids)])
                mob = TraceMobility(trace_df, vehicle_id=veh_id)
                pos, velocity = mob.generate_initial_state()
            else:
                pos = np.array([
                    np.random.uniform(0, self.config.sim_area_x),
                    np.random.uniform(0, self.config.sim_area_y)
                ])
                velocity = self._sample_velocity()  # [B] Eq(1)
                mob = LinearMobility()

            user = User(
                id=i,
                position=pos,
                cpu_frequency=self.config.user_cpu_frequency,
                velocity=velocity,
                tx_power=self.config.user_tx_power,
                mobility_model=mob
            )
            self.users.append(user)

    def _init_mec_servers(self):
        mec_positions = [
            np.array([self.config.sim_area_x * 0.25, self.config.sim_area_y * 0.25]),
            np.array([self.config.sim_area_x * 0.75, self.config.sim_area_y * 0.75]),
        ]
        for i in range(min(self.config.num_mec_servers, len(mec_positions))):
            mec = MECServer(
                id=i,
                position=mec_positions[i],
                cpu_frequency=self.config.mec_cpu_frequency,
                coverage_radius=self.config.mec_coverage_radius,
                curb_distance=self.config.curb_distance,
                base_price=self.config.mec_base_price,
                credibility=self.config.mec_credibility
            )
            self.mec_servers.append(mec)

    def _init_auxiliary_vehicles(self):
        trace_df = self._get_trace_df() if self.config.mobility_type == "trace" else None
        v_ids = trace_df["vehicle_id"].unique().tolist() if trace_df is not None and not trace_df.empty else []

        for i in range(self.config.num_auxiliary_vehicles):
            if trace_df is not None and v_ids:
                veh_id = str(v_ids[(i + self.config.num_users) % len(v_ids)])
                mob = TraceMobility(trace_df, vehicle_id=veh_id)
                pos, velocity = mob.generate_initial_state()
            else:
                pos = np.array([
                    np.random.uniform(0, self.config.sim_area_x),
                    np.random.uniform(0, self.config.sim_area_y)
                ])
                velocity = self._sample_velocity()  # [B] Eq(1)
                mob = LinearMobility()

            aux = AuxVehicle(
                id=i,
                position=pos,
                cpu_frequency=self.config.aux_cpu_frequency,
                base_price=self.config.aux_base_price,
                credibility=self.config.aux_credibility,
                current_load=0,
                velocity=velocity,
                mobility_model=mob
            )
            self.auxiliary_vehicles.append(aux)

    def _init_attack_vehicles(self):
        """Attacker vật lý (attacker_model="realized"). Luôn synthetic, kể cả khi mobility là trace:
        dữ liệu Rio chỉ có xe buýt, nên attacker là mật độ xe con giả định (DANH_GIA_KIEN_TRUC §3.1)."""
        for i in range(self.config.num_attack_vehicles):
            if True:
                pos = np.array([
                    np.random.uniform(0, self.config.sim_area_x),
                    np.random.uniform(0, self.config.sim_area_y)
                ])
                mob = LinearMobility()

            self.attack_vehicles.append(AttackVehicle(
                id=i, position=pos, cpu_frequency=1e8,
                mobility_model=mob
            ))

    def _get_state_dim(self) -> int:
        """
        dim = 5 + 2·(J + N): task (3) + distances (J+N) + utilization (J+N) + rate (1)
        + số attacker nᵉ trong tầm phát V2V (1) — [B] giả định người gửi biết nᵉ để chọn N_i (Eq 39).
        """
        j = self.config.num_mec_servers
        n = self.config.num_auxiliary_vehicles
        return 5 + 2 * (j + n)

    # ==================== T1.1 — Residence time / deadline động ====================

    @staticmethod
    def _time_to_exit_circle(rel_pos: np.ndarray, rel_vel: np.ndarray, radius: float) -> float:
        """
        Thời gian tới khi ||rel_pos + rel_vel·t|| = radius (điểm rời khỏi hình tròn).

        Tổng quát hóa [A] Eq(6)-(8) sang 2D: [A] giả định xe chạy trên đường thẳng
        qua vùng phủ; ta giải trực tiếp thời điểm ra khỏi đường tròn theo hướng
        chuyển động hiện tại (trùng [A] trong trường hợp 1D).
        Trả về 0 nếu đang ở ngoài; +inf nếu đứng yên tương đối bên trong.
        """
        c = float(rel_pos @ rel_pos) - radius ** 2
        if c > 0:
            return 0.0  # đã ở ngoài vùng
        a = float(rel_vel @ rel_vel)
        if a < 1e-12:
            return np.inf
        b = 2.0 * float(rel_pos @ rel_vel)
        disc = b * b - 4 * a * c  # luôn ≥ 0 khi ở trong
        t = (-b + np.sqrt(max(disc, 0.0))) / (2 * a)
        return max(t, 0.0)

    def residence_time_mec(self, user: User, mec: MECServer) -> float:
        """
        [A] Eq(7) / [B] Eq(2): thời gian lưu trú còn lại của user trong vùng phủ.
        Trả về +inf nếu ngoài vùng phủ (deadline do AUX/local quyết định).
        """
        if not mec.is_covered(user):
            return np.inf
        return self._time_to_exit_circle(
            user.position - mec.position, user.velocity_vector(), mec.coverage_radius)

    def v2v_link_time(self, user: User, aux: AuxVehicle) -> float:
        """
        MỞ RỘNG (T1.1 bước 3): [A]/[B] chỉ xét lưu trú với hạ tầng cố định;
        ta mở rộng cho liên kết V2V — thời gian hai xe còn trong tầm, tính từ
        vận tốc TƯƠNG ĐỐI. Trả về +inf nếu ngoài tầm (không ràng buộc deadline).
        """
        rel_pos = user.position - aux.position
        if float(rel_pos @ rel_pos) > self.config.v2v_range ** 2:
            return np.inf
        rel_vel = user.velocity_vector() - aux.velocity_vector()
        return self._time_to_exit_circle(rel_pos, rel_vel, self.config.v2v_range)

    def effective_deadline(self, user: User, task: Task) -> Tuple[float, bool]:
        """
        T1.1: τ_u(t) = min(τ_app, t^w_MEC, t^w_V2V), sàn deadline_floor.

        Returns:
            (tau_eff, link_limited) — link_limited=True nếu deadline bị giới hạn
            bởi thời gian lưu trú/liên kết chứ không phải bởi ứng dụng.
        """
        t_w_mec = min((self.residence_time_mec(user, m) for m in self.mec_servers),
                      default=np.inf)
        aux = self._nearest_aux(user)
        t_w_v2v = self.v2v_link_time(user, aux) if aux is not None else np.inf
        tau_raw = min(task.max_delay, t_w_mec, t_w_v2v)
        tau_eff = float(np.clip(tau_raw, self.config.deadline_floor, task.max_delay))
        return tau_eff, bool(tau_raw < task.max_delay)

    def _nearest_aux(self, user: User) -> Optional[AuxVehicle]:
        if not self.auxiliary_vehicles:
            return None
        idx = int(np.argmin([user.get_distance(a) for a in self.auxiliary_vehicles]))
        return self.auxiliary_vehicles[idx]

    # ==================== T1.6 — Action masking ====================

    @staticmethod
    def _is_present(entity) -> bool:
        """Trace-slot không có xe buýt thật (active=0) coi như không tồn tại (T4.2)."""
        return getattr(entity.mobility_model, "is_active", True)

    def _mec_feasible(self, user: User) -> bool:
        if not self._is_present(user):
            return False
        floor = self.config.resource_floor_fraction * self.config.mec_cpu_frequency
        return bool(self.mec_servers) and self.mec_servers[0].is_covered(user) \
            and self.mec_cpu_available[0] > floor

    def _aux_feasible(self, user: User) -> Tuple[bool, Optional[int]]:
        """AUX khả thi nếu trong tầm V2V, RẢNH ([A] §3.4: ≤1 user/AV) và còn tài nguyên."""
        if not self.auxiliary_vehicles or not self._is_present(user):
            return False, None
        floor = self.config.resource_floor_fraction * self.config.aux_cpu_frequency
        candidates = [
            (user.get_distance(a), i) for i, a in enumerate(self.auxiliary_vehicles)
            if user.get_distance(a) <= self.config.v2v_range
            and self._is_present(a)
            and a.current_load == 0
            and self.aux_cpu_available[i] > floor
        ]
        if not candidates:
            return False, None
        return True, min(candidates)[1]

    def action_mask(self, user: Optional[User] = None) -> np.ndarray:
        """
        T1.6: mask phẳng theo C-order của action_space.nvec — khớp np.ravel_multi_index. True = khả thi.

        - α_l = 1 (level cuối) luôn khả thi.
        - Đích MEC cần vùng phủ + tài nguyên; AUX cần AUX trong tầm, rảnh, còn tài nguyên.
        - Slot trace inactive (T4.2) → không offload.
        - N_i (4D): chỉ có nghĩa với nhánh mã khối → nhánh khác cố định ở N_i mặc định
          (loại action trùng lặp).
        """
        user = user or self.users[0]
        nvec = tuple(int(n) for n in self.action_space.nvec)
        L = nvec[0]
        mask = np.zeros(nvec, dtype=bool)
        ok = {0: self._mec_feasible(user), 1: self._aux_feasible(user)[0]}
        if len(nvec) == 3:
            mask[L - 1] = True
            for t in (0, 1):
                if ok[t]:
                    mask[:, t, :] = True
            return mask.flatten()
        default_k = self._default_block_index()
        mask[L - 1, :, :, default_k] = True
        for t in (0, 1):
            if not ok[t]:
                continue
            if self._cipher_for("mec" if t == 0 else "aux") == "block":
                mask[:-1, t, :, :] = True
            else:
                mask[:-1, t, :, default_k] = True
        return mask.flatten()

    # ---------- dry-run (baseline Greedy/CMSO, hiệu chỉnh) ----------
    def snapshot_resources(self):
        return (list(self.mec_cpu_available), list(self.mec_bw_available), list(self.aux_cpu_available),
                list(self.aux_bw_available), list(self.active_allocations),
                [a.current_load for a in self.auxiliary_vehicles], self.rejected_resource_count)

    def restore_resources(self, snap):
        (mc, mb, ac, ab, alloc, loads, rej) = snap
        self.mec_cpu_available, self.mec_bw_available = list(mc), list(mb)
        self.aux_cpu_available, self.aux_bw_available = list(ac), list(ab)
        self.active_allocations, self.rejected_resource_count = list(alloc), rej
        for a, l in zip(self.auxiliary_vehicles, loads):
            a.current_load = l

    def evaluate_action(self, action: np.ndarray, user: Optional[User] = None) -> Dict[str, Any]:
        """Chạy thử action cho task đầu hàng đợi của user, KHÔNG thay đổi trạng thái env."""
        user = user or self.users[0]
        task = self.task_queue[user.id][0]
        tau_eff, _ = self.effective_deadline(user, task)
        al, aj, an, lvl, blk = self._decode_action_to_alphas(action)
        snap = self.snapshot_resources()
        try:
            return self._execute_task(user, task, al, aj, an, lvl, tau_eff, blk)
        finally:
            self.restore_resources(snap)

    def make_action(self, alpha_level: int, target: int, resource_level: int, block_index: Optional[int] = None) -> np.ndarray:
        """Tạo action đúng số chiều của action_space (3 hoặc 4)."""
        a = [alpha_level, target, resource_level]
        if len(self.action_space.nvec) == 4:
            a.append(self._default_block_index() if block_index is None else block_index)
        return np.array(a, dtype=np.int64)

    def _default_block_index(self) -> int:
        opts = list(self.config.block_length_options)
        target = self.config.encryption_block_length_bits
        return int(np.argmin([abs(o - target) for o in opts]))

    # ==================== Gymnasium API (T1.4) ====================

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        trace_reset = self.config.mobility_type == "trace" and self._reset_trace_mobility()
        for entity in (self.attack_vehicles if trace_reset else [*self.users, *self.auxiliary_vehicles, *self.attack_vehicles]):
            entity.position = np.array([
                np.random.uniform(0, self.config.sim_area_x),
                np.random.uniform(0, self.config.sim_area_y)
            ])
            if hasattr(entity, "velocity"):
                entity.velocity = self._sample_velocity()   # mỗi episode một mẫu vận tốc [B] Eq(1)

        self.current_time = 0.0
        self.step_count = 0
        self.task_queue.clear()
        self.task_history.clear()
        self.background_history.clear()
        self.next_task_id = 0
        self.active_allocations.clear()
        self.rejected_resource_count = 0
        self.background_task_count = 0

        self.mec_cpu_available = [self.config.mec_cpu_frequency] * len(self.mec_servers)
        self.mec_bw_available = [self.config.mec_bandwidth_pool] * len(self.mec_servers)
        self.aux_cpu_available = [self.config.aux_cpu_frequency] * len(self.auxiliary_vehicles)
        self.aux_bw_available = [self.config.aux_bandwidth_pool] * len(self.auxiliary_vehicles)
        for aux in self.auxiliary_vehicles:
            aux.current_load = 0

        # Semi-MDP: tiến thời gian tới khi user 0 có task — state đầu tiên luôn
        # là một state quyết định thật (không có transition rỗng, T1.4).
        self._wait_for_task()

        self.episode_count += 1
        info = {"action_mask": self.action_mask(), "elapsed_time": 0.0}
        return self._get_state(), info

    def _wait_for_task(self) -> int:
        """Tiến thời gian nội bộ tới khi user 0 có task. Trả về số slot đã chờ."""
        user = self.users[0]
        waited = 0
        while not self.task_queue[user.id]:
            self._advance_time()
            waited += 1
            if waited > self.config.max_wait_iterations:
                raise RuntimeError(
                    "Không có task nào đến sau "
                    f"{self.config.max_wait_iterations} slot — kiểm tra task_arrival_rate")
        return waited

    def _advance_time(self):
        """Một slot dt: di chuyển, sinh task, background users hành động, giải phóng tài nguyên."""
        self._update_vehicle_positions()
        self._generate_tasks()
        if self.config.background_users_enabled:
            self._run_background_users()
        elif self.config.background_load_enabled:
            self._step_background_load()
        self.current_time += self.config.dt
        self._release_resources()

    def step(self, action: np.ndarray):
        """
        Một QUYẾT ĐỊNH của agent cho user 0 (semi-MDP, T1.4).

        Returns:
            (state, reward, terminated, truncated, info) — Gymnasium API.
            info["elapsed_time"]: thời gian thực đã trôi từ quyết định trước
                (dùng cho discount γ^(Δt/dt)); không có transition rỗng nào
                được trả về (T1.4 DoD).
        """
        alpha_l, alpha_j, alpha_n, resource_level, block_length_bits = self._decode_action_to_alphas(action)
        user = self.users[0]

        # Bất biến semi-MDP: state đưa cho agent LUÔN chứa task ở đầu hàng đợi
        # (reset() và cuối step() đảm bảo điều này) — action không bao giờ được
        # quyết định trên state thiếu task.
        assert self.task_queue[user.id], (
            "Bất biến semi-MDP bị vỡ: không có task ở đầu quyết định (T1.4)")
        task = self.task_queue[user.id].pop(0)

        # T1.1: deadline hiệu dụng từ thời gian lưu trú
        tau_eff, link_limited = self.effective_deadline(user, task)

        metrics = self._execute_task(
            user, task, alpha_l, alpha_j, alpha_n, resource_level, tau_eff, block_length_bits)

        # ==================== Reward (T1.2 — Phương án 1) ====================
        reward, reward_components = self._compute_reward(metrics)

        # Ghi lịch sử
        task_id = self.next_task_id
        self.next_task_id += 1
        self.task_history[task_id] = {
            "user_id": user.id,
            "tau_app": task.max_delay,
            "tau_eff": tau_eff,
            "link_limited": link_limited,
            "alpha_l": alpha_l, "alpha_j": alpha_j, "alpha_n": alpha_n,
            "timestamp": self.current_time,
            **metrics,
        }

        # Slot của quyết định hiện tại
        self._advance_time()
        self.step_count += 1  # đếm QUYẾT ĐỊNH (T1.4)

        terminated = False  # không có trạng thái kết thúc tự nhiên
        truncated = self.step_count >= self.config.episode_length

        # Semi-MDP (T1.4): tiến thời gian tới TASK KẾ TIẾP ngay bây giờ, để
        # next_state luôn chứa task và thời gian chờ được tính vào elapsed_time
        # của TRANSITION NÀY (γ_eff = γ^(elapsed/dt) discount đúng khoảng chờ).
        waited = 0 if truncated else self._wait_for_task()
        elapsed_time = self.config.dt * (waited + 1)

        state = self._get_state()
        info = {
            "alpha": [alpha_l, alpha_j, alpha_n],
            "elapsed_time": elapsed_time,
            "action_mask": self.action_mask(),
            "tau_eff": tau_eff,
            "link_limited": link_limited,
            "task_delay": metrics["delay"],
            "task_energy": metrics["energy_user"],
            "task_energy_enc": metrics["energy_enc"],
            "task_energy_wait": metrics["energy_wait"],
            "task_energy_compute_tx": metrics["energy_compute_tx"],
            "task_cost": metrics["cost_total"],
            "task_cost_service": metrics["cost_service"],
            "task_cost_energy": metrics["cost_energy"],
            "task_security_risk": metrics["security_risk"],
            "task_cost_security": metrics["cost_security"],
            "task_block_length": metrics["block_length"],
            "task_secure_capacity": metrics["secure_capacity"],
            "deadline_violated": metrics["viol_T"],
            "energy_violation": metrics["viol_E"],
            "failure_cause": metrics["failure_cause"],
            "offload_location": metrics["offload_location"],
            "reward_components": reward_components,
            "num_tasks_completed": len(self.task_history),
        }
        return state, reward, terminated, truncated, info

    # ==================== Action decode ====================

    def _decode_action_to_alphas(self, action: np.ndarray) -> Tuple[float, float, float, int, float]:
        """
        [alpha_l_level, target, resource_level, (optional) n_i_index] → (α_l, α_j, α_n, level, N_i).
        Ràng buộc [A] 23b-23d: α_l + α_j + α_n = 1, α_j·α_n = 0.
        """
        alpha_levels = max(int(self.config.alpha_levels), 1)
        action_arr = np.asarray(action, dtype=np.int64).flatten()
        if action_arr.size < 3:
            raise ValueError(f"Expected action with at least 3 dimensions, got {action_arr.size}: {action}")

        alpha_l_level = int(np.clip(action_arr[0], 0, alpha_levels - 1))
        target_selection = int(np.clip(action_arr[1], 0, 1))
        resource_level = int(np.clip(action_arr[2], 0, int(self.config.action_levels) - 1))

        block_length_bits = self.config.encryption_block_length_bits
        if action_arr.size >= 4:
            n_i_options = tuple(self.config.block_length_options)
            n_i_idx = int(np.clip(action_arr[3], 0, len(n_i_options) - 1))
            block_length_bits = n_i_options[n_i_idx]

        scale = float(max(alpha_levels - 1, 1))
        alpha_l = float(alpha_l_level / scale)
        alpha_remote = max(0.0, 1.0 - alpha_l)

        if alpha_remote <= 1e-12:
            return 1.0, 0.0, 0.0, resource_level, block_length_bits
        if target_selection == 0:
            return alpha_l, alpha_remote, 0.0, resource_level, block_length_bits
        return alpha_l, 0.0, alpha_remote, resource_level, block_length_bits

    # ==================== T1.5 — Thực thi task (chung mọi user) ====================

    def _make_subtask(self, task: Task, ratio: float) -> Optional[Task]:
        ratio = float(max(ratio, 0.0))
        if ratio <= 1e-12:
            return None
        return Task(size=task.size * ratio, cycles_per_bit=task.cycles_per_bit,
                    max_delay=task.max_delay)

    def _execute_task(self, user: User, task: Task, alpha_l: float, alpha_j: float,
                      alpha_n: float, resource_level: int, tau_eff: float,
                      block_length_bits: Optional[float] = None) -> Dict[str, Any]:
        """
        Thực thi một task cho user bất kỳ (T1.5: tách khỏi users[0]).

        - Σβ ≤ 1 ([A] 23j): cấp phát là % pool CÒN LẠI; dưới resource_floor_fraction → từ chối.
        - Mỗi AUX phục vụ ≤ 1 user ([A] §3.4).
        - CPU xe user: mã hóa (phần offload) rồi tính local, TUẦN TỰ trên một CPU, tần số chọn
          bằng DVFS ([B] C4); nhánh remote chạy song song sau mã hóa.
        - An ninh phân vai: MEC ← Paillier HE (P_leak = 0), AUX ← mã khối N_i (P_leak = 1 − S).
        - Chi phí ([A] Eq(22) + rủi ro): C = λ^l·E_local + λ^b·E_branch + g_b + C_sec,
          E_branch = E_enc + E_tx + E_wait của user cho nhánh b ∈ {MEC (λ^j), AUX (λ^n)}.
        """
        cfg = self.config
        n_i = float(block_length_bits if block_length_bits is not None else cfg.encryption_block_length_bits)

        C_local = 0.0            # CPU cycles tính local (gồm cả nhánh fallback)
        remote: Optional[Dict[str, Any]] = None
        offload_locations: List[str] = []

        local_task = self._make_subtask(task, alpha_l)
        if local_task is not None:
            C_local += local_task.cycles_per_bit * local_task.size
            offload_locations.append("local")

        # α_j·α_n = 0 ([A] 23b-c) → tối đa một nhánh remote
        for kind, frac in (("mec", alpha_j), ("aux", alpha_n)):
            sub = self._make_subtask(task, frac)
            if sub is None:
                continue
            branch = self._offload_branch(user, kind, sub, frac, resource_level, tau_eff, n_i)
            if branch is None:   # không khả thi → xử lý local
                C_local += sub.cycles_per_bit * sub.size
                offload_locations.append(f"local_fallback_{kind}")
            else:
                remote = branch
                offload_locations.append(branch["label"])

        # ---- DVFS & tổng hợp thời gian ----
        t_remote = remote["delay"] if remote else 0.0
        C_enc = remote["enc_cycles"] if remote else 0.0
        f_enc, f_loc = self._dvfs_frequencies(C_enc, C_local, t_remote, tau_eff, user.cpu_frequency)
        t_enc = C_enc / f_enc if C_enc > 0 else 0.0
        t_local = C_local / f_loc if C_local > 0 else 0.0
        delay = t_enc + max(t_local, t_remote)

        # ---- Năng lượng user ----
        E_local = cfg.kappa * C_local * f_loc ** 2
        E_enc = cfg.kappa * C_enc * f_enc ** 2
        E_tx = remote["tx_energy"] if remote else 0.0
        E_wait = cfg.user_idle_power * max(0.0, t_remote - t_local) if remote else 0.0   # T0.6
        E_branch = E_enc + E_tx + E_wait
        energy_user = E_local + E_branch

        # ---- Chi phí ([A] Eq(22)) + rủi ro rò rỉ ----
        branch_price = 0.0
        if remote:
            branch_price = cfg.price_energy_mec if remote["kind"] == "mec" else cfg.price_energy_av
        cost_energy = cfg.price_energy_local * E_local + branch_price * E_branch
        cost_service = remote["price"] if remote else 0.0
        bits_at_risk = remote["bits_at_risk"] if remote else 0.0
        cost_security = cfg.leakage_price * bits_at_risk
        cost_total = cost_energy + cost_service + cost_security

        viol_T = bool(delay > tau_eff)
        viol_E = float(max(0.0, energy_user - cfg.energy_budget_max))
        if not viol_T:
            failure_cause = "none"
        elif delay <= task.max_delay:
            failure_cause = "link_loss"   # chỉ vỡ vì mất liên kết (T1.1 bước 7)
        else:
            failure_cause = "deadline"

        return {
            "delay": delay,
            "energy_user": energy_user,
            "energy_local": E_local,
            "energy_enc": E_enc,
            "energy_wait": E_wait,
            "energy_compute_tx": energy_user - E_enc - E_wait,
            "cost_service": cost_service,
            "cost_energy": cost_energy,
            "cost_security": cost_security,
            "cost_total": cost_total,
            "security_risk": remote["frac"] * remote["p_leak"] if remote else 0.0,   # tỉ lệ bit của task bị đe dọa
            "bits_at_risk": bits_at_risk,
            "secure_capacity": remote["frac"] * remote["rate"] * (1.0 - remote["p_leak"]) if remote else 0.0,
            "secure_info_capacity_B": remote["U_B"] if remote else 0.0,   # [B] Eq(9), chỉ nhánh mã khối
            "block_length": n_i if (remote and remote["cipher"] == "block") else 0.0,
            "viol_T": viol_T,
            "viol_E": viol_E,
            "failure_cause": failure_cause,
            "offload_location": "+".join(offload_locations) if offload_locations else "none",
        }

    def _cipher_for(self, kind: str) -> str:
        """'he' | 'block' | 'none' cho nhánh MEC/AUX theo crypto_scheme."""
        cfg = self.config
        if not cfg.encryption_enabled or cfg.crypto_scheme == "none":
            return "none"
        if kind == "mec" and cfg.crypto_scheme == "paillier":
            return "he"
        return "block"

    def _offload_branch(self, user: User, kind: str, sub: Task, frac: float, resource_level: int,
                        tau_eff: float, n_i: float) -> Optional[Dict[str, Any]]:
        """Một nhánh offload (MEC [A] Eq(10)-(12) / AUX [A] Eq(15)-(18)); None nếu không khả thi.
        Có tác dụng phụ: chiếm tài nguyên (Σβ ≤ 1), AUX bận (≤ 1 user)."""
        cfg = self.config
        cipher = self._cipher_for(kind)
        if kind == "mec":
            if not self._mec_feasible(user):
                if self.mec_servers and self.mec_servers[0].is_covered(user):
                    self.rejected_resource_count += 1
                return None
            node, idx = self.mec_servers[0], 0
            cpu_pool, bw_pool = self.mec_cpu_available, self.mec_bw_available
            rate = self.channel_model.get_uplink_rate_mec(user.get_distance(node), user.tx_power)
            sec_range = cfg.security_range_v2i
        else:
            ok, idx = self._aux_feasible(user)
            if not ok:
                near = self._nearest_aux(user)
                if near is not None and user.get_distance(near) <= cfg.v2v_range:
                    self.rejected_resource_count += 1
                return None
            node = self.auxiliary_vehicles[idx]
            cpu_pool, bw_pool = self.aux_cpu_available, self.aux_bw_available
            rate = self.channel_model.get_uplink_rate_v2v(user.get_distance(node), user.tx_power)
            sec_range = cfg.security_range_v2v

        cpu_freq = max(1.0, self._get_resource_allocation(cpu_pool[idx], resource_level))
        bw = max(1.0, self._get_bandwidth_allocation(bw_pool[idx], resource_level))

        # Mã hóa, kích thước truyền, giải mã, rủi ro
        if cipher == "he":
            enc_cycles = cfg.he_cycles_per_bit * sub.size / max(cfg.security_level, 1e-6)
            tx_bits = sub.size * cfg.ciphertext_expansion
            dec_cycles, p_leak, U_B = 0.0, 0.0, 0.0          # HE: server tính trên ciphertext [A] §3.3
        elif cipher == "block":
            enc_cycles = self._encryption_cycles(sub.size, n_i)
            tx_bits = sub.size
            p_att, U_B = self._compute_security_metrics(sec_range, rate, n_i, user)
            if kind == "mec":   # "aes": MEC phải giải mã → server tò mò đọc được dữ liệu
                dec_cycles, p_leak = cfg.block_cipher_cycles_per_bit * sub.size, 1.0
            else:
                dec_cycles, p_leak = 0.0, p_att
        else:
            enc_cycles, tx_bits, dec_cycles, p_leak, U_B = 0.0, sub.size, 0.0, 1.0, 0.0

        tx_time = tx_bits / max(rate, 1.0)
        exe_time = (sub.cycles_per_bit * sub.size + dec_cycles) / cpu_freq
        delay = tx_time + exe_time
        price = (self._compute_mec_cost(node, sub, resource_level, tau_eff) if kind == "mec"
                 else self._compute_aux_cost(node, sub, resource_level, tau_eff))

        cpu_pool[idx] = max(0.0, cpu_pool[idx] - cpu_freq)
        bw_pool[idx] = max(0.0, bw_pool[idx] - bw)
        if kind == "aux":
            node.current_load += 1   # [A] §3.4
        self.active_allocations.append((self.current_time + delay, kind, idx, cpu_freq, bw))

        return {"kind": kind, "label": "MEC_0" if kind == "mec" else f"Aux_{idx}", "cipher": cipher,
                "frac": frac, "rate": rate, "delay": delay, "tx_energy": user.tx_power * tx_time,
                "enc_cycles": enc_cycles, "price": price, "p_leak": p_leak,
                "bits_at_risk": sub.size * p_leak, "U_B": U_B}

    # ==================== T1.5 Mức 2 — Background users (Algorithm 1 [A]) ====================

    def _algorithm1_policy(self, user: User) -> Tuple[float, float, float, int]:
        """
        [A] Algorithm 1 (T2.2, dùng cho background): nếu trong vùng phủ MEC →
        offload MEC; ngược lại chọn AUX (trong tầm, RẢNH) có g_n nhỏ nhất;
        nếu không có gì khả thi → local. α_l cố định (T2.2 bước 3).
        """
        cfg = self.config
        a_l = float(np.clip(cfg.background_alpha_l, 0.0, 1.0))
        level = int(cfg.background_resource_level)
        if self._mec_feasible(user):
            return a_l, 1.0 - a_l, 0.0, level
        aux_ok, aux_idx = self._aux_feasible(user)
        if aux_ok:
            # chọn theo giá g_n nhỏ nhất trong các AUX khả thi (đúng Algorithm 1)
            floor = cfg.resource_floor_fraction * cfg.aux_cpu_frequency
            feasible = [
                (self.auxiliary_vehicles[i].get_dynamic_price(1.0), i)
                for i, a in enumerate(self.auxiliary_vehicles)
                if user.get_distance(a) <= cfg.v2v_range and a.current_load == 0
                and self.aux_cpu_available[i] > floor
            ]
            _ = min(feasible)  # aux được chọn lại trong _execute_task theo khoảng cách;
            # với ≤1 user/AV và giá đồng nhất g_n0, min-giá ≈ min-tải — chấp nhận được
            return a_l, 0.0, 1.0 - a_l, level
        return 1.0, 0.0, 0.0, level

    def _run_background_users(self):
        """users[1:] hành động mỗi slot — chiếm tài nguyên THẬT (T1.5 Mức 2)."""
        for user in self.users[1:]:
            if not self.task_queue[user.id] or not self._is_present(user):
                continue
            task = self.task_queue[user.id].pop(0)
            # chặn backlog: bỏ task tồn quá 5 (tránh queue nổ khi kẹt tài nguyên)
            if len(self.task_queue[user.id]) > 5:
                self.task_queue[user.id] = self.task_queue[user.id][-5:]
            tau_eff, _ = self.effective_deadline(user, task)
            a_l, a_j, a_n, level = self._algorithm1_policy(user)
            metrics = self._execute_task(user, task, a_l, a_j, a_n, level, tau_eff)
            self.background_task_count += 1
            self.background_history.append({
                "user_id": user.id, "timestamp": self.current_time,
                "offload_location": metrics["offload_location"],
                "viol_T": metrics["viol_T"],
            })

    def _step_background_load(self):
        """
        T1.5 Mức 1 (tùy chọn, khi background_users_enabled=False):
        tải nền Markov-modulated: ε(t+1) = clip(ε(t) + Poisson − Binomial, 0, 1).
        Với ràng buộc ≤1 user/AV, tải của AUX là nhị phân bận/rảnh.
        """
        cfg = self.config
        for i, aux in enumerate(self.auxiliary_vehicles):
            if aux.current_load == 0:
                if np.random.random() < 1 - np.exp(-cfg.background_load_arrival):
                    aux.current_load = 1
                    take = 0.5 * self.aux_cpu_available[i]
                    self.aux_cpu_available[i] -= take
                    self.active_allocations.append(
                        (self.current_time + np.random.exponential(1.0), "aux", i, take, 0.0))
        # MEC: chiếm/giải phóng một phần pool ngẫu nhiên
        if self.mec_servers and np.random.random() < 1 - np.exp(-cfg.background_load_arrival):
            take = 0.2 * self.mec_cpu_available[0]
            self.mec_cpu_available[0] -= take
            self.active_allocations.append(
                (self.current_time + np.random.exponential(1.0), "mec", 0, take, 0.0))

    # ==================== Resource management ====================

    def _release_resources(self):
        still_active = []
        for alloc in self.active_allocations:
            release_time, node_type, node_idx, cpu_freq, bw = alloc
            if self.current_time >= release_time:
                if node_type == "mec":
                    self.mec_cpu_available[node_idx] = min(
                        self.config.mec_cpu_frequency,
                        self.mec_cpu_available[node_idx] + cpu_freq)
                    self.mec_bw_available[node_idx] = min(
                        self.config.mec_bandwidth_pool,
                        self.mec_bw_available[node_idx] + bw)
                elif node_type == "aux":
                    self.aux_cpu_available[node_idx] = min(
                        self.config.aux_cpu_frequency,
                        self.aux_cpu_available[node_idx] + cpu_freq)
                    self.aux_bw_available[node_idx] = min(
                        self.config.aux_bandwidth_pool,
                        self.aux_bw_available[node_idx] + bw)
                    self.auxiliary_vehicles[node_idx].current_load = max(
                        0, self.auxiliary_vehicles[node_idx].current_load - 1)
            else:
                still_active.append(alloc)
        self.active_allocations = still_active

    def _get_resource_allocation(self, available_cpu: float, level: int) -> float:
        """level 0 → 10%, level 9 → 100% tài nguyên CÒN LẠI (⇒ Σβ ≤ 1, [A] 23j)."""
        allocation_percent = 0.1 + 0.1 * min(level, 9)
        return available_cpu * allocation_percent

    def _get_bandwidth_allocation(self, available_bw: float, level: int) -> float:
        allocation_percent = 0.1 + 0.1 * min(level, 9)
        return available_bw * allocation_percent

    # ==================== Delay Computation ====================

    def _compute_local_delay(self, user: User, task: Task) -> float:
        """t_local = c·s / f_u. Ref: [A] Eq(11)."""
        return (task.cycles_per_bit * task.size) / user.cpu_frequency

    def attackers_in_range(self, user: Optional[User], d_sender_range: float) -> float:
        """nᵉ: đếm attacker thật trong bán kính phát ("realized", T3.2) hoặc kỳ vọng [B] Eq(8)."""
        if self.config.attacker_model == "realized" and user is not None:
            return float(sum(1 for att in self.attack_vehicles
                             if self._is_present(att) and user.get_distance(att) <= d_sender_range))
        return self.security_overhead_model.expected_attackers(
            coverage_radius_m=d_sender_range, attacker_density=self.config.attacker_density,
            lane_count=self.config.lane_count, max_speed_mps=self.config.road_max_speed_mps)

    def _compute_security_metrics(self, d_sender_range: float,
                                  link_rate_bps: float,
                                  block_length_bits: Optional[float] = None,
                                  user: Optional[User] = None) -> Tuple[float, float]:
        """
        [B] Eq(6)-(9). Ghi chú ngữ nghĩa (T0.2/T1.7/T3.1/T3.2):
        - d_i là BÁN KÍNH VÙNG PHÁT của người gửi;
        - attacker_model == "realized": đếm số attacker vật lý trong tầm phủ (T3.2);
        - block_length_bits: độ dài khối N_i từ action (T3.1).
        """
        n_f = max(self.config.frame_length_bits, 1.0)
        n_i_val = block_length_bits if block_length_bits is not None else self.config.encryption_block_length_bits
        n_i = min(max(float(n_i_val), 1.0), n_f)

        n_e = self.attackers_in_range(user, d_sender_range)

        s = self.security_overhead_model.security_strength(n_i, n_f, n_e)
        p_att = self.security_overhead_model.attack_success_probability(n_i, n_f, n_e)
        u = self.security_overhead_model.secure_information_capacity(link_rate_bps, n_i, s)
        return p_att, u

    def _encryption_cycles(self, data_size_bits: float, block_length_bits: Optional[float] = None) -> float:
        """Mã khối [B]: q^en·(1 + κ_N·log2(N_i/8))·s. Chi phí tăng theo N_i là MỞ RỘNG (số vòng
        tăng theo độ dài khóa, vd AES-128/192/256 = 10/12/14 vòng); [B] dùng q^en hằng số."""
        cfg = self.config
        n_i = max(float(block_length_bits if block_length_bits is not None else cfg.encryption_block_length_bits), 8.0)
        factor = 1.0 + cfg.block_cipher_round_factor * np.log2(n_i / 8.0)
        return cfg.block_cipher_cycles_per_bit * factor * data_size_bits / max(cfg.security_level, 1e-6)

    def _dvfs_frequencies(self, C_enc: float, C_local: float, t_remote: float,
                          tau: float, F_loc: float) -> Tuple[float, float]:
        """
        DVFS cho CPU xe user — [B] C4: 0 ≤ f^en, f^c ≤ F^loc; E = κ·C·f² ([B] Eq(11)).

        min κ(C_en·f_en² + C_l·f_l²)  s.t.  C_en/f_en + C_l/f_l ≤ τ   (local sau mã hóa)
                                            C_en/f_en + T_remote ≤ τ   (upload sau mã hóa)
        KKT: không ràng buộc thứ hai → f_en = f_l = (C_en + C_l)/τ (tần số bằng nhau);
        nếu mã hóa không kịp cho upload → f_en = C_en/(τ − T_remote), f_l = C_l/(τ − t_en).
        Cắt vào [dvfs_min_fraction·F, F]; không khả thi → chạy F (vi phạm deadline được ghi nhận).
        """
        cfg = self.config
        if not cfg.dvfs_enabled:
            return F_loc, F_loc
        f_min = cfg.dvfs_min_fraction * F_loc
        clip = lambda f: float(min(max(f, f_min), F_loc))
        budget = tau * (1.0 - 1e-6)   # tránh vượt deadline do làm tròn
        if budget <= 0 or C_enc + C_local <= 0:
            return F_loc, F_loc
        f_en = f_l = (C_enc + C_local) / budget
        if C_enc > 0 and t_remote > 0 and C_enc / f_en > budget - t_remote:
            b_enc = budget - t_remote
            f_en = C_enc / b_enc if b_enc > 0 else F_loc
            f_en = clip(f_en)
            rem = budget - C_enc / f_en
            f_l = C_local / rem if (C_local > 0 and rem > 0) else F_loc
        return clip(f_en), clip(f_l)

    def _compute_local_energy(self, user: User, task: Task) -> float:
        """E_local = κ·c·s·f_u². Ref: [A] Eq(12)."""
        return self.config.kappa * task.cycles_per_bit * task.size * (user.cpu_frequency ** 2)

    def _compute_offloading_energy(self, user: User, task: Task, tx_time: float) -> float:
        """E_tx = p_m·t_tx. Ref: [A] Eq(13). T0.9: không có hệ số attack."""
        _ = task
        return user.tx_power * tx_time

    # ==================== Cost ([A] §3.5 + Eq(22)) ====================

    def _price_urgency(self, tau_eff: float) -> float:
        """
        ω = 1/(1 + τ_eff) ∈ (0, 1) — [A] §3.5 yêu cầu ω_m ∈ (0,1).
        τ = 1 s → ω = 0.5 (giá trị [A] dùng trong mô phỏng); τ → 0 → ω → 1.
        """
        return float(1.0 / (1.0 + max(tau_eff, self.config.deadline_floor)))

    def _compute_mec_cost(self, mec: MECServer, task: Task, resource_level: int,
                          tau_eff: float) -> float:
        """
        g_j = b_j·exp(ω)·g_j0. Ref: [A] §3.5 (T0.7).
        T1.1: exp(ω) giờ có động lực học thật vì τ_eff động.
        MỞ RỘNG: λ_res và chuẩn hóa giga-cycles (T0.7 bước 4).
        """
        omega = self._price_urgency(tau_eff)
        resource_multiplier = 1.0 + self.config.resource_multiplier_step * resource_level
        workload_scale = task.size * task.cycles_per_bit / self.config.cost_normalization_divisor
        return mec.credibility * np.exp(omega) * mec.base_price * \
            resource_multiplier * workload_scale

    def _compute_aux_cost(self, aux: AuxVehicle, task: Task, resource_level: int,
                          tau_eff: float) -> float:
        """
        g_n = b_n·ln(1+ε_n)·ω·g_n0. Ref: [A] §3.5.
        MỞ RỘNG (T0.7): ε_n tính cả user đang yêu cầu (ε_n ≥ 1 → g_n > 0).
        """
        omega = self._price_urgency(tau_eff)
        effective_load = aux.current_load + 1
        load_factor = np.log(1 + effective_load)
        resource_multiplier = 1.0 + self.config.resource_multiplier_step * resource_level
        workload_scale = task.size * task.cycles_per_bit / self.config.cost_normalization_divisor
        return aux.credibility * load_factor * omega * aux.base_price * \
            resource_multiplier * workload_scale

    # ==================== Reward (T1.2 — Phương án 1) ====================

    def _compute_reward(self, metrics: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        r = −C [cents], C = [A] Eq(22) (λ^n đúng nghĩa) + C_sec (rủi ro rò rỉ, [B] Eq(7)).
        Ràng buộc: T ≤ τ_u ([A] 23e), E ≤ E_max ([B] C3).
            penalty_mode="fixed":    trừ λ_p·1[viol_T] + λ_pE·viol_E
            penalty_mode="external": env không trừ — agent Lagrangian (T1.3) tự cộng multiplier.
        """
        cfg = self.config
        penalty = 0.0
        if cfg.penalty_mode == "fixed":
            if metrics["viol_T"]:
                penalty += cfg.deadline_penalty
            penalty += cfg.energy_violation_penalty * metrics["viol_E"]
        components = {
            "cost_energy": metrics["cost_energy"],
            "cost_service": metrics["cost_service"],
            "cost_security": metrics["cost_security"],
            "penalty": penalty,
        }
        return float(-metrics["cost_total"] - penalty), components

    # ==================== State ====================

    def _get_state(self) -> np.ndarray:
        """
        State = [size_norm, τ_eff_norm, ω_norm] + dist(J+N) + util(J+N) + [rate_norm].
        T1.1: τ_eff và ω là deadline/độ khẩn cấp ĐỘNG (không còn chiều chết).
        """
        user = self.users[0]
        state_vector = []

        if self.task_queue[user.id]:
            task = self.task_queue[user.id][0]
            tau_eff, _ = self.effective_deadline(user, task)
            task_size_norm = min(task.size / self.config.task_size_max, 1.0)
            deadline_norm = min(tau_eff / max(self.config.task_deadline, 1e-6), 1.0)
            urgency_norm = self._price_urgency(tau_eff)  # ω ∈ (0, 1)
        else:
            task_size_norm, deadline_norm, urgency_norm = 0.0, 0.0, 0.0

        state_vector.extend([task_size_norm, deadline_norm, urgency_norm])

        max_dist = np.sqrt(self.config.sim_area_x ** 2 + self.config.sim_area_y ** 2)
        for mec in self.mec_servers:
            state_vector.append(min(user.get_distance(mec) / max_dist, 1.0))
        for aux in self.auxiliary_vehicles:
            state_vector.append(min(user.get_distance(aux) / max_dist, 1.0))

        for i in range(len(self.mec_servers)):
            util = 1.0 - self.mec_cpu_available[i] / self.config.mec_cpu_frequency
            state_vector.append(float(np.clip(util, 0.0, 1.0)))
        for i in range(len(self.auxiliary_vehicles)):
            util = 1.0 - self.aux_cpu_available[i] / self.config.aux_cpu_frequency
            state_vector.append(float(np.clip(util, 0.0, 1.0)))

        if self.mec_servers and self.task_queue[user.id]:
            dist = user.get_distance(self.mec_servers[0])
            rate = self.channel_model.get_uplink_rate_mec(dist, user.tx_power)
            state_vector.append(min(rate / 1e7, 1.0))
        else:
            state_vector.append(0.0)
        state_vector.append(min(self.attackers_in_range(user, self.config.security_range_v2v) / 3.0, 1.0))

        assert len(state_vector) == self.observation_space.shape[0], (
            f"State dim không khớp công thức 5 + 2(J+N): "
            f"{len(state_vector)} != {self.observation_space.shape[0]}")
        return np.array(state_vector, dtype=np.float32)

    # ==================== Vehicle & Task Management ====================

    def _update_vehicle_positions(self):
        for user in self.users:
            user.step_mobility(dt=self.config.dt)
            user.position[0] = user.position[0] % self.config.sim_area_x
            user.position[1] = user.position[1] % self.config.sim_area_y
        # T1.1: AUX di chuyển bền vững theo mobility model riêng (không còn
        # đổi hướng ngẫu nhiên mỗi step) — cần cho thời gian tồn tại liên kết V2V.
        for aux in self.auxiliary_vehicles:
            aux.step_mobility(dt=self.config.dt)
            aux.position[0] = aux.position[0] % self.config.sim_area_x
            aux.position[1] = aux.position[1] % self.config.sim_area_y
        for attacker in self.attack_vehicles:
            attacker.step_mobility(dt=self.config.dt)
            attacker.position[0] = attacker.position[0] % self.config.sim_area_x
            attacker.position[1] = attacker.position[1] % self.config.sim_area_y

    def _generate_tasks(self):
        """Poisson arrival. Ref: [A] §6. T0.4: một đường sinh task duy nhất (bits)."""
        arrival_prob = 1.0 - np.exp(-self.config.task_arrival_rate * self.config.dt)
        for user in self.users:
            if np.random.random() < arrival_prob:
                self.task_queue[user.id].append(Task(
                    size=np.random.uniform(self.config.task_size_min, self.config.task_size_max),
                    cycles_per_bit=self.config.task_cycles_per_bit,
                    max_delay=self.config.task_deadline,
                ))

    # ==================== Utility ====================

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_metrics(self) -> Dict[str, float]:
        if not self.task_history:
            return {}
        tasks = list(self.task_history.values())
        n = len(tasks)
        eps_values = [a.current_load for a in self.auxiliary_vehicles]
        metrics = {
            "num_tasks": n,
            "avg_delay": float(np.mean([t["delay"] for t in tasks])),
            "avg_energy": float(np.mean([t["energy_user"] for t in tasks])),
            "avg_cost": float(np.mean([t["cost_total"] for t in tasks])),
            "avg_cost_service": float(np.mean([t["cost_service"] for t in tasks])),
            "avg_cost_energy": float(np.mean([t["cost_energy"] for t in tasks])),
            "avg_cost_security": float(np.mean([t["cost_security"] for t in tasks])),
            "avg_security_risk": float(np.mean([t["security_risk"] for t in tasks])),
            "offload_ratio": float(np.mean([1.0 - t["alpha_l"] for t in tasks])),
            "aux_share": float(np.mean([float("Aux_" in t["offload_location"]) for t in tasks])),
            "mec_share": float(np.mean([float("MEC_" in t["offload_location"]) for t in tasks])),
            "avg_block_length_aux": float(np.mean([t["block_length"] for t in tasks if t["block_length"] > 0]))
                                    if any(t["block_length"] > 0 for t in tasks) else 0.0,
            "avg_secure_capacity": float(np.mean([t["secure_capacity"] for t in tasks])),
            "avg_tau_eff": float(np.mean([t["tau_eff"] for t in tasks])),
            "deadline_violation_rate": float(np.mean([float(t["viol_T"]) for t in tasks])),
            "energy_violation_rate": float(np.mean([float(t["viol_E"] > 0) for t in tasks])),
            # T1.1 bước 7: thất bại do mất liên kết ≠ thất bại do deadline ứng dụng
            "link_loss_failure_rate": float(np.mean(
                [float(t["failure_cause"] == "link_loss") for t in tasks])),
            # T1.5: metrics cạnh tranh tài nguyên
            "avg_eps_n": float(np.mean(eps_values)) if eps_values else 0.0,
            "mec_utilization": float(1.0 - self.mec_cpu_available[0] /
                                     self.config.mec_cpu_frequency) if self.mec_servers else 0.0,
            "rejected_resource_count": float(self.rejected_resource_count),
            "background_task_count": float(self.background_task_count),
        }
        return metrics
