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
from src.env.mobility import LinearMobility
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
    num_attack_vehicles: int = 2  # MỞ RỘNG: chỉ minh họa, không tham gia tính toán (T0.9)

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
    urgency_cap: float = 3.0  # MỞ RỘNG: chặn ω trong định giá exp(ω) — [A] chỉ xét τ=1s;
                              # với deadline động τ→0.05 thì exp(1/τ) phát nổ (e^20)

    # CPU & Resources — Ref: [A] Section 6 (f_j = 1 GHz, f_n = 0.5 GHz)
    mec_cpu_frequency: float = 1e9   # Hz (1 GHz)  [A] §6
    aux_cpu_frequency: float = 0.5e9  # Hz (0.5 GHz) [A] §6
    user_cpu_frequency: float = 2e9  # Hz (2 GHz) — MỞ RỘNG
    kappa: float = 1e-27  # effective capacitance E = κ·c·s·f² — MỞ RỘNG

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
    price_energy_local: float = 1.0  # λ^l = 1 cents/J [A] §6 (T1.2)
    price_energy_av: float = 0.8     # λ^n = 0.8 cents/J [A] §6 (T1.2)

    # Objective (T1.2 — Phương án 1) & ràng buộc
    varpi_1: float = 1e-5  # ϖ₁ scale secure capacity U — Ref: [B] Table I (T0.8)
    energy_budget_max: float = 0.4  # E_max (J) — [B] Table I (C3)
    # Chế độ xử lý ràng buộc:
    #   "fixed"    — env tự trừ penalty cố định (baseline / T1.2 bước 4, λ_p ∈ [2,10])
    #   "external" — env trả reward gốc + cờ vi phạm trong info; agent Lagrangian
    #                (T1.3) tự cộng −λ_T·viol_T − λ_E·viol_E với λ học được
    penalty_mode: str = "fixed"
    deadline_penalty: float = 5.0        # λ_p — MỞ RỘNG (tạm, thay bằng T1.3)
    energy_violation_penalty: float = 5.0  # cents/J vượt E_max — MỞ RỘNG (tạm, T1.3)

    # Security — hệ HAI LỚP (T0.3 Phương án 1)
    encryption_enabled: bool = True
    encryption_mode: str = "simulation"
    encryption_overhead_cycles_per_bit: float = 890.0  # 800 [A] + 90 [B]
    security_level: float = 1.0

    # Attacker model — Ref: [B] Table I, Eq(5)-(9)
    attacker_density: float = 1e-4  # ρ (vehicles/vehicle) [B] Table I (T0.2)
    frame_length_bits: float = 64.0  # N_f [B] Table I (T0.3)
    encryption_block_length_bits: float = 32.0  # N_i — biến quyết định (T3.1)
    lane_count: float = 8.0        # N_l = 8 [B] Table I
    road_max_speed_mps: float = 25.0  # theo [A]; [B] dùng 24

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
    Reward (T1.2, Phương án 1): r = −(C_u − ϖ₁·U_u) [cents], với
        C_u = λ^l·E_user + λ^n·E_AV + g_j + g_n theo [A] Eq(22).
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
                overhead_cycles_per_bit=config.encryption_overhead_cycles_per_bit,
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
        self.action_space = spaces.MultiDiscrete([
            config.alpha_levels,
            2,  # 0: MEC, 1: AUX
            config.action_levels,
        ])

        self.logger.info(f"Environment initialized with {len(self.users)} users, "
                         f"{len(self.mec_servers)} MEC, {len(self.auxiliary_vehicles)} aux vehicles")

    # ==================== Initialization Methods ====================

    def _init_users(self):
        for i in range(self.config.num_users):
            pos = np.array([
                np.random.uniform(0, self.config.sim_area_x),
                np.random.uniform(0, self.config.sim_area_y)
            ])
            velocity = np.random.uniform(15, 25)  # m/s, Ref: [A] §6
            user = User(
                id=i,
                position=pos,
                cpu_frequency=self.config.user_cpu_frequency,
                velocity=velocity,
                tx_power=self.config.user_tx_power,
                mobility_model=LinearMobility()
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
        for i in range(self.config.num_auxiliary_vehicles):
            pos = np.array([
                np.random.uniform(0, self.config.sim_area_x),
                np.random.uniform(0, self.config.sim_area_y)
            ])
            aux = AuxVehicle(
                id=i,
                position=pos,
                cpu_frequency=self.config.aux_cpu_frequency,
                base_price=self.config.aux_base_price,
                credibility=self.config.aux_credibility,
                current_load=0,
                velocity=np.random.uniform(15, 25),
                mobility_model=LinearMobility()
            )
            self.auxiliary_vehicles.append(aux)

    def _init_attack_vehicles(self):
        """T0.9: chỉ minh họa — mô hình bảo mật dùng kỳ vọng mật độ [B] Eq(8)."""
        for i in range(self.config.num_attack_vehicles):
            pos = np.array([
                np.random.uniform(0, self.config.sim_area_x),
                np.random.uniform(0, self.config.sim_area_y)
            ])
            self.attack_vehicles.append(AttackVehicle(
                id=i, position=pos, cpu_frequency=1e8,
                mobility_model=LinearMobility()
            ))

    def _get_state_dim(self) -> int:
        """
        dim = 4 + 2·(J + N): task (3) + distances (J+N) + utilization (J+N) + rate (1).
        T0.10: không còn chiều chết (dummy SNR, d_att, security_level).
        """
        j = len(self.mec_servers)
        n = len(self.auxiliary_vehicles)
        return 4 + 2 * (j + n)

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

    def _mec_feasible(self, user: User) -> bool:
        floor = self.config.resource_floor_fraction * self.config.mec_cpu_frequency
        return bool(self.mec_servers) and self.mec_servers[0].is_covered(user) \
            and self.mec_cpu_available[0] > floor

    def _aux_feasible(self, user: User) -> Tuple[bool, Optional[int]]:
        """AUX khả thi nếu trong tầm V2V, RẢNH ([A] §3.4: ≤1 user/AV) và còn tài nguyên."""
        if not self.auxiliary_vehicles:
            return False, None
        floor = self.config.resource_floor_fraction * self.config.aux_cpu_frequency
        candidates = [
            (user.get_distance(a), i) for i, a in enumerate(self.auxiliary_vehicles)
            if user.get_distance(a) <= self.config.v2v_range
            and a.current_load == 0
            and self.aux_cpu_available[i] > floor
        ]
        if not candidates:
            return False, None
        return True, min(candidates)[1]

    def action_mask(self, user: Optional[User] = None) -> np.ndarray:
        """
        T1.6: mask (flatten (alpha_levels·2·action_levels,), C-order — khớp
        np.ravel_multi_index của agent). True = khả thi.

        - α_l = 1 (level cuối) luôn khả thi (không có phần remote).
        - target = 0 (MEC) đòi hỏi trong vùng phủ + còn tài nguyên.
        - target = 1 (AUX) đòi hỏi có AUX trong tầm, rảnh, còn tài nguyên.
        """
        user = user or self.users[0]
        L, T, R = self.config.alpha_levels, 2, self.config.action_levels
        mask = np.zeros((L, T, R), dtype=bool)
        mask[L - 1, :, :] = True  # α_l = 1: luôn khả thi
        if self._mec_feasible(user):
            mask[:, 0, :] = True
        aux_ok, _ = self._aux_feasible(user)
        if aux_ok:
            mask[:, 1, :] = True
        return mask.flatten()

    # ==================== Gymnasium API (T1.4) ====================

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        for entity in [*self.users, *self.auxiliary_vehicles, *self.attack_vehicles]:
            entity.position = np.array([
                np.random.uniform(0, self.config.sim_area_x),
                np.random.uniform(0, self.config.sim_area_y)
            ])

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
        alpha_l, alpha_j, alpha_n, resource_level = self._decode_action_to_alphas(action)
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
            user, task, alpha_l, alpha_j, alpha_n, resource_level, tau_eff)

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

    def _decode_action_to_alphas(self, action: np.ndarray) -> Tuple[float, float, float, int]:
        """
        [alpha_l_level, target, resource_level] → (α_l, α_j, α_n, level).
        Ràng buộc [A] 23b-23d: α_l + α_j + α_n = 1, α_j·α_n = 0.
        """
        alpha_levels = max(int(self.config.alpha_levels), 1)
        action_arr = np.asarray(action, dtype=np.int64).flatten()
        if action_arr.size < 3:
            raise ValueError(f"Expected action with 3 dimensions, got {action_arr.size}: {action}")

        alpha_l_level = int(np.clip(action_arr[0], 0, alpha_levels - 1))
        target_selection = int(np.clip(action_arr[1], 0, 1))
        resource_level = int(np.clip(action_arr[2], 0, int(self.config.action_levels) - 1))

        scale = float(max(alpha_levels - 1, 1))
        alpha_l = float(alpha_l_level / scale)
        alpha_remote = max(0.0, 1.0 - alpha_l)

        if alpha_remote <= 1e-12:
            return 1.0, 0.0, 0.0, resource_level
        if target_selection == 0:
            return alpha_l, alpha_remote, 0.0, resource_level
        return alpha_l, 0.0, alpha_remote, resource_level

    # ==================== T1.5 — Thực thi task (chung mọi user) ====================

    def _make_subtask(self, task: Task, ratio: float) -> Optional[Task]:
        ratio = float(max(ratio, 0.0))
        if ratio <= 1e-12:
            return None
        return Task(size=task.size * ratio, cycles_per_bit=task.cycles_per_bit,
                    max_delay=task.max_delay)

    def _execute_task(self, user: User, task: Task, alpha_l: float, alpha_j: float,
                      alpha_n: float, resource_level: int, tau_eff: float) -> Dict[str, Any]:
        """
        Thực thi một task cho user bất kỳ (T1.5: tách khỏi users[0]).

        Thực thi các ràng buộc:
        - Σβ ≤ 1 ([A] 23j): cấp phát là phần trăm của pool CÒN LẠI; nếu còn lại
          dưới resource_floor_fraction → TỪ CHỐI (đếm rejected_resource_count).
        - Mỗi AUX phục vụ ≤ 1 user ([A] §3.4): chỉ AUX có current_load == 0
          mới nhận task.
        """
        cfg = self.config

        # Mã hóa tuần tự cho phần offload
        offload_size = task.size * (alpha_j + alpha_n)
        t_enc, E_enc = 0.0, 0.0
        if cfg.encryption_enabled and offload_size > 0:
            t_enc = self._estimate_encryption_overhead(offload_size)
            E_enc = self.security_overhead_model.encryption_energy(offload_size, kappa=cfg.kappa)

        energy_user = 0.0        # J — năng lượng của xe user
        energy_remote = 0.0      # J — năng lượng tính toán của AV ([A] Eq(22), λ^n)
        cost_service = 0.0       # cents — giá dịch vụ động g_j/g_n
        security_risk = 0.0
        secure_capacity = 0.0
        offload_locations: List[str] = []
        branch_delays: List[float] = []
        local_branch_delays: List[float] = []
        remote_branch_delays: List[float] = []

        # ---- Local branch ----
        local_task = self._make_subtask(task, alpha_l)
        if local_task is not None:
            d = self._compute_local_delay(user, local_task)
            branch_delays.append(d)
            local_branch_delays.append(d)
            energy_user += self._compute_local_energy(user, local_task)
            offload_locations.append("local")

        # ---- MEC branch ----
        mec_task = self._make_subtask(task, alpha_j)
        if mec_task is not None:
            done_flag = False
            if self._mec_feasible(user):
                mec = self.mec_servers[0]
                cpu_freq = max(1.0, self._get_resource_allocation(
                    self.mec_cpu_available[0], resource_level))
                bw = max(1.0, self._get_bandwidth_allocation(
                    self.mec_bw_available[0], resource_level))

                mec_delay, mec_risk, tx_time, mec_cap = self._compute_mec_delay(
                    user, mec, mec_task, cpu_freq, bw)
                energy_user += self._compute_offloading_energy(user, mec_task, tx_time)
                cost_service += self._compute_mec_cost(mec, mec_task, resource_level, tau_eff)

                security_risk += alpha_j * mec_risk
                secure_capacity += alpha_j * mec_cap
                branch_delays.append(mec_delay)
                remote_branch_delays.append(mec_delay)

                self.mec_cpu_available[0] = max(0.0, self.mec_cpu_available[0] - cpu_freq)
                self.mec_bw_available[0] = max(0.0, self.mec_bw_available[0] - bw)
                self.active_allocations.append(
                    (self.current_time + mec_delay, "mec", 0, cpu_freq, bw))
                offload_locations.append("MEC_0")
                done_flag = True
            else:
                if self.mec_servers and self.mec_servers[0].is_covered(user):
                    self.rejected_resource_count += 1  # trong vùng phủ nhưng hết tài nguyên

            if not done_flag:
                d = self._compute_local_delay(user, mec_task)
                branch_delays.append(d)
                local_branch_delays.append(d)
                energy_user += self._compute_local_energy(user, mec_task)
                offload_locations.append("local_fallback_mec")

        # ---- AUX branch ----
        aux_task = self._make_subtask(task, alpha_n)
        if aux_task is not None:
            done_flag = False
            aux_ok, aux_idx = self._aux_feasible(user)
            if aux_ok:
                aux = self.auxiliary_vehicles[aux_idx]
                cpu_freq = max(1.0, self._get_resource_allocation(
                    self.aux_cpu_available[aux_idx], resource_level))
                bw = max(1.0, self._get_bandwidth_allocation(
                    self.aux_bw_available[aux_idx], resource_level))

                aux_delay, aux_risk, tx_time, aux_cap = self._compute_v2v_delay(
                    user, aux, aux_task, cpu_freq, bw)
                energy_user += self._compute_offloading_energy(user, aux_task, tx_time)
                cost_service += self._compute_aux_cost(aux, aux_task, resource_level, tau_eff)
                # [A] Eq(22): năng lượng tính toán của AV quy tiền qua λ^n (T1.2)
                energy_remote += cfg.kappa * aux_task.size * aux_task.cycles_per_bit * cpu_freq ** 2

                security_risk += alpha_n * aux_risk
                secure_capacity += alpha_n * aux_cap
                branch_delays.append(aux_delay)
                remote_branch_delays.append(aux_delay)

                self.aux_cpu_available[aux_idx] = max(0.0, self.aux_cpu_available[aux_idx] - cpu_freq)
                self.aux_bw_available[aux_idx] = max(0.0, self.aux_bw_available[aux_idx] - bw)
                aux.current_load += 1  # [A] §3.4: giờ AUX này bận (≤ 1 user)
                self.active_allocations.append(
                    (self.current_time + aux_delay, "aux", aux_idx, cpu_freq, bw))
                offload_locations.append(f"Aux_{aux_idx}")
                done_flag = True
            else:
                near = self._nearest_aux(user)
                if near is not None and user.get_distance(near) <= cfg.v2v_range:
                    self.rejected_resource_count += 1  # trong tầm nhưng AUX bận/hết tài nguyên

            if not done_flag:
                d = self._compute_local_delay(user, aux_task)
                branch_delays.append(d)
                local_branch_delays.append(d)
                energy_user += self._compute_local_energy(user, aux_task)
                offload_locations.append("local_fallback_aux")

        # ---- Tổng hợp ----
        max_parallel = max(branch_delays) if branch_delays else 0.0
        delay = t_enc + max_parallel
        energy_user += E_enc

        # E^w (T0.6): p_w · max(0, T_remote − T_local)
        t_local_busy = max(local_branch_delays) if local_branch_delays else 0.0
        t_remote_max = max(remote_branch_delays) if remote_branch_delays else 0.0
        E_wait = cfg.user_idle_power * max(0.0, t_remote_max - t_local_busy)
        energy_user += E_wait
        E_compute_tx = energy_user - E_enc - E_wait

        # T1.2: quy đổi năng lượng thành tiền ([A] Eq(22))
        cost_energy = cfg.price_energy_local * energy_user + cfg.price_energy_av * energy_remote
        cost_total = cost_energy + cost_service

        # Ràng buộc
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
            "energy_remote": energy_remote,
            "energy_enc": E_enc,
            "energy_wait": E_wait,
            "energy_compute_tx": E_compute_tx,
            "cost_service": cost_service,
            "cost_energy": cost_energy,
            "cost_total": cost_total,
            "security_risk": security_risk,
            "secure_capacity": secure_capacity,
            "viol_T": viol_T,
            "viol_E": viol_E,
            "failure_cause": failure_cause,
            "offload_location": "+".join(offload_locations) if offload_locations else "none",
        }

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
            if not self.task_queue[user.id]:
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

    def _compute_mec_delay(self, user: User, mec: MECServer, task: Task,
                           cpu_freq: float, bandwidth: float) -> Tuple[float, float, float, float]:
        """
        t_mec = t_upload + t_compute. Ref: [A] Eq(10), Eq(12).
        T0.9: p_att KHÔNG làm giảm rate (eavesdropping thụ động theo [B]).
        Bỏ qua downlink theo [A] §3.4 / [B] §II-B.
        """
        dist = user.get_distance(mec)
        uplink_rate = self.channel_model.get_uplink_rate_mec(dist, user.tx_power)
        attack_prob, secure_capacity = self._compute_security_metrics(
            d_sender_range=self.config.security_range_v2i, link_rate_bps=uplink_rate)
        upload_time = task.size / max(uplink_rate, 1.0)
        computation_time = (task.cycles_per_bit * task.size) / cpu_freq
        return upload_time + computation_time, attack_prob, upload_time, secure_capacity

    def _compute_v2v_delay(self, user: User, aux: AuxVehicle, task: Task,
                           cpu_freq: float, bandwidth: float) -> Tuple[float, float, float, float]:
        """t_aux = t_v2v + t_compute. Ref: [A] Eq(17), Eq(18)."""
        dist = user.get_distance(aux)
        v2v_rate = self.channel_model.get_uplink_rate_v2v(dist, user.tx_power)
        attack_prob, secure_capacity = self._compute_security_metrics(
            d_sender_range=self.config.security_range_v2v, link_rate_bps=v2v_rate)
        transmission_time = task.size / max(v2v_rate, 1.0)
        computation_time = (task.cycles_per_bit * task.size) / cpu_freq
        return transmission_time + computation_time, attack_prob, transmission_time, secure_capacity

    # ==================== Security ([B] Eq(5)-(9)) ====================

    def _compute_security_metrics(self, d_sender_range: float,
                                  link_rate_bps: float) -> Tuple[float, float]:
        """
        [B] Eq(6)-(9). Ghi chú ngữ nghĩa (T0.2/T1.7):
        - d_i là BÁN KÍNH VÙNG PHÁT của người gửi (không phải khoảng cách tới đích);
          hiệu chỉnh riêng V2I/V2V (T1.7) để n^e nằm trong vùng không suy biến.
        - n^e là KỲ VỌNG từ mật độ ρ (không chặn bằng số attacker mô phỏng).
        - N_i không bị chặn bởi task size (T0.3).
        """
        n_f = max(self.config.frame_length_bits, 1.0)
        n_i = min(max(self.config.encryption_block_length_bits, 1.0), n_f)

        n_e = self.security_overhead_model.expected_attackers(
            coverage_radius_m=d_sender_range,
            attacker_density=self.config.attacker_density,
            lane_count=self.config.lane_count,
            max_speed_mps=self.config.road_max_speed_mps,
        )
        s = self.security_overhead_model.security_strength(n_i, n_f, n_e)
        p_att = self.security_overhead_model.attack_success_probability(n_i, n_f, n_e)
        u = self.security_overhead_model.secure_information_capacity(link_rate_bps, n_i, s)
        return p_att, u

    def _estimate_encryption_overhead(self, data_size_bits: float) -> float:
        """t_enc = c_en·s/f. Ref: [A] Eq(4); c_en = 890 (hai lớp, T0.3)."""
        if not self.config.encryption_enabled or self.security_module is None:
            return 0.0
        if self.config.encryption_mode == "simulation":
            return self.security_overhead_model.encryption_time(data_size_bits)
        duration, _ = self.security_module.perform_real_encryption_benchmark(1.0)
        return duration * (data_size_bits / 1024.0)

    # ==================== Energy ====================

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
        ω = 1/τ_eff, chặn tại urgency_cap (MỞ RỘNG: [A] chỉ xét τ = 1 s; với
        deadline động τ→0.05 s thì exp(1/τ) = e^20 làm giá phát nổ — T1.1).
        """
        return float(min(1.0 / max(tau_eff, self.config.deadline_floor),
                         self.config.urgency_cap))

    def _compute_mec_cost(self, mec: MECServer, task: Task, resource_level: int,
                          tau_eff: float) -> float:
        """
        g_j = b_j·exp(ω)·g_j0. Ref: [A] §3.5 (T0.7).
        T1.1: exp(ω) giờ có động lực học thật vì τ_eff động.
        MỞ RỘNG: λ_res và chuẩn hóa giga-cycles (T0.7 bước 4); chặn ω (urgency_cap).
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
        T1.2 Phương án 1: min E[Σ(C_u − ϖ₁·U_u)] → r = −(C_u − ϖ₁·U_u) [cents]
            C_u = λ^l·E_user + λ^n·E_AV + g_j + g_n     Ref: [A] Eq(22)
            U_u — secure information capacity            Ref: [B] Eq(9), ϖ₁ [B] Table I
        Ràng buộc: T ≤ τ_u ([A] 23e), E ≤ E_max ([B] C3, 0.4 J).
            penalty_mode="fixed":    trừ λ_p·1[viol_T] + λ_pE·viol_E (tạm, T1.2 bước 4)
            penalty_mode="external": KHÔNG trừ ở env — agent Lagrangian (T1.3)
                tự cộng với multiplier học được.
        Log riêng từng thành phần (T1.2 bước 5).
        """
        cfg = self.config
        secure_term = cfg.varpi_1 * max(metrics["secure_capacity"], 0.0)
        base = -(metrics["cost_total"] - secure_term)

        penalty = 0.0
        if cfg.penalty_mode == "fixed":
            if metrics["viol_T"]:
                penalty += cfg.deadline_penalty
            penalty += cfg.energy_violation_penalty * metrics["viol_E"]

        components = {
            "cost_energy": metrics["cost_energy"],
            "cost_service": metrics["cost_service"],
            "secure_term": secure_term,
            "penalty": penalty,
        }
        return float(base - penalty), components

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
            omega = 1.0 / max(tau_eff, 0.1)
            urgency_norm = min(omega / 10.0, 1.0)  # ω ∈ [1, 10] → [0.1, 1]
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

        assert len(state_vector) == self.observation_space.shape[0], (
            f"State dim không khớp công thức 4 + 2(J+N): "
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
            "avg_security_risk": float(np.mean([t["security_risk"] for t in tasks])),
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
