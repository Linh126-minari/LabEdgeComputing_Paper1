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
    #Cấu hình môi trường
    num_users: int = 5
    num_mec_servers: int = 2
    num_auxiliary_vehicles: int = 3
    num_attack_vehicles: int = 2
    
    # Khu vực mô phỏng
    sim_area_x: float = 1000.0  # meters
    sim_area_y: float = 1000.0  # meters
    
    # Task generation
    task_arrival_rate: float = 0.5  # Poisson λ (tasks/second)
    task_size_range: Tuple[float, float] = (100, 1000)  # KB
    task_cycles_per_bit: float = 1000  # cycles/bit
    task_deadline: float = 1.0  # seconds
    
    # CPU & Resources
    mec_cpu_frequency: float = 1e10  # Hz (10 GHz)
    aux_cpu_frequency: float = 5e9   # Hz (5 GHz)
    user_cpu_frequency: float = 2e9  # Hz (2 GHz)
    
    # Reward weights
    weight_delay: float = 1.0
    weight_energy: float = 0.5
    weight_cost: float = 0.3
    weight_security: float = 0.4
    weight_secure_capacity: float = 0.2
    weight_penalty: float = 100.0  # Penalty khi vi phạm deadline
    
    # Security
    encryption_enabled: bool = True
    encryption_mode: str = "simulation"  # "simulation" hoặc "emulation"
    encryption_overhead_cycles_per_bit: float = 800.0
    security_level: float = 1.0

    # Attacker & security risk model
    attack_risk_alpha: float = 3.0
    attack_risk_beta: float = 2.0
    attack_rate_penalty: float = 0.35
    attack_energy_penalty: float = 0.45
    attacker_tx_power: float = 0.3
    attacker_jamming_radius: float = 250.0
    attacker_density: float = 2e-6

    # Paper adversary model (Eq. 5-9)
    frame_length_bits: float = 8192.0
    encryption_block_length_bits: float = 2048.0
    lane_count: float = 3.0
    road_max_speed_mps: float = 25.0
    v2v_range: float = 200.0
    
    # Action discretization
    alpha_levels: int = 10  # Discretization levels for alpha_l, alpha_j, alpha_n
    action_levels: int = 10  # 0-9 resource levels
    
    # Other
    dt: float = 0.1  # Time step (seconds)
    episode_length: int = 1000  # Max steps per episode


class VehicularEdgeEnvironment(gym.Env):
    """
    Môi trường MDP cho Vehicular Edge Computing.
    
    State: Vector chứa thông tin task, khoảng cách, tài nguyên, kênh
    Action: [alpha_l_level, alpha_j_level, alpha_n_level, resource_level]
    Reward: Weighted combination của delay, energy, cost, penalty
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config: EnvironmentConfig):
        """
        Khởi tạo môi trường.
        
        Args:
            config: EnvironmentConfig object
        """
        super().__init__()
        self.config = config
        self.logger = Logger("VehicularEdgeEnv")
        
        # Track active resource allocations to dynamically release them later
        self.active_allocations: List[Tuple[float, str, int, float, float]] = []
        
        # ==================== Initialize Entities ====================
        # User vehicles
        self.users: List[User] = []
        self._init_users()
        
        # MEC servers
        self.mec_servers: List[MECServer] = []
        self._init_mec_servers()
        
        # Auxiliary vehicles
        self.auxiliary_vehicles: List[AuxVehicle] = []
        self._init_auxiliary_vehicles()

        # Attack vehicles
        self.attack_vehicles: List[AttackVehicle] = []
        self._init_attack_vehicles()
        
        # ==================== Initialize Models ====================
        self.channel_model = ChannelModel()
        self.security_module = SecurityModule(key_length=1024) if config.encryption_enabled else None
        self.security_overhead_model = SecurityOverheadModel(
            SecurityOverheadConfig(
                overhead_cycles_per_bit=config.encryption_overhead_cycles_per_bit,
                user_cpu_frequency=config.user_cpu_frequency,
                security_level=config.security_level,
                attack_rate_penalty=config.attack_rate_penalty,
                attack_energy_penalty=config.attack_energy_penalty,
                frame_length_bits=config.frame_length_bits,
                default_block_length_bits=config.encryption_block_length_bits,
                attacker_density=config.attacker_density,
                lane_count=config.lane_count,
                max_speed_mps=config.road_max_speed_mps,
            )
        )
        
        # ==================== Simulation State ====================
        self.current_time: float = 0.0
        self.step_count: int = 0
        self.episode_count: int = 0
        
        # Task management
        self.task_queue: Dict[int, List[Task]] = defaultdict(list)  # user_id -> [tasks]
        self.task_history: Dict[int, Dict[str, Any]] = {}  # task_id -> {info}
        self.next_task_id: int = 0
        
        # Resource tracking
        self.mec_cpu_available: List[float] = [config.mec_cpu_frequency] * len(self.mec_servers)
        self.mec_bw_available: List[float] = [1e7] * len(self.mec_servers)  # 10 Mbps per MEC
        
        self.aux_cpu_available: List[float] = [config.aux_cpu_frequency] * len(self.auxiliary_vehicles)
        self.aux_bw_available: List[float] = [5e6] * len(self.auxiliary_vehicles)  # 5 Mbps per aux
        
        # ==================== Gym Spaces ====================
        # State: [task_size, deadline, urgency, dist_to_mec, dist_to_aux_vehicles, 
        #         mec_load, aux_loads, channel_snr, ...]
        state_dim = self._get_state_dim()
        self.observation_space = spaces.Box(low=0, high=1, shape=(state_dim,), dtype=np.float32)
        
        # Action: [alpha_l_level, alpha_j_level, alpha_n_level, resource_level]
        # Constraint is enforced in step: local+MEC or local+AUX (not both MEC and AUX).
        self.action_space = spaces.MultiDiscrete([
            config.alpha_levels,
            2,  # 0: MEC, 1: AUX target selection
            config.action_levels,
        ])
        
        self.logger.info(f"Environment initialized with {len(self.users)} users, "
                        f"{len(self.mec_servers)} MEC, {len(self.auxiliary_vehicles)} aux vehicles")
    
    # ==================== Initialization Methods ====================
    
    def _init_users(self):
        """Khởi tạo user vehicles"""
        for i in range(self.config.num_users):
            pos = np.array([
                np.random.uniform(0, self.config.sim_area_x),
                np.random.uniform(0, self.config.sim_area_y)
            ])
            velocity = np.random.uniform(15, 25)  # m/s (54-90 km/h)
            user = User(
                id=i,
                position=pos,
                cpu_frequency=self.config.user_cpu_frequency,
                velocity=velocity,
                tx_power=0.2,  # 0.2W from paper
                mobility_model=LinearMobility()
            )
            self.users.append(user)
    
    def _init_mec_servers(self):
        """Khởi tạo MEC servers tại RSU"""
        # Đặt tại vị trí cố định
        mec_positions = [
            np.array([self.config.sim_area_x * 0.25, self.config.sim_area_y * 0.25]),
            np.array([self.config.sim_area_x * 0.75, self.config.sim_area_y * 0.75]),
        ]
        
        for i in range(min(self.config.num_mec_servers, len(mec_positions))):
            mec = MECServer(
                id=i,
                position=mec_positions[i],
                cpu_frequency=self.config.mec_cpu_frequency,
                coverage_radius=300.0,  # meters
                base_price=1.0,
                credibility=0.95
            )
            self.mec_servers.append(mec)
    
    def _init_auxiliary_vehicles(self):
        """Khởi tạo auxiliary vehicles (V2V capable)"""
        for i in range(self.config.num_auxiliary_vehicles):
            pos = np.array([
                np.random.uniform(0, self.config.sim_area_x),
                np.random.uniform(0, self.config.sim_area_y)
            ])
            aux = AuxVehicle(
                id=i,
                position=pos,
                cpu_frequency=self.config.aux_cpu_frequency,
                base_price=0.5,
                credibility=0.85,
                current_load=0
            )
            self.auxiliary_vehicles.append(aux)
    
    def _get_state_dim(self) -> int:
        """Tính toán số chiều của state vector"""
        # Task: size (norm), deadline (norm), urgency
        # Distances: to MEC (x num_mec), to aux (x num_aux)
        # Resources: MEC utilization (x num_mec), aux utilization (x num_aux)
        # Channel: SNR, rate
        # Security: min attacker distance, security_level
        dim = 3 + len(self.mec_servers) + len(self.auxiliary_vehicles) + \
              len(self.mec_servers) + len(self.auxiliary_vehicles) + 4
        return dim

    def _init_attack_vehicles(self):
        """Khởi tạo attack vehicles"""
        for i in range(self.config.num_attack_vehicles):
            pos = np.array([
                np.random.uniform(0, self.config.sim_area_x),
                np.random.uniform(0, self.config.sim_area_y)
            ])
            attacker = AttackVehicle(
                id=i,
                position=pos,
                cpu_frequency=1e8,
                tx_power=self.config.attacker_tx_power,
                jamming_radius=self.config.attacker_jamming_radius,
                mobility_model=LinearMobility()
            )
            self.attack_vehicles.append(attacker)
    
    # ==================== MDP Methods ====================
    
    def reset(self) -> np.ndarray:
        """
        Reset môi trường về trạng thái ban đầu.
        
        Returns:
            Initial state vector
        """
        # Reset positions
        for user in self.users:
            user.position = np.array([
                np.random.uniform(0, self.config.sim_area_x),
                np.random.uniform(0, self.config.sim_area_y)
            ])
        
        for aux in self.auxiliary_vehicles:
            aux.position = np.array([
                np.random.uniform(0, self.config.sim_area_x),
                np.random.uniform(0, self.config.sim_area_y)
            ])

        for attacker in self.attack_vehicles:
            attacker.position = np.array([
                np.random.uniform(0, self.config.sim_area_x),
                np.random.uniform(0, self.config.sim_area_y)
            ])
        
        # Reset state
        self.current_time = 0.0
        self.step_count = 0
        self.task_queue.clear()
        self.task_history.clear()
        self.next_task_id = 0
        self.active_allocations.clear()
        
        # Reset resources
        self.mec_cpu_available = [self.config.mec_cpu_frequency] * len(self.mec_servers)
        self.mec_bw_available = [1e7] * len(self.mec_servers)
        self.aux_cpu_available = [self.config.aux_cpu_frequency] * len(self.auxiliary_vehicles)
        self.aux_bw_available = [5e6] * len(self.auxiliary_vehicles)
        
        # Reset auxiliary vehicles
        for aux in self.auxiliary_vehicles:
            aux.current_load = 0
        
        # Generate first task for first user
        self._generate_tasks()
        
        self.episode_count += 1
        state = self._get_state()
        
        return state
    
    def _decode_action_to_alphas(self, action: np.ndarray) -> Tuple[float, float, float, int]:
        """
        Decode action into alpha ratios and resource level.

        Action format: [alpha_l_level, target_selection, resource_level]
        Constraint is enforced:
        - either local + MEC: alpha_l + alpha_j = 1, alpha_n = 0
        - or local + AUX: alpha_l + alpha_n = 1, alpha_j = 0
        """
        alpha_levels = max(int(self.config.alpha_levels), 1)
        action_arr = np.asarray(action, dtype=np.int64).flatten()
        if action_arr.size < 3:
            raise ValueError(f"Expected action with 3 dimensions, got {action_arr.size}: {action}")

        alpha_l_level = int(np.clip(action_arr[0], 0, alpha_levels - 1))
        target_selection = int(np.clip(action_arr[1], 0, 1))  # 0: MEC, 1: AUX
        resource_level = int(np.clip(action_arr[2], 0, int(self.config.action_levels) - 1))

        # alpha_l is directly controlled by first dimension.
        scale = float(max(alpha_levels - 1, 1))
        alpha_l = float(alpha_l_level / scale)
        alpha_remote = max(0.0, 1.0 - alpha_l)

        if alpha_remote <= 1e-12:
            return 1.0, 0.0, 0.0, resource_level

        if target_selection == 0:
            return alpha_l, alpha_remote, 0.0, resource_level

        return alpha_l, 0.0, alpha_remote, resource_level

    def _release_resources(self):
        """Release CPU and bandwidth resources for tasks that have finished execution."""
        still_active = []
        for alloc in self.active_allocations:
            release_time, node_type, node_idx, cpu_freq, bw = alloc
            if self.current_time >= release_time:
                if node_type == "mec":
                    self.mec_cpu_available[node_idx] = min(
                        self.config.mec_cpu_frequency,
                        self.mec_cpu_available[node_idx] + cpu_freq
                    )
                    self.mec_bw_available[node_idx] = min(
                        1e7,  # max MEC bandwidth pool
                        self.mec_bw_available[node_idx] + bw
                    )
                elif node_type == "aux":
                    self.aux_cpu_available[node_idx] = min(
                        self.config.aux_cpu_frequency,
                        self.aux_cpu_available[node_idx] + cpu_freq
                    )
                    self.aux_bw_available[node_idx] = min(
                        5e6,  # max AUX bandwidth pool
                        self.aux_bw_available[node_idx] + bw
                    )
                    self.auxiliary_vehicles[node_idx].current_load = max(
                        0,
                        self.auxiliary_vehicles[node_idx].current_load - 1
                    )
            else:
                still_active.append(alloc)
        self.active_allocations = still_active

    def _make_subtask(self, task: Task, ratio: float) -> Optional[Task]:
        """Create a proportional subtask for a given offloading ratio."""
        ratio = float(max(ratio, 0.0))
        if ratio <= 1e-12:
            return None
        return Task(
            size=task.size * ratio,
            cycles_per_bit=task.cycles_per_bit,
            max_delay=task.max_delay,
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Thực hiện một step trong môi trường.
        
        Args:
            action: [alpha_l_level, target_selection, resource_level]
        
        Returns:
            (state, reward, done, info)
        """
        # ==================== Decode Action ====================
        alpha_l, alpha_j, alpha_n, resource_level = self._decode_action_to_alphas(action)
        
        # ==================== Get Current User & Task ====================
        user = self.users[0]
        
        # Event-driven task pop: advance time/positions until a task arrives
        while not self.task_queue[user.id]:
            self._update_vehicle_positions()
            self._generate_tasks()
            self.current_time += self.config.dt
            self.step_count += 1
            self._release_resources()
            
            if self.step_count >= self.config.episode_length:
                done = True
                return self._get_state(), 0.0, done, {"error": "no_task_timeout"}
        
        task = self.task_queue[user.id].pop(0)
        
        # ==================== Decrypt & Encrypt Sequential Overhead ====================
        # Encryption is performed sequentially at the UV ONLY for the offloaded task portion.
        offload_size = task.size * (alpha_j + alpha_n)
        t_enc = 0.0
        E_enc = 0.0
        if self.config.encryption_enabled and offload_size > 0:
            t_enc = self._estimate_encryption_overhead(offload_size)
            E_enc = self.security_overhead_model.encryption_energy(offload_size)
        
        # ==================== Decide Offloading (Parallel Execution) ====================
        energy = 0.0
        cost = 0.0
        security_risk = 0.0
        secure_capacity = 0.0
        offload_locations: List[str] = []
        branch_delays: List[float] = []

        # 1) Local branch: alpha_l
        local_task = self._make_subtask(task, alpha_l)
        if local_task is not None:
            local_delay = self._compute_local_delay(user, local_task)
            local_energy = self._compute_local_energy(user, local_task)
            branch_delays.append(local_delay)
            energy += local_energy
            offload_locations.append("local")

        # 2) MEC branch: alpha_j
        mec_task = self._make_subtask(task, alpha_j)
        if mec_task is not None:
            mec_done = False
            if self.mec_servers:
                mec = self.mec_servers[0]
                if mec.is_covered(user):
                    cpu_alloc = self._get_resource_allocation(self.mec_cpu_available[0], resource_level)
                    bw_alloc = self._get_bandwidth_allocation(self.mec_bw_available[0], resource_level)

                    # CPU frequency allocation is independent of alpha to fix the cancellation bug.
                    cpu_freq = max(1.0, cpu_alloc)
                    bw = max(1.0, bw_alloc)

                    mec_delay, mec_risk, tx_time, mec_secure_capacity = self._compute_mec_delay(
                        user, mec, mec_task, cpu_freq, bw
                    )
                    mec_energy = self._compute_offloading_energy(user, mec_task, tx_time, mec_risk)
                    mec_cost = self._compute_mec_cost(mec, mec_task, resource_level)

                    energy += mec_energy
                    cost += mec_cost
                    security_risk += alpha_j * mec_risk
                    secure_capacity += alpha_j * mec_secure_capacity
                    branch_delays.append(mec_delay)

                    # Reserve resources and track allocation
                    self.mec_cpu_available[0] = max(0.0, self.mec_cpu_available[0] - cpu_freq)
                    self.mec_bw_available[0] = max(0.0, self.mec_bw_available[0] - bw)
                    self.active_allocations.append((
                        self.current_time + mec_delay,
                        "mec",
                        0,
                        cpu_freq,
                        bw
                    ))
                    
                    offload_locations.append("MEC_0")
                    mec_done = True

            if not mec_done:
                # Fallback: process MEC fraction locally when MEC is unavailable/out-of-coverage.
                fb_delay = self._compute_local_delay(user, mec_task)
                fb_energy = self._compute_local_energy(user, mec_task)
                energy += fb_energy
                branch_delays.append(fb_delay)
                offload_locations.append("local_fallback_mec")

        # 3) Auxiliary branch: alpha_n (route to nearest in-range auxiliary vehicle)
        aux_task = self._make_subtask(task, alpha_n)
        if aux_task is not None:
            aux_done = False
            if self.auxiliary_vehicles:
                nearest_aux_idx = int(np.argmin([user.get_distance(aux) for aux in self.auxiliary_vehicles]))
                aux = self.auxiliary_vehicles[nearest_aux_idx]
                if user.get_distance(aux) <= self.config.v2v_range:
                    cpu_alloc = self._get_resource_allocation(self.aux_cpu_available[nearest_aux_idx], resource_level)
                    bw_alloc = self._get_bandwidth_allocation(self.aux_bw_available[nearest_aux_idx], resource_level)
                    
                    # CPU frequency allocation is independent of alpha to fix the cancellation bug.
                    cpu_freq = max(1.0, cpu_alloc)
                    bw = max(1.0, bw_alloc)

                    aux_delay, aux_risk, tx_time, aux_secure_capacity = self._compute_v2v_delay(
                        user, aux, aux_task, cpu_freq, bw
                    )
                    aux_energy = self._compute_offloading_energy(user, aux_task, tx_time, aux_risk)
                    aux_cost = self._compute_aux_cost(aux, aux_task, resource_level)

                    energy += aux_energy
                    cost += aux_cost
                    security_risk += alpha_n * aux_risk
                    secure_capacity += alpha_n * aux_secure_capacity
                    branch_delays.append(aux_delay)

                    # Reserve resources and track allocation
                    self.aux_cpu_available[nearest_aux_idx] = max(0.0, self.aux_cpu_available[nearest_aux_idx] - cpu_freq)
                    self.aux_bw_available[nearest_aux_idx] = max(0.0, self.aux_bw_available[nearest_aux_idx] - bw)
                    aux.current_load += 1
                    self.active_allocations.append((
                        self.current_time + aux_delay,
                        "aux",
                        nearest_aux_idx,
                        cpu_freq,
                        bw
                    ))
                    
                    offload_locations.append(f"Aux_{nearest_aux_idx}")
                    aux_done = True

            if not aux_done:
                # Fallback: process auxiliary fraction locally when no AV is in range.
                fb_delay = self._compute_local_delay(user, aux_task)
                fb_energy = self._compute_local_energy(user, aux_task)
                energy += fb_energy
                branch_delays.append(fb_delay)
                offload_locations.append("local_fallback_aux")

        # E2E delay is sequential encryption delay + max parallel branch delay
        max_parallel_delay = max(branch_delays) if branch_delays else 0.0
        delay = t_enc + max_parallel_delay
        
        # Add sequential encryption energy
        energy += E_enc

        offload_location = "+".join(offload_locations) if offload_locations else "none"
        
        # Check deadline violation
        deadline_violated = delay > task.max_delay
        
        # ==================== Update Environment State ====================
        # Move vehicles
        self._update_vehicle_positions()
        
        # Generate new tasks
        self._generate_tasks()
        
        # Record task execution
        task_id = self.next_task_id
        self.task_history[task_id] = {
            "user_id": user.id,
            "delay": delay,
            "energy": energy,
            "cost": cost,
            "security_risk": security_risk,
            "secure_capacity": secure_capacity,
            "deadline_violated": deadline_violated,
            "offload_location": offload_location,
            "alpha_l": alpha_l,
            "alpha_j": alpha_j,
            "alpha_n": alpha_n,
            "timestamp": self.current_time
        }
        self.next_task_id += 1
        
        # ==================== Compute Reward ====================
        reward = self._compute_reward(delay, energy, cost, security_risk, secure_capacity, deadline_violated, task)
        
        # ==================== Update Time ====================
        self.current_time += self.config.dt
        self.step_count += 1
        
        # ==================== Release resources at this timestep ====================
        self._release_resources()
        
        # ==================== Check Done ====================
        done = self.step_count >= self.config.episode_length
        
        # ==================== Get Next State ====================
        state = self._get_state()
        
        # ==================== Info Dict ====================
        info = {
            "alpha": [alpha_l, alpha_j, alpha_n],
            "task_delay": delay,
            "task_energy": energy,
            "task_cost": cost,
            "task_security_risk": security_risk,
            "task_secure_capacity": secure_capacity,
            "deadline_violated": deadline_violated,
            "offload_location": offload_location,
            "num_tasks_completed": len(self.task_history)
        }
        
        return state, reward, done, info

        offload_location = "+".join(offload_locations) if offload_locations else "none"
        
        # Check deadline violation
        deadline_violated = delay > task.max_delay
        
        # ==================== Update Environment State ====================
        # Move vehicles
        self._update_vehicle_positions()
        
        # Generate new tasks
        self._generate_tasks()
        
        # Record task execution
        task_id = self.next_task_id
        self.task_history[task_id] = {
            "user_id": user.id,
            "delay": delay,
            "energy": energy,
            "cost": cost,
            "security_risk": security_risk,
            "secure_capacity": secure_capacity,
            "deadline_violated": deadline_violated,
            "offload_location": offload_location,
            "alpha_l": alpha_l,
            "alpha_j": alpha_j,
            "alpha_n": alpha_n,
            "timestamp": self.current_time
        }
        self.next_task_id += 1
        
        # ==================== Compute Reward ====================
        reward = self._compute_reward(delay, energy, cost, security_risk, secure_capacity, deadline_violated, task)
        
        # ==================== Update Time ====================
        self.current_time += self.config.dt
        self.step_count += 1
        
        # ==================== Check Done ====================
        done = self.step_count >= self.config.episode_length
        
        # ==================== Get Next State ====================
        state = self._get_state()
        
        # ==================== Info Dict ====================
        info = {
            "alpha": [alpha_l, alpha_j, alpha_n],
            "task_delay": delay,
            "task_energy": energy,
            "task_cost": cost,
            "task_security_risk": security_risk,
            "task_secure_capacity": secure_capacity,
            "deadline_violated": deadline_violated,
            "offload_location": offload_location,
            "num_tasks_completed": len(self.task_history)
        }
        
        return state, reward, done, info
    
    # ==================== Delay Computation ====================
    
    def _compute_local_delay(self, user: User, task: Task) -> float:
        computation_time = (task.cycles_per_bit * task.size) / user.cpu_frequency
        return computation_time
    
    def _compute_mec_delay(self, user: User, mec: MECServer, task: Task,
                           cpu_freq: float, bandwidth: float) -> Tuple[float, float, float, float]:
        """
        Tính độ trễ offload tới MEC.
        
        Partial MEC component:
        t_mec = t_upload + t_compute, với s_j = alpha_j * s_m (mã hóa đã tính tuần tự trước đó)
        """
        # Upload time: t_upload = s_m / r_ul
        dist_to_mec = user.get_distance(mec)
        nominal_rate = self.channel_model.get_uplink_rate_mec(dist_to_mec, user.tx_power)
        attack_prob, secure_capacity = self._compute_security_metrics(
            task=task,
            communication_radius=mec.coverage_radius,
            link_rate_bps=nominal_rate,
        )
        uplink_rate = self.security_overhead_model.effective_rate(nominal_rate, attack_prob)
        upload_time = task.size / max(uplink_rate, 1.0)  # bits to bit/s
        
        # Computation time at MEC
        computation_time = (task.cycles_per_bit * task.size) / cpu_freq
        
        total_delay = upload_time + computation_time
        return total_delay, attack_prob, upload_time, secure_capacity
    
    def _compute_v2v_delay(self, user: User, aux: AuxVehicle, task: Task,
                           cpu_freq: float, bandwidth: float) -> Tuple[float, float, float, float]:
        """
        Tính độ trễ offload tới auxiliary vehicle (V2V).
        """
        # V2V transmission rate
        dist_to_aux = user.get_distance(aux)
        nominal_rate = self.channel_model.get_uplink_rate_v2v(dist_to_aux, user.tx_power)
        attack_prob, secure_capacity = self._compute_security_metrics(
            task=task,
            communication_radius=self.config.v2v_range,
            link_rate_bps=nominal_rate,
        )
        v2v_rate = self.security_overhead_model.effective_rate(nominal_rate, attack_prob)
        transmission_time = task.size / max(v2v_rate, 1.0)
        
        # Computation time
        computation_time = (task.cycles_per_bit * task.size) / cpu_freq
        
        total_delay = transmission_time + computation_time
        return total_delay, attack_prob, transmission_time, secure_capacity

    def _resolve_attacker_density(self) -> float:
        """Ưu tiên mật độ attacker từ config, fallback theo số attacker trong map."""
        if self.config.attacker_density > 0.0:
            return self.config.attacker_density

        area = max(self.config.sim_area_x * self.config.sim_area_y, 1.0)
        return float(len(self.attack_vehicles) / area)

    def _compute_security_metrics(
        self,
        task: Task,
        communication_radius: float,
        link_rate_bps: float,
    ) -> Tuple[float, float]:
        """
        Áp dụng mô hình attacker trong paper (Eq. 6-9):
        - Phi_i = (N_f - N_i) / N_f
        - n_i^e = pi * d_i^2 * rho * (N_l / (3 v_max))
        - S_i = (N_i / N_f)^(n_i^e)
        - p_att = 1 - S_i
        - U_i = (R_i / (8N_i)) * log2(8N_i) * S_i
        """
        n_f = max(self.config.frame_length_bits, 1.0)
        n_i = min(max(self.config.encryption_block_length_bits, 1.0), n_f, task.size)

        rho = self._resolve_attacker_density()
        n_attackers_expected = self.security_overhead_model.expected_attackers(
            coverage_radius_m=communication_radius,
            attacker_density=rho,
            lane_count=self.config.lane_count,
            max_speed_mps=self.config.road_max_speed_mps,
        )

        if self.attack_vehicles:
            n_attackers_expected = min(n_attackers_expected, float(len(self.attack_vehicles)))

        security_strength = self.security_overhead_model.security_strength(
            block_length_bits=n_i,
            frame_length_bits=n_f,
            n_attackers=n_attackers_expected,
        )
        attack_prob = self.security_overhead_model.attack_success_probability(
            block_length_bits=n_i,
            frame_length_bits=n_f,
            n_attackers=n_attackers_expected,
        )
        secure_capacity = self.security_overhead_model.secure_information_capacity(
            link_rate_bps=link_rate_bps,
            block_length_bits=n_i,
            security_strength=security_strength,
        )

        return attack_prob, secure_capacity

    def _get_attack_success_probability(self, user: User) -> float:
        """Compat wrapper: lấy p_att cho task/range mặc định."""
        _ = user
        dummy_task = Task(
            size=self.config.encryption_block_length_bits,
            cycles_per_bit=self.config.task_cycles_per_bit,
            max_delay=self.config.task_deadline,
        )
        attack_prob, _ = self._compute_security_metrics(
            task=dummy_task,
            communication_radius=self.config.attacker_jamming_radius,
            link_rate_bps=1.0,
        )
        return float(attack_prob)
    
    def _estimate_encryption_overhead(self, data_size_bits: float) -> float:
        """
        Ước lượng thời gian mã hóa (simulation mode).
        
        t_encrypt = c_en * s_m / f_m
        
        Ref: Eq 4 (paper)
        """
        if not self.config.encryption_enabled or self.security_module is None:
            return 0.0
        
        if self.config.encryption_mode == "simulation":
            encryption_time = self.security_overhead_model.encryption_time(data_size_bits)
            return encryption_time
        else:
            # emulation mode: run actual encryption
            duration, _ = self.security_module.perform_real_encryption_benchmark(1.0)
            # Scale to actual data size (rough approximation)
            return duration * (data_size_bits / 1024.0)
    
    # ==================== Energy Computation ====================
    
    def _compute_local_energy(self, user: User, task: Task) -> float:
        """
        Tính năng lượng tiêu thụ cục bộ.
        
        E_local = P * t_local (nếu có power model)
        Simplified: assume computing power ~ 1W
        """
        computation_time = (task.cycles_per_bit * task.size) / user.cpu_frequency
        power = 1.0  # Watts (simplified)
        return power * computation_time
    
    def _compute_offloading_energy(self, user: User, task: Task,
                                   tx_time: float, attack_probability: float = 0.0) -> float:
        """
        Tính năng lượng tiêu thụ khi offload.
        
        E_offload = P_tx * t_transmission (mã hóa đã tính tuần tự trước đó)
        """
        nominal_tx_energy = user.tx_power * tx_time
        effective_tx_energy = self.security_overhead_model.effective_tx_energy(
            nominal_tx_energy,
            attack_probability,
        )
        return effective_tx_energy
    
    # ==================== Cost Computation ====================
    
    def _compute_mec_cost(self, mec: MECServer, task: Task, resource_level: int) -> float:
        """
        Tính chi phí sử dụng MEC.
        
        Partial MEC cost component:
        Cost_j \\propto workload_j * urgency * resource_level,
        với workload_j = s_j * c_m và s_j = alpha_j * s_m.
        """
        # Urgency (deadline sensitivity)
        urgency = 1.0 / max(task.max_delay, 0.1)
        
        # Resource multiplier
        resource_multiplier = 1.0 + 0.1 * resource_level  # 1.0 to 2.0

        # Scale cost with partial offloaded workload.
        workload_cycles = task.size * task.cycles_per_bit
        workload_scale = workload_cycles / 1e9  # normalized to giga-cycles

        cost = mec.base_price * urgency * resource_multiplier * workload_scale
        return cost
    
    def _compute_aux_cost(self, aux: AuxVehicle, task: Task, resource_level: int) -> float:
        """
        Tính chi phí sử dụng auxiliary vehicle.
        
        Partial AUX cost component:
        g_n = b_n * ln(1 + ε_n) * ω_m * g_n0 * workload_n,
        với workload_n = s_n * c_m và s_n = alpha_n * s_m.
        
        Ref: Eq 478 (paper)
        """
        urgency = 1.0 / max(task.max_delay, 0.1)
        load_factor = np.log(1 + aux.current_load)
        resource_multiplier = 1.0 + 0.1 * resource_level

        workload_cycles = task.size * task.cycles_per_bit
        workload_scale = workload_cycles / 1e9  # normalized to giga-cycles

        cost = (
            aux.credibility * load_factor * urgency * aux.base_price *
            resource_multiplier * workload_scale
        )
        return cost
    
    # ==================== Reward Computation ====================
    
    def _compute_reward(self, delay: float, energy: float, cost: float, security_risk: float,
                       secure_capacity: float, deadline_violated: bool, task: Task) -> float:
        """
        Tính reward dựa trên weighted combination.
        
        reward = -(w_d * delay + w_e * energy + w_c * cost + penalty)
        """
        # Normalized values (to 0-1 range)
        delay_norm = min(delay / task.max_delay, 1.0) if task.max_delay > 0 else 1.0
        energy_norm = min(energy / 10.0, 1.0)  # Assume max 10J per task
        cost_norm = min(cost / 5.0, 1.0)  # Assume max cost 5 units
        security_norm = min(max(security_risk, 0.0), 1.0)
        secure_capacity_norm = float(np.tanh(max(secure_capacity, 0.0)))
        
        # Weighted combination
        reward = -(
            self.config.weight_delay * delay_norm +
            self.config.weight_energy * energy_norm +
            self.config.weight_cost * cost_norm +
            self.config.weight_security * security_norm
        )

        reward += self.config.weight_secure_capacity * secure_capacity_norm
        
        # Penalty for deadline violation
        if deadline_violated:
            reward -= self.config.weight_penalty
        else:
            # Bonus for meeting deadline
            reward += 1.0
        
        return float(reward)
    
    # ==================== State Extraction ====================
    
    def _get_state(self) -> np.ndarray:
        """
        Trích xuất state vector.
        
        State = [task_size_norm, deadline_norm, urgency,
                 dist_to_mec[0..n], dist_to_aux[0..n],
                 mec_utilization[0..n], aux_utilization[0..n],
                 snr, transmission_rate_norm]
        """
        user = self.users[0]  # Focus on first user for now
        
        state_vector = []
        
        # Task info (normalized to 0-1)
        if self.task_queue[user.id]:
            task = self.task_queue[user.id][0]
            task_size_norm = min(task.size / (500 * 1024 * 8), 1.0)  # 500 KB max
            deadline_norm = min(task.max_delay / 2.0, 1.0)  # 2s max
            urgency = min(1.0 / max(task.max_delay, 0.1), 1.0)
        else:
            task_size_norm = 0.0
            deadline_norm = 0.0
            urgency = 0.0
        
        state_vector.extend([task_size_norm, deadline_norm, urgency])
        
        # Distances to MEC servers (normalized to 0-1)
        max_dist = np.sqrt(self.config.sim_area_x**2 + self.config.sim_area_y**2)
        for mec in self.mec_servers:
            dist = user.get_distance(mec)
            state_vector.append(min(dist / max_dist, 1.0))
        
        # Distances to auxiliary vehicles
        for aux in self.auxiliary_vehicles:
            dist = user.get_distance(aux)
            state_vector.append(min(dist / max_dist, 1.0))
        
        # Resource utilization at MEC
        for i, mec in enumerate(self.mec_servers):
            utilization = 1.0 - (self.mec_cpu_available[i] / self.config.mec_cpu_frequency)
            state_vector.append(min(utilization, 1.0))
        
        # Resource utilization at auxiliary vehicles
        for i, aux in enumerate(self.auxiliary_vehicles):
            utilization = 1.0 - (self.aux_cpu_available[i] / self.config.aux_cpu_frequency)
            state_vector.append(min(utilization, 1.0))
        
        # Channel info (SNR, rate)
        if self.mec_servers and self.task_queue[user.id]:
            mec = self.mec_servers[0]
            dist = user.get_distance(mec)
            uplink_rate = self.channel_model.get_uplink_rate_mec(dist, user.tx_power)
            rate_norm = min(uplink_rate / 1e7, 1.0)  # 10 Mbps max
            state_vector.append(rate_norm)
        else:
            state_vector.append(0.0)
        
        # Dummy SNR (normalized)
        state_vector.append(0.5)

        # Security context in state
        if self.attack_vehicles:
            d_min = min(user.get_distance(attacker) for attacker in self.attack_vehicles)
            max_dist = np.sqrt(self.config.sim_area_x**2 + self.config.sim_area_y**2)
            state_vector.append(min(d_min / max_dist, 1.0))
        else:
            state_vector.append(1.0)
        state_vector.append(min(max(self.config.security_level, 0.0), 1.0))
        
        # Pad to match observation space
        while len(state_vector) < self.observation_space.shape[0]:
            state_vector.append(0.0)
        
        return np.array(state_vector[:self.observation_space.shape[0]], dtype=np.float32)
    
    # ==================== Vehicle & Task Management ====================
    
    def _update_vehicle_positions(self):
        """
        Cập nhật vị trí tất cả xe.
        """
        for user in self.users:
            user.step_mobility(dt=self.config.dt)
            
            # Wrap around (vehicles moving out of bounds)
            user.position[0] = user.position[0] % self.config.sim_area_x
            user.position[1] = user.position[1] % self.config.sim_area_y
        
        # Auxiliary vehicles also move (at constant velocity)
        for aux in self.auxiliary_vehicles:
            # Simple linear motion
            vel = np.array([np.cos(np.random.random() * 2 * np.pi),
                           np.sin(np.random.random() * 2 * np.pi)]) * 15.0
            aux.position += vel * self.config.dt
            
            # Wrap around
            aux.position[0] = aux.position[0] % self.config.sim_area_x
            aux.position[1] = aux.position[1] % self.config.sim_area_y

        # Attack vehicles move
        for attacker in self.attack_vehicles:
            attacker.step_mobility(dt=self.config.dt)
            attacker.position[0] = attacker.position[0] % self.config.sim_area_x
            attacker.position[1] = attacker.position[1] % self.config.sim_area_y
    
    def _generate_tasks(self):
        """
        Sinh tasks mới theo Poisson process.
        """
        for user in self.users:
            # Poisson arrival
            arrival_prob = self.config.task_arrival_rate * self.config.dt
            if np.random.random() < arrival_prob:
                # Generate task
                size_kb = np.random.uniform(
                    self.config.task_size_range[0],
                    self.config.task_size_range[1]
                )
                size_bits = size_kb * 1024 * 8
                
                task = Task(
                    size=size_bits,
                    cycles_per_bit=self.config.task_cycles_per_bit,
                    max_delay=self.config.task_deadline
                )
                
                self.task_queue[user.id].append(task)
    
    # ==================== Resource Allocation ====================
    
    def _get_resource_allocation(self, available_cpu: float, level: int) -> float:
        """
        Chuyển đổi resource level (0-9) sang CPU frequency allocation.
        
        level 0 → 10% available_cpu
        level 9 → 100% available_cpu
        """
        allocation_percent = 0.1 + 0.1 * min(level, 9)
        return available_cpu * allocation_percent
    
    def _get_bandwidth_allocation(self, available_bw: float, level: int) -> float:
        """
        Chuyển đổi resource level sang bandwidth allocation.
        """
        allocation_percent = 0.1 + 0.1 * min(level, 9)
        return available_bw * allocation_percent
    
    # ==================== Utility Methods ====================
    
    def render(self, mode='human'):
        """
        Render visualization (optional).
        """
        # TODO: Implement visualization
        pass
    
    def close(self):
        """Close environment"""
        pass
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Trích xuất các metrics từ episode hiện tại.
        
        Returns:
            Dictionary chứa: average_delay, average_energy, deadline_violation_rate, v.v.
        """
        if not self.task_history:
            return {}
        
        tasks = list(self.task_history.values())
        
        metrics = {
            "num_tasks": len(tasks),
            "avg_delay": float(np.mean([t["delay"] for t in tasks])),
            "avg_energy": float(np.mean([t["energy"] for t in tasks])),
            "avg_cost": float(np.mean([t["cost"] for t in tasks])),
            "avg_security_risk": float(np.mean([t.get("security_risk", 0.0) for t in tasks])),
            "avg_secure_capacity": float(np.mean([t.get("secure_capacity", 0.0) for t in tasks])),
            "deadline_violation_rate": float(np.mean([float(t["deadline_violated"]) for t in tasks])),
        }
        
        return metrics
# Mobility model
from src.env.mobility import LinearMobility

# Security
from src.security.crypto_engine import SecurityModule

# Algorithms (convex optimization, selection nếu có)
# from src.algorithms.selection import OffloadingSelector  # optional

# Config & logging
from omegaconf import DictConfig
from src.utils.logger import Logger