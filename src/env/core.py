"""
Vehicular Edge Computing Environment (MDP)
==========================================
Mô phỏng hệ thống Edge Computing cho xe tự động với:
- Xe người dùng sinh task tính toán
- MEC servers tại RSU
- Auxiliary vehicles (V2V collaboration)
- Mô hình truyền, di chuyển, bảo mật

Tham khảo Paper: CSO-Sim-Framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List
from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces

from src.env.entities import User, MECServer, AuxVehicle, Task, ComputingNode
from src.env.channel import ChannelModel
from src.env.mobility import LinearMobility
from src.security.crypto_engine import SecurityModule
from src.utils.logger import Logger


@dataclass
class EnvironmentConfig:
    """Cấu hình môi trường"""
    num_users: int = 5
    num_mec_servers: int = 2
    num_auxiliary_vehicles: int = 3
    
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
    weight_penalty: float = 100.0  # Penalty khi vi phạm deadline
    
    # Security
    encryption_enabled: bool = True
    encryption_mode: str = "simulation"  # "simulation" hoặc "emulation"
    
    # Action discretization
    action_levels: int = 10  # 0-9 resource levels
    
    # Other
    dt: float = 0.1  # Time step (seconds)
    episode_length: int = 1000  # Max steps per episode


class VehicularEdgeEnvironment(gym.Env):
    """
    Môi trường MDP cho Vehicular Edge Computing.
    
    State: Vector chứa thông tin task, khoảng cách, tài nguyên, kênh
    Action: Quyết định offload (target node) + mức tài nguyên (discretized)
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
        
        # ==================== Initialize Models ====================
        self.channel_model = ChannelModel()
        self.security_module = SecurityModule(key_length=1024) if config.encryption_enabled else None
        
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
        
        # Action: [offload_target (0=local, 1=MEC, 2+=AuxVehicles), resource_level]
        num_targets = 1 + 1 + len(self.auxiliary_vehicles)  # local + MEC + aux vehicles
        self.action_space = spaces.MultiDiscrete([num_targets, config.action_levels])
        
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
        dim = 3 + len(self.mec_servers) + len(self.auxiliary_vehicles) + \
              len(self.mec_servers) + len(self.auxiliary_vehicles) + 2
        return dim
    
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
        
        # Reset state
        self.current_time = 0.0
        self.step_count = 0
        self.task_queue.clear()
        self.task_history.clear()
        self.next_task_id = 0
        
        # Reset resources
        self.mec_cpu_available = [self.config.mec_cpu_frequency] * len(self.mec_servers)
        self.aux_cpu_available = [self.config.aux_cpu_frequency] * len(self.auxiliary_vehicles)
        
        # Reset auxiliary vehicles
        for aux in self.auxiliary_vehicles:
            aux.current_load = 0
        
        # Generate first task for first user
        self._generate_tasks()
        
        self.episode_count += 1
        state = self._get_state()
        
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Thực hiện một step trong môi trường.
        
        Args:
            action: [offload_target, resource_level]
                offload_target: 0=local, 1=MEC[0], 2=MEC[1], 2+i=Aux[i]
                resource_level: 0-9 (10 levels của tài nguyên allocation)
        
        Returns:
            (state, reward, done, info)
        """
        # ==================== Decode Action ====================
        offload_target = int(action[0])
        resource_level = int(action[1])
        
        # ==================== Get Current User & Task ====================
        # Giả sử chúng ta xử lý task của user đầu tiên
        # Có thể mở rộng để multi-user sau
        user = self.users[0]
        
        if not self.task_queue[user.id]:
            # Không có task, generate một task mới
            self._generate_tasks()
        
        if self.task_queue[user.id]:
            task = self.task_queue[user.id].pop(0)
        else:
            # Không thể sinh task, trả về penalty
            reward = -10.0
            done = self.step_count >= self.config.episode_length
            return self._get_state(), reward, done, {"error": "no_task"}
        
        # ==================== Decide Offloading ====================
        selected_node = None
        delay = 0.0
        energy = 0.0
        cost = 0.0
        deadline_violated = False
        offload_location = None
        
        if offload_target == 0:
            # Local computation
            delay = self._compute_local_delay(user, task)
            energy = self._compute_local_energy(user, task)
            cost = 0.0
            offload_location = "local"
        
        elif offload_target == 1:
            # Offload to MEC server (first MEC)
            if self.mec_servers:
                mec = self.mec_servers[0]
                
                # Check coverage
                if mec.is_covered(user):
                    # Allocate resources
                    cpu_freq = self._get_resource_allocation(
                        self.mec_cpu_available[0], resource_level
                    )
                    bw = self._get_bandwidth_allocation(
                        self.mec_bw_available[0], resource_level
                    )
                    
                    delay = self._compute_mec_delay(user, mec, task, cpu_freq, bw)
                    energy = self._compute_offloading_energy(user, task, delay)
                    cost = self._compute_mec_cost(mec, task, resource_level)
                    
                    # Update resource availability
                    self.mec_cpu_available[0] -= cpu_freq
                    self.mec_bw_available[0] -= bw
                    
                    offload_location = "MEC_0"
                else:
                    # Out of coverage, fallback to local
                    delay = self._compute_local_delay(user, task)
                    energy = self._compute_local_energy(user, task)
                    cost = 0.0
                    offload_location = "local_fallback"
        
        else:
            # Offload to auxiliary vehicle
            aux_idx = offload_target - 2
            if aux_idx < len(self.auxiliary_vehicles):
                aux = self.auxiliary_vehicles[aux_idx]
                
                # Check distance (V2V range)
                dist = user.get_distance(aux)
                if dist <= 200.0:  # V2V range
                    cpu_freq = self._get_resource_allocation(
                        self.aux_cpu_available[aux_idx], resource_level
                    )
                    bw = self._get_bandwidth_allocation(
                        self.aux_bw_available[aux_idx], resource_level
                    )
                    
                    delay = self._compute_v2v_delay(user, aux, task, cpu_freq, bw)
                    energy = self._compute_offloading_energy(user, task, delay)
                    cost = self._compute_aux_cost(aux, task, resource_level)
                    
                    # Update resource availability
                    self.aux_cpu_available[aux_idx] -= cpu_freq
                    self.aux_bw_available[aux_idx] -= bw
                    aux.current_load += 1
                    
                    offload_location = f"Aux_{aux_idx}"
                else:
                    # Out of V2V range, fallback to local
                    delay = self._compute_local_delay(user, task)
                    energy = self._compute_local_energy(user, task)
                    cost = 0.0
                    offload_location = "local_fallback"
        
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
            "deadline_violated": deadline_violated,
            "offload_location": offload_location,
            "timestamp": self.current_time
        }
        self.next_task_id += 1
        
        # ==================== Compute Reward ====================
        reward = self._compute_reward(delay, energy, cost, deadline_violated, task)
        
        # ==================== Update Time ====================
        self.current_time += self.config.dt
        self.step_count += 1
        
        # ==================== Check Done ====================
        done = self.step_count >= self.config.episode_length
        
        # ==================== Get Next State ====================
        state = self._get_state()
        
        # ==================== Info Dict ====================
        info = {
            "task_delay": delay,
            "task_energy": energy,
            "task_cost": cost,
            "deadline_violated": deadline_violated,
            "offload_location": offload_location,
            "num_tasks_completed": len(self.task_history)
        }
        
        return state, reward, done, info
    
    # ==================== Delay Computation ====================
    
    def _compute_local_delay(self, user: User, task: Task) -> float:
        """
        Tính độ trễ tính toán cục bộ.
        
        t_local = (c_m * s_m) / f_m + t_encrypt
        """
        # Computation time
        computation_time = (task.cycles_per_bit * task.size) / user.cpu_frequency
        
        # Encryption overhead (nếu có)
        encryption_time = 0.0
        if self.config.encryption_enabled:
            encryption_time = self._estimate_encryption_overhead(task.size)
        
        return computation_time + encryption_time
    
    def _compute_mec_delay(self, user: User, mec: MECServer, task: Task, 
                           cpu_freq: float, bandwidth: float) -> float:
        """
        Tính độ trễ offload tới MEC.
        
        t_total = t_upload + t_compute + t_download + t_encrypt
        
        Ref: Eq 9, 17 (paper)
        """
        # Upload time: t_upload = s_m / r_ul
        dist_to_mec = user.get_distance(mec)
        uplink_rate = self.channel_model.get_uplink_rate_mec(dist_to_mec, user.tx_power)
        upload_time = task.size / max(uplink_rate, 1.0)  # bits to bit/s
        
        # Computation time at MEC
        computation_time = (task.cycles_per_bit * task.size) / cpu_freq
        
        # Download time (simplified: assume same as upload)
        download_time = upload_time
        
        # Encryption overhead
        encryption_time = 0.0
        if self.config.encryption_enabled:
            encryption_time = self._estimate_encryption_overhead(task.size)
        
        total_delay = upload_time + computation_time + download_time + encryption_time
        return total_delay
    
    def _compute_v2v_delay(self, user: User, aux: AuxVehicle, task: Task,
                           cpu_freq: float, bandwidth: float) -> float:
        """
        Tính độ trễ offload tới auxiliary vehicle (V2V).
        
        Similar to MEC but with V2V channel model.
        """
        # V2V transmission rate
        dist_to_aux = user.get_distance(aux)
        v2v_rate = self.channel_model.get_uplink_rate_v2v(dist_to_aux, user.tx_power)
        transmission_time = task.size / max(v2v_rate, 1.0)
        
        # Computation time
        computation_time = (task.cycles_per_bit * task.size) / cpu_freq
        
        # Encryption
        encryption_time = 0.0
        if self.config.encryption_enabled:
            encryption_time = self._estimate_encryption_overhead(task.size)
        
        total_delay = transmission_time + computation_time + encryption_time
        return total_delay
    
    def _estimate_encryption_overhead(self, data_size_bits: float) -> float:
        """
        Ước lượng thời gian mã hóa (simulation mode).
        
        t_encrypt = c_en * s_m / f_m
        
        Ref: Eq 4 (paper)
        """
        if not self.config.encryption_enabled or self.security_module is None:
            return 0.0
        
        if self.config.encryption_mode == "simulation":
            # Use formula: c_en (cycles/bit) ~ 800, f_m = user CPU frequency
            cycles_per_bit = 800  # from paillier_params.yaml
            encryption_time = cycles_per_bit * data_size_bits / self.config.user_cpu_frequency
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
    
    def _compute_offloading_energy(self, user: User, task: Task, delay: float) -> float:
        """
        Tính năng lượng tiêu thụ khi offload.
        
        E_offload = P_tx * t_transmission (simplified, không tính computation)
        """
        # Ước lượng: transmission time ~ 70% của total delay
        transmission_time = delay * 0.7
        energy = user.tx_power * transmission_time
        return energy
    
    # ==================== Cost Computation ====================
    
    def _compute_mec_cost(self, mec: MECServer, task: Task, resource_level: int) -> float:
        """
        Tính chi phí sử dụng MEC.
        
        Cost = g_j * (based on urgency, resource level)
        """
        # Urgency (deadline sensitivity)
        urgency = 1.0 / max(task.max_delay, 0.1)
        
        # Resource multiplier
        resource_multiplier = 1.0 + 0.1 * resource_level  # 1.0 to 2.0
        
        cost = mec.base_price * urgency * resource_multiplier
        return cost
    
    def _compute_aux_cost(self, aux: AuxVehicle, task: Task, resource_level: int) -> float:
        """
        Tính chi phí sử dụng auxiliary vehicle.
        
        g_n = b_n * ln(1 + ε_n) * ω_m * g_n0
        
        Ref: Eq 478 (paper)
        """
        urgency = 1.0 / max(task.max_delay, 0.1)
        load_factor = np.log(1 + aux.current_load)
        resource_multiplier = 1.0 + 0.1 * resource_level
        
        cost = aux.credibility * load_factor * urgency * aux.base_price * resource_multiplier
        return cost
    
    # ==================== Reward Computation ====================
    
    def _compute_reward(self, delay: float, energy: float, cost: float,
                       deadline_violated: bool, task: Task) -> float:
        """
        Tính reward dựa trên weighted combination.
        
        reward = -(w_d * delay + w_e * energy + w_c * cost + penalty)
        """
        # Normalized values (to 0-1 range)
        delay_norm = min(delay / task.max_delay, 1.0) if task.max_delay > 0 else 1.0
        energy_norm = min(energy / 10.0, 1.0)  # Assume max 10J per task
        cost_norm = min(cost / 5.0, 1.0)  # Assume max cost 5 units
        
        # Weighted combination
        reward = -(
            self.config.weight_delay * delay_norm +
            self.config.weight_energy * energy_norm +
            self.config.weight_cost * cost_norm
        )
        
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