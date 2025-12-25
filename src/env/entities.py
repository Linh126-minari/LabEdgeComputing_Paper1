import numpy as np
from dataclasses import dataclass, field
from src.env.mobility import LinearMobility
from src.env.channel import ChannelModel

# --- Base Class ---
@dataclass
class ComputingNode:
    id: int
    position: np.ndarray
    cpu_frequency: float # Hz (cycles/s)
    
    def get_distance(self, other_node):
        return np.linalg.norm(self.position - other_node.position)

# --- Task Definition ---
@dataclass
class Task:
    size: float         # s_m (bits)
    cycles_per_bit: float # c_m (cycles/bit)
    max_delay: float    # t_m (seconds)

# --- User Vehicle ---
@dataclass
class User(ComputingNode):
    """
    Xe người dùng có nhu cầu tính toán.
    Attributes:
        tx_power: Công suất phát p_m 
        velocity: Vận tốc v_m
    """
    velocity: float = 20.0
    tx_power: float = 0.2 # 0.2 W 
    current_task: Task = None
    mobility_model: LinearMobility = field(default_factory=LinearMobility)
    
    def step_mobility(self, dt=1.0):
        """Cập nhật vị trí xe"""
        self.position = self.mobility_model.step(self.position, self.velocity, dt)

    def generate_task(self):
        """Sinh task ngẫu nhiên (Poisson process simulation)"""
        # Task size: 100kb - 1000kb 
        size = np.random.uniform(100, 1000) * 1024 * 8 # đổi ra bits
        # Cycles/bit: 1000 
        cycles = 1000 
        # Max delay: 1.0s (giả định)
        delay = 1.0 
        self.current_task = Task(size, cycles, delay)

# --- MEC Server ---
@dataclass
class MECServer(ComputingNode):
    """
    Máy chủ MEC tại RSU.
    Attributes:
        coverage_radius: Bán kính phủ sóng L 
        base_price: Giá cơ sở g_j0
    """
    coverage_radius: float = 300.0
    base_price: float = 1.0 # unit price
    credibility: float = 0.9 # b_j

    def is_covered(self, user: User):
        """Kiểm tra user có trong vùng phủ sóng không """
        dist = self.get_distance(user)
        return dist <= self.coverage_radius

# --- Auxiliary Vehicle ---
@dataclass
class AuxVehicle(ComputingNode):
    """
    Xe hỗ trợ tính toán (Auxiliary Vehicle).
    Attributes:
        base_price: Giá cơ sở g_n0
        load: Số lượng user đang phục vụ epsilon_n 
    """
    base_price: float = 0.5
    credibility: float = 0.8
    current_load: int = 0 # epsilon_n
    
    def get_dynamic_price(self, task_urgency):
        """
        Tính giá động dựa trên độ khẩn cấp và tải hiện tại.
        Eq: g_n = b_n * ln(1 + epsilon_n) * omega_m * g_n0 
        """
        price = (self.credibility * np.log(1 + self.current_load) * task_urgency * self.base_price)
        return price