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

    def velocity_vector(self) -> np.ndarray:
        """Vector vận tốc [vx, vy]. Node tĩnh trả về [0, 0]."""
        return np.zeros(2)

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
    tx_power: float = 0.2 # p_m = 0.2 W, Ref: [A] §6
    current_task: Task = None
    mobility_model: LinearMobility = field(default_factory=LinearMobility)

    def step_mobility(self, dt=1.0):
        """Cập nhật vị trí xe"""
        self.position = self.mobility_model.step(self.position, self.velocity, dt)

    def velocity_vector(self) -> np.ndarray:
        """Vector vận tốc theo hướng hiện tại của mobility model (T1.1)."""
        theta = self.mobility_model.current_direction
        return self.velocity * np.array([np.cos(theta), np.sin(theta)])

    # T0.4: generate_task() đã bị XÓA. Chỉ TaskGenerator (src/env/traffic.py)
    # được phép sinh task — một đường sinh task duy nhất, đơn vị bits theo [A].

# --- MEC Server ---
@dataclass
class MECServer(ComputingNode):
    """
    Máy chủ MEC tại RSU.
    Attributes:
        coverage_radius: Bán kính phủ sóng L
        curb_distance: Khoảng cách RSU tới lề đường (z trong [A] Eq(6)) — T1.1
        base_price: Giá cơ sở g_j0
    """
    coverage_radius: float = 300.0
    curb_distance: float = 20.0  # z, [A] Eq(6) — giá trị MỞ RỘNG ([A] không cho số)
    base_price: float = 1.0 # unit price
    credibility: float = 0.9 # b_j

    def is_covered(self, user: User):
        """Kiểm tra user có trong vùng phủ sóng không """
        dist = self.get_distance(user)
        return dist <= self.coverage_radius

    def road_segment_length(self) -> float:
        """
        [A] Eq(6): x = 2*sqrt(y^2 - z^2), với y = coverage_radius, z = curb_distance.

        Chiều dài đoạn đường nằm trong vùng phủ sóng của RSU (T1.1).
        """
        return 2.0 * np.sqrt(max(self.coverage_radius ** 2 - self.curb_distance ** 2, 0.0))

# --- Auxiliary Vehicle ---
@dataclass
class AuxVehicle(ComputingNode):
    """
    Xe hỗ trợ tính toán (Auxiliary Vehicle).
    Attributes:
        base_price: Giá cơ sở g_n0
        current_load: Số user đang phục vụ epsilon_n.
            [A] §3.4: "the service of one auxiliary vehicle is limited to one
            user" — ràng buộc current_load <= 1 được THỰC THI trong env (T1.5).
        velocity/mobility_model: AUX di chuyển bền vững theo LinearMobility
            (T1.1 — cần vector vận tốc ổn định để tính thời gian tồn tại
            liên kết V2V; bản cũ đổi hướng ngẫu nhiên mỗi step là vô nghĩa).
    """
    base_price: float = 0.5
    credibility: float = 0.8
    current_load: int = 0 # epsilon_n
    velocity: float = 15.0
    mobility_model: LinearMobility = field(default_factory=LinearMobility)

    def step_mobility(self, dt=1.0):
        self.position = self.mobility_model.step(self.position, self.velocity, dt)

    def velocity_vector(self) -> np.ndarray:
        theta = self.mobility_model.current_direction
        return self.velocity * np.array([np.cos(theta), np.sin(theta)])

    def get_dynamic_price(self, task_urgency):
        """
        Tính giá động dựa trên độ khẩn cấp và tải hiện tại.

        g_n = b_n * ln(1 + eps_n) * omega_m * g_n0     Ref: [A] Section 3.5

        MỞ RỘNG so với [A] (T0.7): eps_n bao gồm CẢ user đang phát yêu cầu
        (eps_n = current_load + 1 >= 1), nếu không một AUX rảnh sẽ có giá 0
        và luôn được chọn — một suy biến của Algorithm 1 trong [A].
        """
        effective_load = self.current_load + 1
        price = (self.credibility * np.log(1 + effective_load) * task_urgency * self.base_price)
        return price


# --- Attack Vehicle ---
@dataclass
class AttackVehicle(ComputingNode):
    """
    Xe tấn công (nghe lén).

    LƯU Ý (T0.9): thực thể này CHỈ dùng để minh họa/hình vẽ. Mô hình bảo mật
    dùng kỳ vọng mật độ [B] Eq(8), không đếm thực thể. Các tham số jamming
    (tx_power, jamming_radius) đã bị xóa — eavesdropping thụ động không gây
    suy giảm kênh; mô hình jamming là mở rộng riêng (T3.2).
    """
    mobility_model: LinearMobility = field(default_factory=LinearMobility)

    def step_mobility(self, dt=1.0):
        """Cập nhật vị trí xe tấn công"""
        # Giả định vận tốc cố định để tránh thêm nhiều biến trạng thái.
        velocity = 18.0
        self.position = self.mobility_model.step(self.position, velocity, dt)