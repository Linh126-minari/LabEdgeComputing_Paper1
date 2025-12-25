"""
Traffic & Task Generation Module
=================================
Mô phỏng việc sinh task tính toán từ xe người dùng trong mạng Vehicular Edge Computing.

Các thông số lấy từ config files (configs/env/scenario_*.yaml):
- Task size s_m: theo config (default 100kb - 1000kb)
- Cycles per bit c_m: theo config (default 1000 cycles/bit)  
- Max delay t_m: theo config (default 1.0s)
- Task arrival: Poisson process (hoặc deterministic/bursty)

Ref: Paper Section 6 - Simulation Setup
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum
from collections import deque

from src.env.entities import Task, User


class TrafficType(Enum):
    """Các loại traffic pattern"""
    POISSON = "poisson"           # Poisson process (default from paper)
    DETERMINISTIC = "deterministic"  # Fixed interval arrivals
    BURSTY = "bursty"             # Bursty traffic với peak periods


@dataclass
class TrafficConfig:
    """
    Cấu hình cho traffic generation.
    
    Các giá trị mặc định khớp với Paper Section 6.
    Khuyến khích load từ YAML config để dễ thay đổi scenario.
    """
    # === Task Size (bits) ===
    task_size_min: float = 102400      # 100 KB default
    task_size_max: float = 1024000     # 1000 KB default
    
    # === Computation ===
    cycles_per_bit: float = 1000.0
    
    # === QoS ===
    max_delay: float = 1.0
    
    # === Traffic Pattern ===
    traffic_type: TrafficType = TrafficType.POISSON
    arrival_rate: float = 0.5
    
    # === Bursty Parameters ===
    burst_probability: float = 0.1
    burst_multiplier: float = 3.0
    burst_duration_range: Tuple[float, float] = (1.0, 3.0)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrafficConfig":
        """
        Tạo TrafficConfig từ dictionary (YAML/scenario config).
        
        Args:
            config_dict: Dict với các key như task_size_min, arrival_rate, etc.
        
        Returns:
            TrafficConfig instance
        """
        traffic_type_str = config_dict.get("traffic_type", "poisson")
        if isinstance(traffic_type_str, str):
            traffic_type = TrafficType(traffic_type_str.lower())
        else:
            traffic_type = traffic_type_str
        
        burst_range = config_dict.get("burst_duration_range", (1.0, 3.0))
        if isinstance(burst_range, list):
            burst_range = tuple(burst_range)
        
        return cls(
            task_size_min=config_dict.get("task_size_min", 102400),
            task_size_max=config_dict.get("task_size_max", 1024000),
            cycles_per_bit=config_dict.get("cycles_per_bit", 1000.0),
            max_delay=config_dict.get("max_delay", 1.0),
            traffic_type=traffic_type,
            arrival_rate=config_dict.get("arrival_rate", 0.5),
            burst_probability=config_dict.get("burst_probability", 0.1),
            burst_multiplier=config_dict.get("burst_multiplier", 3.0),
            burst_duration_range=burst_range
        )


class TaskGenerator:
    """
    Sinh task tính toán cho user vehicles theo Paper specifications.
    
    Task parameters theo Paper Section 6:
    - s_m ∈ [100, 1000] KB: kích thước task
    - c_m = 1000 cycles/bit: độ phức tạp tính toán
    - t_m = 1.0s: deadline tối đa
    
    Arrival process: Poisson với rate λ
    """
    
    def __init__(self, config: TrafficConfig = None):
        """
        Initialize task generator.
        
        Args:
            config: TrafficConfig object (uses defaults from paper if None)
        """
        self.config = config or TrafficConfig()
        
        # Statistics tracking
        self.total_tasks_generated = 0
        self.task_sizes_generated: List[float] = []
        
        # Bursty mode tracking
        self.in_burst_mode = False
        self.burst_end_time = 0.0
        
        # For deterministic arrivals
        self.last_arrival_time: Dict[int, float] = {}
    
    def generate_task(self) -> Task:
        """
        Sinh một task mới với parameters theo Paper.
        
        Task có:
        - size: s_m ∈ [task_size_min, task_size_max] bits
        - cycles_per_bit: c_m = 1000 cycles/bit
        - max_delay: t_m = 1.0s
        
        Returns:
            Task object
        """
        # Task size s_m (uniform distribution)
        size_bits = np.random.uniform(
            self.config.task_size_min,
            self.config.task_size_max
        )
        
        # Create task with paper parameters
        task = Task(
            size=size_bits,
            cycles_per_bit=self.config.cycles_per_bit,
            max_delay=self.config.max_delay
        )
        
        # Update statistics
        self.total_tasks_generated += 1
        self.task_sizes_generated.append(size_bits)
        
        return task
    
    def should_generate(self, user_id: int, current_time: float, dt: float) -> bool:
        """
        Kiểm tra xem có nên sinh task hay không dựa trên traffic model.
        
        Theo Poisson process: P(arrival in dt) = 1 - e^(-λ*dt) ≈ λ*dt (khi dt nhỏ)
        
        Args:
            user_id: ID của user
            current_time: Thời gian hiện tại
            dt: Time step
        
        Returns:
            True nếu nên sinh task
        """
        if self.config.traffic_type == TrafficType.POISSON:
            return self._poisson_arrival(current_time, dt)
        
        elif self.config.traffic_type == TrafficType.DETERMINISTIC:
            return self._deterministic_arrival(user_id, current_time, dt)
        
        elif self.config.traffic_type == TrafficType.BURSTY:
            return self._bursty_arrival(current_time, dt)
        
        return self._poisson_arrival(current_time, dt)
    
    def _poisson_arrival(self, current_time: float, dt: float) -> bool:
        """
        Poisson process arrival.
        
        P(at least 1 arrival in dt) = 1 - e^(-λ*dt)
        """
        rate = self.config.arrival_rate
        prob = 1 - np.exp(-rate * dt)
        return np.random.random() < prob
    
    def _deterministic_arrival(self, user_id: int, current_time: float, dt: float) -> bool:
        """
        Deterministic (periodic) arrivals.
        
        Task arrives every 1/λ seconds.
        """
        period = 1.0 / self.config.arrival_rate
        last_time = self.last_arrival_time.get(user_id, -period)
        
        if current_time - last_time >= period:
            self.last_arrival_time[user_id] = current_time
            return True
        return False
    
    def _bursty_arrival(self, current_time: float, dt: float) -> bool:
        """
        Bursty traffic: periods of high arrival rate.
        """
        # Update burst mode
        self._update_burst_mode(current_time, dt)
        
        # Get effective arrival rate
        rate = self.config.arrival_rate
        if self.in_burst_mode:
            rate *= self.config.burst_multiplier
        
        prob = 1 - np.exp(-rate * dt)
        return np.random.random() < prob
    
    def _update_burst_mode(self, current_time: float, dt: float):
        """Update burst mode status"""
        if self.in_burst_mode:
            if current_time >= self.burst_end_time:
                self.in_burst_mode = False
        else:
            # Check if entering burst mode
            if np.random.random() < self.config.burst_probability * dt:
                self.in_burst_mode = True
                duration = np.random.uniform(*self.config.burst_duration_range)
                self.burst_end_time = current_time + duration
    
    def get_num_arrivals(self, dt: float) -> int:
        """
        Lấy số lượng task arrivals trong interval dt.
        
        Theo Poisson distribution: N ~ Poisson(λ*dt)
        
        Args:
            dt: Time interval
        
        Returns:
            Number of arrivals
        """
        rate = self.config.arrival_rate
        if self.in_burst_mode:
            rate *= self.config.burst_multiplier
        
        expected = rate * dt
        return np.random.poisson(expected)
    
    def generate_for_user(self, user: User, current_time: float, dt: float) -> Optional[Task]:
        """
        Sinh task cho một user cụ thể.
        
        Cập nhật trực tiếp user.current_task nếu sinh task mới.
        
        Args:
            user: User object
            current_time: Current simulation time
            dt: Time step
        
        Returns:
            Task if generated, None otherwise
        """
        if self.should_generate(user.id, current_time, dt):
            task = self.generate_task()
            user.current_task = task
            return task
        return None
    
    def reset(self):
        """Reset generator state"""
        self.total_tasks_generated = 0
        self.task_sizes_generated.clear()
        self.in_burst_mode = False
        self.burst_end_time = 0.0
        self.last_arrival_time.clear()
    
    def get_statistics(self) -> Dict[str, float]:
        """Get generation statistics"""
        stats = {
            "total_tasks": self.total_tasks_generated,
            "in_burst_mode": self.in_burst_mode
        }
        
        if self.task_sizes_generated:
            sizes_kb = [s / 8 / 1024 for s in self.task_sizes_generated]
            stats["avg_task_size_kb"] = np.mean(sizes_kb)
            stats["min_task_size_kb"] = np.min(sizes_kb)
            stats["max_task_size_kb"] = np.max(sizes_kb)
        
        return stats


class MultiUserTrafficManager:
    """
    Quản lý traffic cho nhiều users trong simulation.
    
    Phối hợp task generation cho tất cả users, quản lý task queues,
    và tracking statistics theo Paper requirements.
    """
    
    def __init__(self, num_users: int, config: TrafficConfig = None):
        """
        Initialize traffic manager.
        
        Args:
            num_users: Số lượng users (M trong paper)
            config: TrafficConfig object
        """
        self.num_users = num_users
        self.config = config or TrafficConfig()
        
        # Task generator
        self.generator = TaskGenerator(self.config)
        
        # Per-user task queues
        self.task_queues: Dict[int, deque] = {i: deque() for i in range(num_users)}
        
        # Statistics
        self.tasks_generated_per_user: Dict[int, int] = {i: 0 for i in range(num_users)}
        self.current_time = 0.0
    
    def step(self, users: List[User], dt: float = 0.1) -> Dict[int, Task]:
        """
        Advance one time step, generate tasks for all users.
        
        Args:
            users: List of User objects
            dt: Time step
        
        Returns:
            Dictionary mapping user_id -> newly generated Task (if any)
        """
        new_tasks: Dict[int, Task] = {}
        
        for user in users:
            task = self.generator.generate_for_user(user, self.current_time, dt)
            if task is not None:
                self.task_queues[user.id].append(task)
                self.tasks_generated_per_user[user.id] += 1
                new_tasks[user.id] = task
        
        self.current_time += dt
        return new_tasks
    
    def get_pending_task(self, user_id: int) -> Optional[Task]:
        """Get and remove next pending task for user"""
        if self.task_queues[user_id]:
            return self.task_queues[user_id].popleft()
        return None
    
    def peek_task(self, user_id: int) -> Optional[Task]:
        """Peek at next task without removing"""
        if self.task_queues[user_id]:
            return self.task_queues[user_id][0]
        return None
    
    def has_pending_task(self, user_id: int) -> bool:
        """Check if user has pending tasks"""
        return len(self.task_queues[user_id]) > 0
    
    def get_queue_length(self, user_id: int) -> int:
        """Get queue length for user"""
        return len(self.task_queues[user_id])
    
    def get_total_pending(self) -> int:
        """Get total pending tasks across all users"""
        return sum(len(q) for q in self.task_queues.values())
    
    def reset(self):
        """Reset manager state"""
        self.generator.reset()
        self.task_queues = {i: deque() for i in range(self.num_users)}
        self.tasks_generated_per_user = {i: 0 for i in range(self.num_users)}
        self.current_time = 0.0
    
    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive statistics"""
        gen_stats = self.generator.get_statistics()
        
        return {
            **gen_stats,
            "tasks_per_user": self.tasks_generated_per_user.copy(),
            "queue_lengths": {i: len(q) for i, q in self.task_queues.items()},
            "total_pending": self.get_total_pending(),
            "simulation_time": self.current_time
        }


def create_traffic_config(config_dict: dict) -> TrafficConfig:
    """
    Helper function: tạo TrafficConfig từ dict.
    
    Args:
        config_dict: Dictionary từ YAML hoặc scenario config
    
    Returns:
        TrafficConfig instance
    """
    return TrafficConfig.from_dict(config_dict)


def compute_task_urgency(task: Task, remaining_time: float = None) -> float:
    """
    Tính độ khẩn cấp của task (ω_m trong paper).
    
    ω_m = 1 / t_m (hoặc dựa trên thời gian còn lại)
    
    Urgency cao hơn → cần xử lý sớm hơn → giá có thể cao hơn
    
    Args:
        task: Task object
        remaining_time: Thời gian còn lại trước deadline (nếu có)
    
    Returns:
        Urgency value (normalized)
    """
    if remaining_time is not None:
        # Dựa trên thời gian còn lại
        if remaining_time <= 0:
            return 10.0  # Maximum urgency
        return min(1.0 / remaining_time, 10.0)
    else:
        # Dựa trên deadline ban đầu
        return 1.0 / max(task.max_delay, 0.1)


def compute_task_computation_load(task: Task) -> float:
    """
    Tính tổng số cycles cần xử lý cho task.
    
    Total cycles = s_m * c_m
    
    Args:
        task: Task object
    
    Returns:
        Total computation cycles
    """
    return task.size * task.cycles_per_bit
