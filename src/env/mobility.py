"""
Mobility Models for Vehicular Networks
=======================================
Implements various mobility models for vehicle movement simulation.
Ref: Paper section on vehicular dynamics and trajectory generation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Any, Optional


class LinearMobility:
    """
    Linear mobility model with constant velocity.
    
    Xe di chuyển theo một hướng không đổi với vận tốc không đổi.
    Sử dụng cho mô phỏng đơn giản: x_{t+1} = x_t + v * dt
    """
    
    def __init__(self, change_direction_prob: float = 0.01):
        """
        Initialize linear mobility model.
        
        Args:
            change_direction_prob: Probability to change direction at each step
        """
        self.change_direction_prob = change_direction_prob
        self.current_direction = np.random.uniform(0, 2 * np.pi)  # Random initial angle
    
    def step(self, position: np.ndarray, velocity: float, dt: float = 1.0) -> np.ndarray:
        """
        Compute next position after time dt.
        
        Eq: position_{t+1} = position_t + velocity * direction * dt
        
        Args:
            position: Current position [x, y]
            velocity: Speed (m/s)
            dt: Time step (seconds)
        
        Returns:
            New position [x, y]
        """
        # Randomly change direction
        if np.random.random() < self.change_direction_prob:
            self.current_direction = np.random.uniform(0, 2 * np.pi)
        
        # Compute displacement
        displacement = np.array([
            velocity * np.cos(self.current_direction) * dt,
            velocity * np.sin(self.current_direction) * dt
        ])
        
        # Update position
        new_position = position + displacement
        
        return new_position
    
    def generate_initial_state(self) -> Tuple[np.ndarray, float]:
        """
        Sinh trạng thái ban đầu ngẫu nhiên cho một phương tiện.
        
        Returns:
            (position [x, y], velocity)
        """
        x = np.random.uniform(-500, 500)
        y = np.random.uniform(-20, 20)
        v = np.random.uniform(15, 25)  # m/s
        return np.array([x, y]), v


class RandomWaypointMobility:
    """
    Random Waypoint Mobility Model.
    
    Xe di chuyển đến một điểm ngẫu nhiên, sau đó chọn điểm mới.
    Phù hợp hơn với hành vi lái xe thực tế.
    """
    
    def __init__(self, area_bounds: Tuple[Tuple[float, float], Tuple[float, float]], 
                 pause_time: float = 0.0):
        """
        Initialize random waypoint model.
        
        Args:
            area_bounds: ((x_min, x_max), (y_min, y_max))
            pause_time: Time to pause at waypoint (seconds)
        """
        self.area_bounds = area_bounds
        self.pause_time = pause_time
        self.target_waypoint = None
        self.pause_counter = 0.0
        self._generate_new_waypoint()
    
    def _generate_new_waypoint(self):
        """Generate new random waypoint"""
        x_bounds, y_bounds = self.area_bounds
        self.target_waypoint = np.array([
            np.random.uniform(x_bounds[0], x_bounds[1]),
            np.random.uniform(y_bounds[0], y_bounds[1])
        ])
    
    def step(self, position: np.ndarray, velocity: float, dt: float = 1.0) -> np.ndarray:
        """
        Compute next position after time dt.
        
        Args:
            position: Current position [x, y]
            velocity: Speed (m/s)
            dt: Time step (seconds)
        
        Returns:
            New position [x, y]
        """
        # Check if paused
        if self.pause_counter > 0:
            self.pause_counter -= dt
            return position
        
        # Distance to target
        direction = self.target_waypoint - position
        distance = np.linalg.norm(direction)
        
        # If reached target, pick new one
        if distance < velocity * dt:
            self.pause_counter = self.pause_time
            self._generate_new_waypoint()
            return position
        
        # Move towards target
        direction_normalized = direction / (distance + 1e-6)  # Avoid division by zero
        displacement = direction_normalized * velocity * dt
        new_position = position + displacement
        
        return new_position


class GaussMarkovMobility:
    """
    Gauss-Markov Mobility Model.
    
    Mô hình vận tốc và hướng di chuyển là Markov process.
    Phù hợp với lái xe thực tế hơn.
    
    Ref: Mobility model literature, commonly used in vehicular networks.
    """
    
    def __init__(self, velocity_mean: float = 20.0, 
                 velocity_std: float = 5.0,
                 alpha: float = 0.9):
        """
        Initialize Gauss-Markov model.
        
        Args:
            velocity_mean: Mean velocity (m/s)
            velocity_std: Standard deviation of velocity
            alpha: Correlation parameter (0-1, higher = more correlated with previous)
        """
        self.velocity_mean = velocity_mean
        self.velocity_std = velocity_std
        self.alpha = alpha
        
        # Initial velocity and direction
        self.current_velocity = velocity_mean
        self.current_direction = np.random.uniform(0, 2 * np.pi)
    
    def step(self, position: np.ndarray, velocity: float, dt: float = 1.0) -> np.ndarray:
        """
        Compute next position after time dt.
        
        Uses Gauss-Markov process for smooth velocity transitions:
        v_new = alpha * v_old + (1 - alpha) * v_mean + noise
        
        Args:
            position: Current position [x, y]
            velocity: Desired speed (m/s) - can be overridden by model
            dt: Time step (seconds)
        
        Returns:
            New position [x, y]
        """
        # Update velocity using Gauss-Markov process
        noise = np.random.normal(0, self.velocity_std)
        self.current_velocity = (
            self.alpha * self.current_velocity +
            (1 - self.alpha) * self.velocity_mean +
            noise
        )
        self.current_velocity = max(0.1, self.current_velocity)  # Ensure positive
        
        # Update direction similarly
        direction_change = np.random.normal(0, 0.1)
        self.current_direction += direction_change
        self.current_direction = self.current_direction % (2 * np.pi)
        
        # Compute displacement
        displacement = np.array([
            self.current_velocity * np.cos(self.current_direction) * dt,
            self.current_velocity * np.sin(self.current_direction) * dt
        ])
        
        new_position = position + displacement
        
        return new_position


class SumoMobility:
    """
    Placeholder cho việc tích hợp SUMO sau này (theo mobility_sumo.yaml).
    
    SUMO (Simulation of Urban Mobility) có thể cung cấp dữ liệu di chuyển thực tế.
    """
    
    def __init__(self, net_file: str = None, route_file: str = None):
        """
        Initialize SUMO mobility model.
        
        Args:
            net_file: Path to SUMO network file (.net.xml)
            route_file: Path to SUMO route file (.rou.xml)
        """
        self.net_file = net_file
        self.route_file = route_file
        # TODO: Integrate SUMO library for real trajectory data
    
    def step(self, position: np.ndarray, velocity: float, dt: float = 1.0) -> np.ndarray:
        """
        Placeholder for SUMO step.
        """
        # TODO: Implement SUMO integration
        return position


class TraceMobility:
    """
    Mobility từ quỹ đạo GPS thật (Phase 4 - T4.2), dữ liệu Rio bus đã tiền xử lý
    (`scripts/preprocess_riobus.py`, xem BAO_CAO_PHASE4_RIOBUS.md).

    Vị trí được nội suy theo THỜI GIAN mô phỏng (cột `t_rel`), nên trace lưu ở bước
    1 s vẫn dùng được với dt = 0.1 s. Cột tùy chọn `active` (0/1): slot đang không có
    xe buýt thật (xe đã rời vùng, chưa có xe thay) — `is_active` trả False.
    """

    def __init__(self, trace_df: Any, vehicle_id: str, loop: bool = True):
        self.vehicle_id = vehicle_id
        self.loop = loop

        if trace_df is not None and not trace_df.empty:
            v_df = trace_df[trace_df["vehicle_id"] == vehicle_id]
            if v_df.empty:
                v_df = trace_df
            v_df = v_df.sort_values("t_rel").reset_index(drop=True)
            self.times = v_df["t_rel"].to_numpy(dtype=float) - float(v_df["t_rel"].iloc[0])
            self.positions = v_df[["x_m", "y_m"]].to_numpy(dtype=float)
            self.velocities = v_df["v_mps"].to_numpy(dtype=float)
            self.active = (v_df["active"].to_numpy(dtype=int) == 1) if "active" in v_df else np.ones(len(v_df), bool)
        else:
            self.times = np.array([0.0])
            self.positions = np.array([[0.0, 0.0]], dtype=float)
            self.velocities = np.array([15.0], dtype=float)
            self.active = np.array([True])

        self.num_points = len(self.positions)
        self.duration = float(self.times[-1])
        self.t = 0.0
        self._last_pos = self.positions[0].copy()
        self._direction = 0.0

    def _idx(self) -> int:
        return int(np.clip(np.searchsorted(self.times, self.t, side="right") - 1, 0, self.num_points - 1))

    def _position_at(self, t: float) -> np.ndarray:
        return np.array([np.interp(t, self.times, self.positions[:, 0]),
                         np.interp(t, self.times, self.positions[:, 1])])

    def step(self, position: np.ndarray, velocity: float, dt: float = 0.1) -> np.ndarray:
        self.t += dt
        if self.t > self.duration:
            self.t = (self.t % self.duration) if (self.loop and self.duration > 0) else self.duration
        pos = self._position_at(self.t)
        delta = pos - self._last_pos
        if np.hypot(*delta) > 1e-9:
            self._direction = float(np.arctan2(delta[1], delta[0]))
        self._last_pos = pos
        return pos.copy()

    def current_velocity(self) -> float:
        return float(np.interp(self.t, self.times, self.velocities))

    @property
    def is_active(self) -> bool:
        return bool(self.active[self._idx()])

    @property
    def current_direction(self) -> float:
        return self._direction

    def generate_initial_state(self) -> Tuple[np.ndarray, float]:
        self.t = 0.0
        self._last_pos = self.positions[0].copy()
        if self.num_points > 1:
            d = self.positions[1] - self.positions[0]
            self._direction = float(np.arctan2(d[1], d[0])) if np.hypot(*d) > 1e-9 else 0.0
        return self.positions[0].copy(), float(self.velocities[0])
