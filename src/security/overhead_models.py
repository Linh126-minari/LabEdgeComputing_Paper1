from dataclasses import dataclass
import numpy as np


@dataclass
class SecurityOverheadConfig:
	# Encryption model
	overhead_cycles_per_bit: float = 800.0
	user_cpu_frequency: float = 2e9
	security_level: float = 1.0

	# Attacker penalty on communication/energy
	attack_rate_penalty: float = 0.35
	attack_energy_penalty: float = 0.45

	# Paper attacker model (Eq. 5-9)
	frame_length_bits: float = 8192.0
	default_block_length_bits: float = 2048.0
	attacker_density: float = 2e-6
	lane_count: float = 3.0
	max_speed_mps: float = 25.0


class SecurityOverheadModel:
	"""
	Mô hình bảo mật hợp nhất cho bài toán offloading:
	- Overhead mã hóa
	- Security quantification theo mô hình attacker của bài báo (Eq. 5-9)
	- Hệ số suy giảm tốc độ/khuếch đại năng lượng do tấn công
	"""

	def __init__(self, config: SecurityOverheadConfig):
		self.config = config

	def encryption_time(self, data_size_bits: float) -> float:
		"""t_enc = c_en * s / (f_u * sec_level)"""
		sec = max(self.config.security_level, 1e-6)
		f_u = max(self.config.user_cpu_frequency, 1.0)
		return (self.config.overhead_cycles_per_bit * data_size_bits) / (f_u * sec)

	def encryption_energy(self, data_size_bits: float, kappa: float = 1e-27) -> float:
		"""E_enc = kappa * c_en * s * f_u^2 / sec_level"""
		sec = max(self.config.security_level, 1e-6)
		f_u = max(self.config.user_cpu_frequency, 1.0)
		return kappa * self.config.overhead_cycles_per_bit * data_size_bits * (f_u ** 2) / sec

	def frame_vulnerability(self, block_length_bits: float, frame_length_bits: float | None = None) -> float:
		"""
		Eq. (6): Phi_i = (N_f - N_i) / N_f = 1 - N_i / N_f.
		Được chặn trong [0, 1].
		"""
		n_f = max(float(frame_length_bits or self.config.frame_length_bits), 1e-6)
		n_i = float(np.clip(block_length_bits, 0.0, n_f))
		return float(np.clip((n_f - n_i) / n_f, 0.0, 1.0))

	def expected_attackers(
		self,
		coverage_radius_m: float,
		attacker_density: float | None = None,
		lane_count: float | None = None,
		max_speed_mps: float | None = None,
	) -> float:
		"""
		Eq. (8): n_i^e = pi * d_i^2 * rho * gamma,
		gamma = N_l / (3 * v_max).
		"""
		d_i = max(float(coverage_radius_m), 0.0)
		rho = max(float(attacker_density if attacker_density is not None else self.config.attacker_density), 0.0)
		n_l = max(float(lane_count if lane_count is not None else self.config.lane_count), 1e-6)
		v_max = max(float(max_speed_mps if max_speed_mps is not None else self.config.max_speed_mps), 1e-6)

		gamma = n_l / (3.0 * v_max)
		return float(np.pi * (d_i ** 2) * rho * gamma)

	def security_strength(
		self,
		block_length_bits: float,
		frame_length_bits: float,
		n_attackers: float,
	) -> float:
		"""
		Eq. (7): S_i = (1 - Phi_i)^(n_i^e) = (N_i / N_f)^(n_i^e).
		"""
		n_f = max(float(frame_length_bits), 1e-6)
		n_i = float(np.clip(block_length_bits, 1.0, n_f))
		n_e = max(float(n_attackers), 0.0)

		base = np.clip(n_i / n_f, 1e-9, 1.0)
		return float(np.clip(base ** n_e, 0.0, 1.0))

	def attack_success_probability(
		self,
		block_length_bits: float,
		frame_length_bits: float,
		n_attackers: float,
	) -> float:
		"""
		Diễn giải theo paper: xác suất bị rò rỉ/frame attack = 1 - S_i.
		"""
		s_i = self.security_strength(block_length_bits, frame_length_bits, n_attackers)
		return float(np.clip(1.0 - s_i, 0.0, 1.0))

	def secure_information_capacity(
		self,
		link_rate_bps: float,
		block_length_bits: float,
		security_strength: float,
	) -> float:
		"""
		Eq. (9): U_i = (R_i / (8N_i)) * log2(8N_i) * S_i.
		"""
		n_i = max(float(block_length_bits), 1.0)
		r_i = max(float(link_rate_bps), 0.0)
		s_i = float(np.clip(security_strength, 0.0, 1.0))
		return float((r_i / (8.0 * n_i)) * np.log2(8.0 * n_i) * s_i)

	def effective_rate(self, nominal_rate: float, attack_probability: float) -> float:
		"""r_eff = r_nom * (1 - eta_r * p_att)."""
		factor = 1.0 - self.config.attack_rate_penalty * np.clip(attack_probability, 0.0, 1.0)
		return max(nominal_rate * factor, 1.0)

	def effective_tx_energy(self, nominal_energy: float, attack_probability: float) -> float:
		"""E_tx_eff = E_tx * (1 + eta_e * p_att)."""
		factor = 1.0 + self.config.attack_energy_penalty * np.clip(attack_probability, 0.0, 1.0)
		return nominal_energy * factor
