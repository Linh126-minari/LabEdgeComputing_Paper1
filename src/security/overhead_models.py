"""
Mô hình overhead bảo mật và lượng hóa security strength.

Hệ HAI LỚP (T0.3 Phương án 1 — quyết định 26/07/2026):
  - Lớp 1: Mã hóa đồng cấu Paillier [A] — bảo vệ nội dung khỏi MEC server tò mò
    (c_en = 800 cycles/bit, [A] Section 6).
  - Lớp 2: Mã khối đối xứng lớp liên kết [B] — bảo vệ khỏi eavesdropper trên kênh
    (N_f = 64 bits, q_en = 90 cycles/bit, [B] Table I).
Tổng chi phí mã hóa: 800 + 90 = 890 cycles/bit.

Nguồn:
  [A] Xue, Yao, Wang — Ad Hoc Networks 161 (2024) 103545
  [B] Xiao et al. — IEEE TWC 22(12), 2023
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class SecurityOverheadConfig:
	# Encryption model — hệ hai lớp: 800 (Paillier [A]) + 90 (block cipher [B])
	overhead_cycles_per_bit: float = 890.0
	user_cpu_frequency: float = 2e9
	security_level: float = 1.0

	# Mô hình attacker của [B] Eq(5)-(9)
	# N_f = 64 bits: maximum block length của mã khối — [B] Table I (T0.3).
	# KHÔNG phải độ dài khóa Paillier (lỗi cũ: 8192).
	frame_length_bits: float = 64.0
	# N_i: block length thực dùng, biến quyết định (T3.1). Lỗi cũ: 2048.
	default_block_length_bits: float = 32.0
	# rho = 1e-4 vehicles/vehicle: TỈ LỆ attacker trên tổng số xe — [B] Table I.
	# KHÔNG phải mật độ không gian 1/m² (lỗi cũ nhỏ hơn 50 lần) — xem T0.2.
	attacker_density: float = 1e-4
	lane_count: float = 8.0        # N_l = 8 — [B] Table I
	max_speed_mps: float = 25.0    # theo [A]; [B] dùng 24 — ghi chú trong bài


class SecurityOverheadModel:
	"""
	Mô hình bảo mật hợp nhất cho bài toán offloading:
	- Overhead mã hóa hai lớp (Paillier [A] + block cipher [B])
	- Security quantification theo mô hình attacker của [B] Eq(5)-(9)

	T0.9: eavesdropping là THỤ ĐỘNG — p_att chỉ ảnh hưởng security strength S
	và secure capacity U, KHÔNG làm giảm rate hay tăng năng lượng (các hàm
	effective_rate/effective_tx_energy cũ với η_r, η_e đã bị xóa vì không có
	trong [A]/[B]; mô hình jamming là mở rộng riêng — xem T3.2).
	"""

	def __init__(self, config: SecurityOverheadConfig):
		self.config = config

	def compute_cycles_per_bit(self, block_length_bits: float | None = None, kappa_N: float = 0.1) -> float:
		"""
		Tính c_en(N_i) = c_en_base * (1 + kappa_N * log2(N_i / N_min)) (Task T3.1).
		Khép kín vòng trade-off security ↔ economic cost.
		"""
		base_c_en = self.config.overhead_cycles_per_bit
		if block_length_bits is None:
			return base_c_en
		n_i = max(float(block_length_bits), 8.0)
		n_min = 8.0
		return base_c_en * (1.0 + kappa_N * np.log2(n_i / n_min))

	def encryption_time(self, data_size_bits: float, block_length_bits: float | None = None) -> float:
		"""t_enc = c_en(N_i) * s / (f_u * sec_level). Ref: [A] Eq(4)."""
		sec = max(self.config.security_level, 1e-6)
		f_u = max(self.config.user_cpu_frequency, 1.0)
		c_en = self.compute_cycles_per_bit(block_length_bits)
		return (c_en * data_size_bits) / (f_u * sec)

	def encryption_energy(self, data_size_bits: float, kappa: float = 1e-27, block_length_bits: float | None = None) -> float:
		"""E_enc = kappa * c_en(N_i) * s * f_u^2 / sec_level. Ref: [A] Eq(5)."""
		sec = max(self.config.security_level, 1e-6)
		f_u = max(self.config.user_cpu_frequency, 1.0)
		c_en = self.compute_cycles_per_bit(block_length_bits)
		return kappa * c_en * data_size_bits * (f_u ** 2) / sec

	def frame_vulnerability(self, block_length_bits: float, frame_length_bits: float | None = None) -> float:
		"""
		Ref: [B] Eq(6): Phi_i = (N_f - N_i) / N_f = 1 - N_i / N_f.
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
		Ref: [B] Eq(8): n^e_i = pi * d_i^2 * rho * gamma, gamma = N_l / (3 * v_max).

		Ghi chú ngữ nghĩa (T0.2):
		- rho là TỈ LỆ attacker trên tổng số xe (vehicles/vehicle), [B] Table I.
		- d_i là BÁN KÍNH VÙNG PHÁT của người gửi ("the coverage range radius of
		  the sending vehicle i"), KHÔNG phải khoảng cách tới node đích.
		- n^e là KỲ VỌNG từ mật độ, không phải số attacker mô phỏng đếm được.
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
		Ref: [B] Eq(7): S_i = (1 - Phi_i)^(n_i^e) = (N_i / N_f)^(n_i^e).
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
		Diễn giải theo [B]: xác suất bị rò rỉ/frame attack = 1 - S_i.
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
		Ref: [B] Eq(9): U_i = (R_i / (8N_i)) * log2(8N_i) * S_i.
		"""
		n_i = max(float(block_length_bits), 1.0)
		r_i = max(float(link_rate_bps), 0.0)
		s_i = float(np.clip(security_strength, 0.0, 1.0))
		return float((r_i / (8.0 * n_i)) * np.log2(8.0 * n_i) * s_i)
