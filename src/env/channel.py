import numpy as np


class ChannelModel:
    """
    Mô hình kênh truyền không dây tính toán tốc độ dữ liệu (Data Rate).
    Sử dụng công thức Shannon: B * log2(1 + SNR).

    T0.5 — Tái tham số hóa với khoảng cách tham chiếu tường minh:

        r = B · log2(1 + p · δ₀ · (d/d₀)^(-θ) / W₀)

    Ref: [A] Eq(9), (17). MỞ RỘNG: [A] không định nghĩa khoảng cách tham chiếu
    d₀ cho δ₀, khiến bộ tham số gốc (δ₀=1e-13, W₀=3e-13, θ=4, p=0.2 W) cho
    SNR ≈ 6.7e-9 (rate ≈ 0) ngay tại d = 100 m — tự mâu thuẫn với Fig.3 của [A]
    (xe trải trên ±500 m). Ta khai báo tường minh d₀ = 1 m và hiệu chỉnh lại δ₀
    bằng scripts/calibrate_channel.py sao cho:
      - SNR ≈ 25 dB tại d = 100 m (rate ≈ 8 Mbps với B = 1 MHz)
      - SNR ≈ 6 dB tại d = 300 m (rate ≈ 2.3 Mbps)
    Hack chia khoảng cách cho 1000 (đổi sang km, nhân SNR lên 10¹² không ghi
    chép) đã bị xóa.
    """

    def __init__(self,
                 bandwidth_mec=1e6,      # B = 1 MHz          — [A] §6
                 bandwidth_v2v=0.8e6,    # B_V2V = 0.8 MHz    — [A] §6
                 noise_power=3e-13,      # W0 = 3e-13 W       — [A] §6
                 channel_gain=4.75e-2,   # δ₀ tại d₀ — MỞ RỘNG (hiệu chỉnh, T0.5)
                 ref_distance=1.0,       # d₀ = 1 m — MỞ RỘNG (T0.5)
                 path_loss_exp=4):       # θ = 4              — [A] §6

        self.B_mec = bandwidth_mec
        self.B_v2v = bandwidth_v2v
        self.W0 = noise_power
        self.delta_0 = channel_gain
        self.d0 = max(float(ref_distance), 1e-6)
        self.theta = path_loss_exp

    def calc_rate(self, dist, tx_power, bandwidth):
        """
        Tính tốc độ truyền tải (bit/s).

        r = B · log2(1 + p · δ₀ · (d/d₀)^(-θ) / W₀)

        Ref: [A] Eq(9), (17) + tái tham số hóa d₀ tường minh (T0.5).
        """
        d = max(float(dist), self.d0)
        path_gain = self.delta_0 * (d / self.d0) ** (-self.theta)
        snr = tx_power * path_gain / self.W0
        return bandwidth * np.log2(1.0 + snr)

    def calc_snr(self, dist, tx_power):
        """SNR (tỉ số tuyến tính) tại khoảng cách dist."""
        d = max(float(dist), self.d0)
        path_gain = self.delta_0 * (d / self.d0) ** (-self.theta)
        return tx_power * path_gain / self.W0

    def get_uplink_rate_mec(self, dist, tx_power):
        """Tính tốc độ upload lên MEC Server. Ref: [A] Eq(9)."""
        return self.calc_rate(dist, tx_power, self.B_mec)

    def get_uplink_rate_v2v(self, dist, tx_power):
        """Tính tốc độ truyền V2V sang xe hỗ trợ. Ref: [A] Eq(17)."""
        return self.calc_rate(dist, tx_power, self.B_v2v)
