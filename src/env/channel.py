import numpy as np

class ChannelModel:
    """
    Mô hình kênh truyền không dây tính toán tốc độ dữ liệu (Data Rate).
    Sử dụng công thức Shannon: B * log2(1 + SNR).
    """
    def __init__(self, 
                 bandwidth_mec=1e6,      # 1 MHz 
                 bandwidth_v2v=0.8e6,    # 0.8 MHz 
                 noise_power=3e-13,      # W0 = 3e-13 W 
                 channel_gain=1e-13,     # delta_0 = 1e-13 
                 path_loss_exp=4):       # theta = 4 
        
        self.B_mec = bandwidth_mec
        self.B_v2v = bandwidth_v2v
        self.W0 = noise_power
        self.delta_0 = channel_gain
        self.theta = path_loss_exp

    def calc_rate(self, dist, tx_power, bandwidth):
        """
        Tính tốc độ truyền tải (bit/s).
        Công thức Eq 9 và Eq 17.
        
        r = B * log2(1 + (p * d^(-theta) * delta_0) / W0)
        """
        # Tránh lỗi chia cho 0 nếu khoảng cách quá nhỏ
        d = max(dist, 1.0) 
        d_km = d / 1000.0  # Chuyển đổi sang km nếu cần thiết
        # Tính công suất nhận được (Received Power)
        rx_power = tx_power * (d_km ** -self.theta) * self.delta_0
        
        # Tỷ lệ Tín hiệu trên Nhiễu (SNR)
        snr = rx_power / self.W0
        
        # Công thức Shannon
        rate = bandwidth * np.log2(1 + snr)
        return rate

    def get_uplink_rate_mec(self, dist, tx_power):
        """Tính tốc độ upload lên MEC Server"""
        return self.calc_rate(dist, tx_power, self.B_mec)

    def get_uplink_rate_v2v(self, dist, tx_power):
        """Tính tốc độ truyền V2V sang xe hỗ trợ"""
        return self.calc_rate(dist, tx_power, self.B_v2v)