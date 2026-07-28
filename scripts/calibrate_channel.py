"""
Hiệu chỉnh δ₀ cho mô hình kênh truyền (T0.5).

Bối cảnh: bộ tham số của [A] (δ₀=1e-13, W₀=3e-13, θ=4, p=0.2 W) tự nó không
nhất quán khi d tính bằng mét — SNR ≈ 6.7e-9 tại d=100 m → rate ≈ 0, trong khi
Fig.3 của [A] vẽ xe trải trên ±500 m. [A] thiếu định nghĩa khoảng cách tham
chiếu cho δ₀. Ta khai báo tường minh d₀ = 1 m và hiệu chỉnh δ₀.

Mục tiêu hiệu chỉnh:
  - SNR trong dải thực tế: ~25 dB tại d=100 m, ~5-10 dB tại d=300 m
  - rate tại d=100 m với B=1 MHz ở bậc vài Mbps (task 1 Mbit, deadline 1 s
    → t_up chiếm phần đáng kể nhưng không phải 100% deadline)

Usage:
    python scripts/calibrate_channel.py            # bảng cho δ₀ đã chọn
    python scripts/calibrate_channel.py --sweep    # sweep nhiều δ₀
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.channel import ChannelModel  # noqa: E402

TX_POWER = 0.2      # p_m = 0.2 W, [A] §6
BANDWIDTH = 1e6     # B = 1 MHz, [A] §6
DISTANCES = [10, 50, 100, 200, 300, 500]


def print_table(delta_0: float):
    ch = ChannelModel(channel_gain=delta_0, ref_distance=1.0)
    print(f"\ndelta_0 = {delta_0:g} (d0 = 1 m, theta = 4, W0 = 3e-13 W, p = {TX_POWER} W)")
    print(f"{'d (m)':>8} | {'SNR (dB)':>10} | {'rate (Mbps, B=1MHz)':>20}")
    print("-" * 46)
    for d in DISTANCES:
        snr = ch.calc_snr(d, TX_POWER)
        snr_db = 10 * np.log10(snr) if snr > 0 else float("-inf")
        rate = ch.calc_rate(d, TX_POWER, BANDWIDTH) / 1e6
        print(f"{d:>8} | {snr_db:>10.2f} | {rate:>20.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="Sweep nhiều giá trị delta_0")
    ap.add_argument("--delta0", type=float, default=4.75e-2,
                    help="Giá trị delta_0 đã chọn/khóa (mặc định 4.75e-2)")
    args = ap.parse_args()

    if args.sweep:
        for delta_0 in [1e-13, 1e-3, 1e-2, 4.75e-2, 1e-1, 1.0]:
            print_table(delta_0)
        print("\n=> Chọn delta_0 sao cho SNR(100m) ~ 25 dB, SNR(300m) ~ 5-10 dB, "
              "rate(100m) ~ vài Mbps. Giá trị khóa: 4.75e-2.")
    else:
        print_table(args.delta0)
        # Kiểm tra mục tiêu hiệu chỉnh
        ch = ChannelModel(channel_gain=args.delta0, ref_distance=1.0)
        rate_100 = ch.calc_rate(100, TX_POWER, BANDWIDTH)
        snr_300_db = 10 * np.log10(ch.calc_snr(300, TX_POWER))
        ok_rate = 1e6 <= rate_100 <= 20e6
        ok_snr = 3.0 <= snr_300_db <= 12.0
        print(f"\nKiểm tra: rate(100m) = {rate_100/1e6:.2f} Mbps ∈ [1, 20] Mbps: "
              f"{'OK' if ok_rate else 'FAIL'}")
        print(f"          SNR(300m) = {snr_300_db:.2f} dB ~ [5, 10] dB: "
              f"{'OK' if ok_snr else 'FAIL'}")


if __name__ == "__main__":
    main()
