"""
Hiệu chỉnh ϖ₁ — hệ số scale secure information capacity U trong reward (T0.8).

Bối cảnh: code cũ dùng tanh(U). Với N_i = 2048 sai, U ≈ 7690 nên tanh(U) = 1.0
bão hòa hoàn toàn — số hạng này là HẰNG SỐ, không có gradient, không đóng góp gì
vào việc học. [B] Table I cho sẵn đáp án: ϖ₁ ∈ [0,1] × 10⁻⁵, tức [B] cũng phải
scale U xuống. Sau T0.3 (N_i = 32) thang của U đổi đáng kể nên cần hiệu chỉnh lại.

Mục tiêu: ϖ₁·U cùng bậc với các số hạng khác trong reward (tất cả ~[0,1]).

Script này lấy mẫu từ phân phối state THỰC của môi trường (không phải rate tổng
hợp) và in phân vị của U cùng ϖ₁·U cho vài giá trị ϖ₁ ứng viên.

Usage:
    python scripts/calibrate_varpi.py [--samples 2000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig  # noqa: E402

CANDIDATES = [1e-5, 5e-6, 3e-6, 1e-6]  # tất cả nằm trong dải [0,1]x1e-5 của [B]


def collect(samples: int) -> np.ndarray:
    np.random.seed(0)
    env = VehicularEdgeEnvironment(EnvironmentConfig(
        episode_length=1000, task_arrival_rate=5.0))
    env.reset(seed=0)
    caps = []
    while len(caps) < samples:
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        cap = info.get("task_secure_capacity")
        if cap is not None and cap > 0:
            caps.append(cap)
        if terminated or truncated:
            env.reset()
    return np.array(caps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000)
    args = ap.parse_args()

    u = collect(args.samples)
    pct = [5, 25, 50, 75, 95, 99, 100]

    print(f"\nPhân phối U (secure information capacity) trên {len(u)} mẫu:")
    for p in pct:
        label = "max" if p == 100 else f"p{p}"
        print(f"  {label:>4} = {np.percentile(u, p):>14,.1f}")

    print(f"\nϖ₁·U theo từng ứng viên ϖ₁ (mục tiêu: median ~[0.1, 1], p95 < ~1.5):")
    header = "  ϖ₁".ljust(12) + "".join(f"{('max' if p == 100 else f'p{p}'):>10}" for p in pct)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for varpi in CANDIDATES:
        row = f"  {varpi:<10.1e}"
        for p in pct:
            row += f"{varpi * np.percentile(u, p):>10.3f}"
        print(row)

    print("\nGhi chú: mọi ứng viên đều nằm trong dải ϖ₁ ∈ [0,1] × 1e-5 mà [B] Table I")
    print("khai báo, nên chọn giá trị nào cũng trích dẫn được [B]. Sau khi chọn, cập")
    print("nhật configs/params_master.yaml (khóa `varpi_1`) — KHÔNG sửa trực tiếp code.")

    # Kiểm tra giá trị đang dùng
    current = EnvironmentConfig().varpi_1
    med = current * np.median(u)
    p95 = current * np.percentile(u, 95)
    print(f"\nGiá trị đang dùng: ϖ₁ = {current:.1e} -> median = {med:.3f}, p95 = {p95:.3f}")
    print(f"  Cùng bậc với các số hạng khác (median ∈ [1e-3, 1]): "
          f"{'OK' if 1e-3 < med < 1.0 else 'CẦN HIỆU CHỈNH'}")
    print(f"  Có gradient (phương sai != 0): "
          f"{'OK' if u.var() > 0 else 'FAIL — bão hòa như tanh cũ'}")


if __name__ == "__main__":
    main()
