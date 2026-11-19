"""
M/M/n Queueing Model — Erlang-C formula for Multi-core Edge Servers.
Ref: [P3] Xing Chen et al., IEEE TMC 2025, Sec III-C, Eq.(8)-(12).
"""
import math
from typing import Tuple


def erlang_c(n_cores: int, offered_load: float) -> float:
    """
    Xác suất Erlang-C: C(n, a) - Xác suất một tác vụ đến gặp tất cả n lõi đều bận và phải xếp hàng.

    Args:
        n_cores: Số lõi tính toán song song (n >= 1)
        offered_load: a = λ / μ (Erlang, cường độ tải đến)

    Returns:
        Xác suất phải chờ C(n, a) ∈ [0, 1]
    """
    n = max(int(n_cores), 1)
    a = max(float(offered_load), 0.0)
    if a <= 1e-12:
        return 0.0

    rho = a / n
    if rho >= 1.0:
        return 1.0  # Tải vượt quá hoặc bằng dung lượng phục vụ tối đa -> chắc chắn phải chờ

    # Dùng log-space để tính số hạng (a^n / n!) * (1 / (1 - rho)) tránh tràn số với n lớn
    log_numerator = n * math.log(a) - math.lgamma(n + 1) - math.log(1.0 - rho)

    # Tổng chuỗi sum_{k=0}^{n-1} a^k / k!
    summation = 0.0
    for k in range(n):
        summation += math.exp(k * math.log(a) - math.lgamma(k + 1))

    numerator = math.exp(log_numerator)
    return float(numerator / (summation + numerator))


def mm_n_queue_delay(n_cores: int, arrival_rate: float,
                     service_rate_per_core: float) -> float:
    """
    Thời gian xếp hàng trung bình trong hàng đợi M/M/n:
    T_queue = C(n, a) / (n*μ - λ)
    với a = λ / μ, rho = a / n < 1.

    Args:
        n_cores: Số lõi xử lý (n >= 1)
        arrival_rate: Tốc độ tác vụ đến tổng hợp λ (tasks/second)
        service_rate_per_core: Tốc độ xử lý của mỗi lõi μ (tasks/second)

    Returns:
        T_queue (seconds). Trả về 0.0 nếu λ <= 0 hoặc μ <= 0.
        Trả về float('inf') nếu rho >= 1.0 (hệ thống quá tải bất ổn định).
    """
    if arrival_rate <= 1e-12 or service_rate_per_core <= 1e-12:
        return 0.0

    n = max(int(n_cores), 1)
    a = arrival_rate / service_rate_per_core
    rho = a / n

    if rho >= 1.0:
        return float('inf')

    prob_c = erlang_c(n, a)
    denominator = (n * service_rate_per_core) - arrival_rate
    if denominator <= 1e-12:
        return float('inf')

    return float(prob_c / denominator)
