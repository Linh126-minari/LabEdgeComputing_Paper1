"""
Hiệu chỉnh chế độ bảo mật (T1.7) — sweep (ρ, d, N_l, v_max, N_f) và phân tích
vùng suy biến của nghiệm đóng Eq(39) trong [B].

Phát hiện phân tích (động lực cho T3.1 — học N_i thay vì dùng nghiệm đóng):

    N_i* = round(2^{1/((1−n^e)·ln2) − 3})  khi 0 ≤ n^e < 1,   = N_f khi n^e ≥ 1

    - n^e < ~0.52  → N_i* ≤ 1   : SUY BIẾN — "không mã hóa"
    - n^e ∈ [0.52, 0.84]        : vùng CÓ NGHĨA (N_i* ∈ [1, N_f])
    - n^e ≥ ~0.85  → N_i* = N_f : BÃO HÒA — "mã hóa tối đa"

    Với Table I của [B] (ρ=1e-4, N_l=8, v=24, bán kính 50 m): n^e ≈ 0.087 → suy biến.
    Với tầm V2V 200 m: n^e ≈ 1.4 → bão hòa. Vùng hữu ích hẹp bất ngờ.

Bộ tham số vận hành đã chọn (ghi trong params_master.yaml):
    security_range_v2i = 200 m → n^e = 1.34, p_att = 0.61
    security_range_v2v = 150 m → n^e = 0.75, p_att = 0.41 (trong vùng có nghĩa)

Usage:
    python scripts/calibrate_security.py            # bảng phân tích + kiểm tra
    python scripts/calibrate_security.py --plot     # kèm heatmap PNG (cần matplotlib)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.security.overhead_models import SecurityOverheadConfig, SecurityOverheadModel  # noqa: E402

N_F = 64.0       # [B] Table I
N_I = 32.0       # mặc định của dự án
RHO = 1e-4       # [B] Table I
N_L = 8.0        # [B] Table I
V_MAX = 25.0     # [A]; [B] dùng 24


def n_i_star_eq39(n_e: float, n_f: float = N_F) -> float:
    """
    Nghiệm đóng block length của [B] Eq(39).

    N_i* = round(2^{1/((1−n^e)ln2) − 3}) khi 0 ≤ n^e < 1; = N_f khi n^e ≥ 1.
    """
    if n_e >= 1.0:
        return n_f
    exponent = 1.0 / ((1.0 - n_e) * np.log(2.0)) - 3.0
    return float(np.clip(round(2.0 ** exponent), 0.0, n_f))


def gamma(n_l: float = N_L, v_max: float = V_MAX) -> float:
    return n_l / (3.0 * v_max)


def n_e_of(d: float, rho: float = RHO, n_l: float = N_L, v_max: float = V_MAX) -> float:
    return float(np.pi * d * d * rho * gamma(n_l, v_max))


def print_regime_table():
    print("=" * 78)
    print("1) Vùng suy biến của nghiệm đóng [B] Eq(39):  N_i*(n^e), N_f = 64")
    print("=" * 78)
    print(f"{'n^e':>6} | {'N_i*':>6} | nhận xét")
    print("-" * 46)
    for n_e in [0.1, 0.3, 0.52, 0.6, 0.7, 0.8, 0.84, 0.85, 1.0, 1.4]:
        ni = n_i_star_eq39(n_e)
        if ni <= 1:
            note = "SUY BIẾN (không mã hóa)"
        elif ni >= N_F:
            note = "BÃO HÒA (N_i* = N_f)"
        else:
            note = "có nghĩa"
        print(f"{n_e:>6.2f} | {ni:>6.0f} | {note}")
    print("\n=> Vùng thực sự có nghĩa: n^e ∈ [~0.52, ~0.84] — dải hẹp;")
    print("   động lực trực tiếp cho T3.1 (đưa N_i vào action space).\n")


def print_sweep():
    model = SecurityOverheadModel(SecurityOverheadConfig())
    print("=" * 78)
    print("2) Sweep d (bán kính vùng phát người gửi) với ρ=1e-4, N_l=8, v=25:")
    print("=" * 78)
    print(f"{'d (m)':>7} | {'n^e':>7} | {'S(N_i=32)':>10} | {'p_att':>7} | "
          f"{'N_i* Eq39':>9} | chế độ Eq(39)")
    print("-" * 66)
    for d in [50, 100, 109, 150, 189, 200, 250, 300]:
        n_e = n_e_of(d)
        s = model.security_strength(N_I, N_F, n_e)
        p = 1.0 - s
        ni = n_i_star_eq39(n_e)
        regime = ("suy biến" if ni <= 1 else "bão hòa" if ni >= N_F else "có nghĩa")
        print(f"{d:>7} | {n_e:>7.3f} | {s:>10.3f} | {p:>7.3f} | {ni:>9.0f} | {regime}")
    print("\n=> n^e ∈ [0.4, 1.2] tương ứng d ∈ [~109, ~189] m.")


def print_rho_sweep():
    print("\n" + "=" * 78)
    print("3) Sweep ρ tại d = 150 m (V2V) và d = 200 m (V2I):")
    print("=" * 78)
    model = SecurityOverheadModel(SecurityOverheadConfig())
    print(f"{'ρ':>9} | {'n^e(150)':>9} | {'p_att(150)':>10} | {'n^e(200)':>9} | {'p_att(200)':>10}")
    print("-" * 60)
    for rho in [5e-5, 1e-4, 2e-4, 5e-4, 9e-4]:
        n1, n2 = n_e_of(150, rho), n_e_of(200, rho)
        p1 = 1 - model.security_strength(N_I, N_F, n1)
        p2 = 1 - model.security_strength(N_I, N_F, n2)
        print(f"{rho:>9.0e} | {n1:>9.3f} | {p1:>10.3f} | {n2:>9.3f} | {p2:>10.3f}")


def check_operating_point():
    """Kiểm tra bộ tham số vận hành đã chọn (DoD T1.7)."""
    from src.env.core import EnvironmentConfig
    cfg = EnvironmentConfig()
    model = SecurityOverheadModel(SecurityOverheadConfig())

    print("\n" + "=" * 78)
    print("4) Kiểm tra bộ tham số vận hành (params_master):")
    print("=" * 78)
    p_atts = []
    for name, d in [("V2I (security_range_v2i)", cfg.security_range_v2i),
                    ("V2V (security_range_v2v)", cfg.security_range_v2v)]:
        n_e = n_e_of(d, cfg.attacker_density, cfg.lane_count, cfg.road_max_speed_mps)
        p = 1 - model.security_strength(cfg.encryption_block_length_bits,
                                        cfg.frame_length_bits, n_e)
        p_atts.append(p)
        in_range = 0.4 <= n_e <= 1.4
        print(f"  {name}: d = {d:.0f} m → n^e = {n_e:.3f}, p_att = {p:.3f} "
              f"[{'OK' if in_range else 'NGOÀI vùng khuyến nghị'}]")

    mean_p = float(np.mean(p_atts))
    var_ok = len(set(np.round(p_atts, 6))) > 1
    print(f"\n  p_att trung bình (2 nhánh) = {mean_p:.3f} ∈ [0.1, 0.6]: "
          f"{'OK' if 0.1 <= mean_p <= 0.6 else 'FAIL'}")
    print(f"  p_att có phương sai khác 0 (V2I ≠ V2V): {'OK' if var_ok else 'FAIL'}")


def plot_heatmaps(outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = SecurityOverheadModel(SecurityOverheadConfig())
    rhos = np.linspace(5e-5, 9e-4, 60)
    ds = np.linspace(30, 300, 60)
    R, D = np.meshgrid(rhos, ds)
    NE = np.pi * D ** 2 * R * gamma()
    P = 1.0 - np.clip((N_I / N_F) ** NE, 0, 1)
    NI = np.vectorize(n_i_star_eq39)(NE)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, Z, title in [
        (axes[0], NE, r"$n^e = \pi d^2 \rho \gamma$"),
        (axes[1], P, r"$p_{att} = 1 - (N_i/N_f)^{n^e}$  ($N_i$=32)"),
        (axes[2], NI, r"$N_i^*$ theo [B] Eq(39)"),
    ]:
        im = ax.pcolormesh(R, D, Z, shading="auto")
        ax.set_xlabel(r"$\rho$ (attackers/vehicle)")
        ax.set_ylabel("d (m)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    # vùng có nghĩa của Eq(39): 0.52 ≤ n^e ≤ 0.84
    for ax in axes:
        ax.contour(R, D, NE, levels=[0.52, 0.84], colors="red", linewidths=1.2)
    fig.tight_layout()
    out = outdir / "security_calibration_heatmap.png"
    fig.savefig(out, dpi=150)
    print(f"\nĐã lưu heatmap: {out}")

    # Hình N_i* vs n^e với vùng suy biến/bão hòa tô màu (T1.7 bước 4)
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    ne_axis = np.linspace(0.0, 1.4, 400)
    ax.plot(ne_axis, [n_i_star_eq39(x) for x in ne_axis], lw=2)
    ax.axvspan(0.0, 0.52, alpha=0.15, color="red", label="suy biến ($N_i^* \\leq 1$)")
    ax.axvspan(0.85, 1.4, alpha=0.15, color="orange", label="bão hòa ($N_i^* = N_f$)")
    ax.set_xlabel(r"$n^e$")
    ax.set_ylabel(r"$N_i^*$ (bits)")
    ax.set_title("Nghiệm đóng [B] Eq(39): vùng có nghĩa chỉ là $n^e \\in [0.52, 0.84]$")
    ax.legend()
    fig2.tight_layout()
    out2 = outdir / "eq39_degenerate_regions.png"
    fig2.savefig(out2, dpi=150)
    print(f"Đã lưu hình vùng suy biến: {out2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="Xuất heatmap PNG (cần matplotlib)")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "outputs"))
    args = ap.parse_args()

    print_regime_table()
    print_sweep()
    print_rho_sweep()
    check_operating_point()

    if args.plot:
        plot_heatmaps(Path(args.outdir))


if __name__ == "__main__":
    main()
