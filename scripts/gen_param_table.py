"""
Sinh bảng tham số LaTeX từ configs/params_master.yaml (T0.1).

Từ giờ KHÔNG BAO GIỜ gõ tay số vào .tex — bảng tham số được chèn bằng:
    \\input{params_table.tex}

Ba lỗi đã sửa so với bản đầu:
  1. PyYAML chuẩn parse "1.0e9" thành CHUỖI (regex float của YAML 1.1 đòi dấu ở
     số mũ). Ta dùng OmegaConf.load — cùng loader mà Hydra dùng — nên script
     thấy đúng giá trị mà môi trường huấn luyện thấy; kèm coerce dự phòng.
  2. Ký tự `^`, `_`, `%`, `&`... trong cột unit/source làm pdflatex lỗi. Ta
     escape, và render các đơn vị cần số mũ qua UNITS_TEX (math mode).
  3. Dấu tiếng Việt trong unit/source làm pdflatex lỗi (inputenc utf8 không tự
     xử lý). Ta VALIDATE: unit/source phải ASCII, nếu không thì báo lỗi rõ ràng
     và chỉ chỗ sửa (đưa tiếng Việt sang field `note`, không xuất ra .tex).

Usage:
    python scripts/gen_param_table.py [--out notebooks/params_table.tex] [--check]
"""
import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]

# Ánh xạ tên tham số -> ký hiệu LaTeX (mở rộng dần khi cần)
SYMBOLS = {
    "num_users": r"$U$",
    "num_mec_servers": r"$J$",
    "num_auxiliary_vehicles": r"$N$",
    "velocity_min": r"$v_{\min}$",
    "velocity_max": r"$v_{\max}$",
    "lane_count": r"$N_l$",
    "road_max_speed": r"$v_{\max}^{\mathrm{road}}$",
    "task_size_min": r"$s_m^{\min}$",
    "task_size_max": r"$s_m^{\max}$",
    "task_cycles_per_bit": r"$c_m$",
    "task_deadline_app": r"$\tau_{\mathrm{app}}$",
    "task_arrival_rate": r"$\lambda$",
    "mec_cpu_frequency": r"$f_j$",
    "aux_cpu_frequency": r"$f_n$",
    "user_cpu_frequency": r"$f_u$",
    "kappa_effective_capacitance": r"$\kappa$",
    "bandwidth_mec": r"$B$",
    "bandwidth_v2v": r"$B_{\mathrm{V2V}}$",
    "noise_power": r"$W_0$",
    "tx_power": r"$p_m$",
    "idle_power": r"$p_w$",
    "path_loss_exponent": r"$\theta$",
    "channel_ref_distance": r"$d_0$",
    "channel_ref_gain": r"$\delta_0$",
    "mec_coverage_radius": r"$L$",
    "v2v_range": r"$d_{\mathrm{V2V}}$",
    "homomorphic_key_length": r"$K_{\mathrm{HE}}$",
    "homomorphic_cycles_per_bit": r"$c^{\mathrm{en}}_m$",
    "block_cipher_frame_length": r"$N_f$",
    "block_cipher_block_length": r"$N_i$",
    "block_cipher_cycles_per_bit": r"$q^{\mathrm{en}}_i$",
    "block_cipher_round_factor": r"$\kappa_N$",
    "attacker_density": r"$\rho$",
    "leakage_price": r"$\varpi_s$",
    "price_energy_mec": r"$\lambda^j$",
    "velocity_mean": r"$\mu_v$",
    "velocity_std": r"$\sigma_v$",
    "energy_budget_max": r"$E_{\max}$",
    "curb_distance": r"$z$",
    "security_range_v2i": r"$d_{i}^{\mathrm{V2I}}$",
    "security_range_v2v": r"$d_{i}^{\mathrm{V2V}}$",
    "deadline_floor": r"$\tau_{\min}$",
    "deadline_penalty": r"$\lambda_p$",
    "energy_violation_penalty": r"$\lambda_{pE}$",
    "price_energy_local": r"$\lambda^l$",
    "price_energy_av": r"$\lambda^n$",
    "mec_base_price": r"$g_{j0}$",
    "aux_base_price": r"$g_{n0}$",
    "mec_credibility": r"$b_j$",
    "aux_credibility": r"$b_n$",
}

# Đơn vị cần math mode (số mũ). Key = tên tham số.
UNITS_TEX = {
    "kappa_effective_capacitance": r"$\mathrm{W}\cdot\mathrm{s}^{3}/\mathrm{cycles}^{3}$",
    "attacker_density": r"vehicles/vehicle (ratio, not per $\mathrm{m}^{2}$)",
}

# Các tham số thuần kỹ thuật mô phỏng, không đưa vào bảng của bài
EXCLUDE = {
    "sim_area_x", "sim_area_y", "num_attack_vehicles",
    "dt", "episode_length",
    "mec_bandwidth_pool", "aux_bandwidth_pool",
    "cost_normalization_divisor", "resource_multiplier_step",
}

_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    """Escape mọi ký tự đặc biệt của LaTeX trong văn bản thuần."""
    return "".join(_ESCAPE.get(ch, ch) for ch in str(text))


def coerce_number(v):
    """
    Trả về số nếu được. Dự phòng cho trường hợp YAML parse '1.0e9' thành chuỗi
    (regex float của YAML 1.1 đòi dấu ở số mũ). OmegaConf thường đã xử lý, nhưng
    giữ lớp này để script không phụ thuộc phiên bản loader.
    """
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return v
    try:
        text = str(v).strip()
        return int(text) if text.lstrip("+-").isdigit() else float(text)
    except (TypeError, ValueError):
        return v


def fmt_value(v) -> str:
    v = coerce_number(v)
    if isinstance(v, float):
        if v == 0 or (1e-3 <= abs(v) < 1e5):
            # Bỏ .0 thừa: 300.0 -> 300
            return f"{v:g}"
        mantissa, exp = f"{v:.2e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exp = int(exp)
        if mantissa == "1":
            return rf"$10^{{{exp}}}$"
        return rf"${mantissa}\times 10^{{{exp}}}$"
    return latex_escape(v)


def validate(name: str, entry: dict) -> list[str]:
    """Kiểm tra một entry; trả về danh sách lỗi (rỗng nếu hợp lệ)."""
    errors = []
    for field in ("value", "unit", "source"):
        if field not in entry:
            errors.append(f"{name}: thiếu field '{field}'")
    for field in ("unit", "source"):
        text = str(entry.get(field, ""))
        if not text.isascii():
            non_ascii = sorted({ch for ch in text if not ch.isascii()})
            errors.append(
                f"{name}.{field} chứa ký tự không phải ASCII {non_ascii}: {text!r}\n"
                f"      -> `unit` và `source` được xuất trực tiếp vào bảng LaTeX; "
                f"hãy viết bằng ASCII tiếng Anh và chuyển phần giải thích tiếng "
                f"Việt sang field `note` (không xuất ra .tex)."
            )
    if coerce_number(entry.get("value")) is None:
        errors.append(f"{name}.value là None")
    return errors


def build_rows(params: dict) -> tuple[list[str], list[str]]:
    rows, errors = [], []
    for name, entry in params.items():
        if name in EXCLUDE or not isinstance(entry, dict):
            continue
        errors.extend(validate(name, entry))
        symbol = SYMBOLS.get(name, rf"\texttt{{{latex_escape(name)}}}")
        value = fmt_value(entry.get("value"))
        unit = UNITS_TEX.get(name) or latex_escape(entry.get("unit", ""))
        source = latex_escape(entry.get("source", ""))
        rows.append(f"    {symbol} & {value} & {unit} & {source} \\\\")
    return rows, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(ROOT / "configs" / "params_master.yaml"))
    ap.add_argument("--out", default=str(ROOT / "notebooks" / "params_table.tex"))
    ap.add_argument("--check", action="store_true",
                    help="Chỉ validate, không ghi file (dùng cho CI)")
    args = ap.parse_args()

    # OmegaConf: cùng loader mà Hydra dùng -> giá trị khớp với lúc huấn luyện
    params = OmegaConf.to_container(OmegaConf.load(args.master), resolve=True)

    rows, errors = build_rows(params)

    if errors:
        print("LỖI trong params_master.yaml:\n", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    if args.check:
        print(f"OK: {len(rows)} tham số hợp lệ (unit/source ASCII, value là số).")
        return

    lines = [
        "% ============================================================",
        "% AUTO-GENERATED by scripts/gen_param_table.py -- DO NOT EDIT.",
        "% Single source of truth: configs/params_master.yaml (T0.1)",
        "% ============================================================",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Simulation parameters (auto-generated from \texttt{params\_master.yaml})}",
        r"  \label{tab:params}",
        r"  \begin{tabular}{llll}",
        r"    \hline",
        r"    Symbol & Value & Unit & Source \\",
        r"    \hline",
        *rows,
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} parameters to {out}")


if __name__ == "__main__":
    main()
