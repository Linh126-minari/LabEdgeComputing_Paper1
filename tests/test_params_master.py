"""
T0.1 — Canh `configs/params_master.yaml` là single source of truth.

Ba nhóm bất biến:
  1. Cấu trúc: mỗi entry có value/unit/source/note; `value` phải là SỐ
     (bắt lỗi YAML parse "1.0e9" thành chuỗi vì thiếu dấu ở số mũ).
  2. An toàn LaTeX: `unit`/`source` phải ASCII và không chứa ký tự đặc biệt
     chưa escape — vì chúng được xuất trực tiếp vào bảng của bài báo.
  3. Đồng bộ: default trong `EnvironmentConfig` phải khớp params_master;
     nếu ai sửa một bên mà quên bên kia thì test đỏ.
"""
import re
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.env.core import EnvironmentConfig  # noqa: E402

MASTER = ROOT / "configs" / "params_master.yaml"

# params_master key -> EnvironmentConfig field
CODE_MAPPING = {
    "num_users": "num_users",
    "num_mec_servers": "num_mec_servers",
    "num_auxiliary_vehicles": "num_auxiliary_vehicles",
    "num_attack_vehicles": "num_attack_vehicles",
    "sim_area_x": "sim_area_x",
    "sim_area_y": "sim_area_y",
    "task_size_min": "task_size_min",
    "task_size_max": "task_size_max",
    "task_cycles_per_bit": "task_cycles_per_bit",
    "task_deadline_app": "task_deadline",
    "task_arrival_rate": "task_arrival_rate",
    "mec_cpu_frequency": "mec_cpu_frequency",
    "aux_cpu_frequency": "aux_cpu_frequency",
    "user_cpu_frequency": "user_cpu_frequency",
    "kappa_effective_capacitance": "kappa",
    "bandwidth_mec": "bandwidth_mec",
    "bandwidth_v2v": "bandwidth_v2v",
    "noise_power": "noise_power",
    "channel_ref_gain": "channel_ref_gain",
    "channel_ref_distance": "channel_ref_distance",
    "path_loss_exponent": "path_loss_exp",
    "tx_power": "user_tx_power",
    "idle_power": "user_idle_power",
    "mec_coverage_radius": "mec_coverage_radius",
    "curb_distance": "curb_distance",
    "v2v_range": "v2v_range",
    "security_range_v2i": "security_range_v2i",
    "security_range_v2v": "security_range_v2v",
    "deadline_floor": "deadline_floor",
    "energy_budget_max": "energy_budget_max",
    "price_energy_local": "price_energy_local",
    "price_energy_mec": "price_energy_mec",
    "price_energy_av": "price_energy_av",
    "mec_bandwidth_pool": "mec_bandwidth_pool",
    "aux_bandwidth_pool": "aux_bandwidth_pool",
    "mec_base_price": "mec_base_price",
    "aux_base_price": "aux_base_price",
    "mec_credibility": "mec_credibility",
    "aux_credibility": "aux_credibility",
    "cost_normalization_divisor": "cost_normalization_divisor",
    "resource_multiplier_step": "resource_multiplier_step",
    "homomorphic_cycles_per_bit": "he_cycles_per_bit",
    "block_cipher_cycles_per_bit": "block_cipher_cycles_per_bit",
    "block_cipher_round_factor": "block_cipher_round_factor",
    "velocity_min": "velocity_min",
    "velocity_max": "velocity_max",
    "velocity_mean": "velocity_mean",
    "velocity_std": "velocity_std",
    "attacker_density": "attacker_density",
    "block_cipher_frame_length": "frame_length_bits",
    "block_cipher_block_length": "encryption_block_length_bits",
    "lane_count": "lane_count",
    "road_max_speed": "road_max_speed_mps",
    "leakage_price": "leakage_price",
    "deadline_penalty": "deadline_penalty",
    "energy_violation_penalty": "energy_violation_penalty",
    "dt": "dt",
    "episode_length": "episode_length",
}


@pytest.fixture(scope="module")
def params():
    return OmegaConf.to_container(OmegaConf.load(MASTER), resolve=True)


# ---------------------------------------------------------------------------
# 1. Cấu trúc
# ---------------------------------------------------------------------------
def test_every_entry_has_required_fields(params):
    for name, entry in params.items():
        assert isinstance(entry, dict), f"{name} phải là mapping 4 field"
        for field in ("value", "unit", "source", "note"):
            assert field in entry, f"{name} thiếu field '{field}'"


def test_every_value_is_numeric(params):
    """Bắt lỗi YAML: '1.0e9' (thiếu dấu ở số mũ) bị parse thành CHUỖI."""
    for name, entry in params.items():
        value = entry["value"]
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{name}.value = {value!r} (kiểu {type(value).__name__}) không phải số. "
            f"Nếu là số khoa học, viết dạng có dấu ở số mũ: 1.0e+9 thay vì 1.0e9."
        )


def test_scientific_notation_written_canonically():
    """YAML 1.1 đòi dấu ở số mũ — chặn dạng `1.0e9` ngay ở mức văn bản."""
    text = MASTER.read_text(encoding="utf-8")
    offenders = re.findall(r"^\s*value:\s*[0-9.]+e[0-9]", text, flags=re.MULTILINE)
    assert not offenders, (
        f"Các dòng sau viết số khoa học thiếu dấu ở số mũ (PyYAML sẽ parse thành "
        f"chuỗi): {offenders}. Dùng dạng 1.0e+9."
    )


# ---------------------------------------------------------------------------
# 2. An toàn LaTeX
# ---------------------------------------------------------------------------
def test_unit_and_source_are_ascii(params):
    """`unit`/`source` xuất trực tiếp vào .tex — pdflatex + inputenc utf8 không
    xử lý được dấu tiếng Việt. Giải thích tiếng Việt phải nằm ở `note`."""
    for name, entry in params.items():
        for field in ("unit", "source"):
            text = str(entry[field])
            bad = sorted({ch for ch in text if not ch.isascii()})
            assert not bad, (
                f"{name}.{field} chứa ký tự không phải ASCII {bad}: {text!r}. "
                f"Chuyển phần tiếng Việt sang field `note`."
            )


def _strip_latex_safe_constructs(cell: str) -> str:
    """
    Bỏ những chỗ ký tự đặc biệt là HỢP LỆ, để phần còn lại chỉ chứa ký tự thô:
      - math mode `$...$` (trong đó ^ và _ là cú pháp đúng);
      - chuỗi đã escape `\\_`, `\\&`, `\\%`, `\\#`, `\\{`, `\\}`;
      - lệnh dạng `\\textasciicircum{}`, `\\textbackslash{}`...
    """
    without_math = re.sub(r"\$[^$]*\$", "", cell)
    without_cmds = re.sub(r"\\text[a-z]+\{\}", "", without_math)
    return re.sub(r"\\[&%$#_{}]", "", without_cmds)


def test_generated_table_has_no_unescaped_latex_specials():
    """Bảng sinh ra phải biên dịch được: không còn ^ / _ / % / # THÔ ngoài math.

    Lưu ý: `\\_` là dạng escape đúng và phải được chấp nhận; chỉ `_` trần mới là lỗi.
    """
    from gen_param_table import build_rows

    params = OmegaConf.to_container(OmegaConf.load(MASTER), resolve=True)
    rows, errors = build_rows(params)
    assert not errors, f"build_rows báo lỗi: {errors}"
    assert rows, "Không sinh được dòng nào"

    for row in rows:
        cells = [c.strip() for c in row.rstrip(" \\").split("&")]
        assert len(cells) == 4, f"Dòng không đủ 4 cột: {row!r}"
        for cell in cells[1:]:  # Value / Unit / Source
            residual = _strip_latex_safe_constructs(cell)
            for ch in ("^", "_", "%", "#"):
                assert ch not in residual, (
                    f"Ký tự LaTeX '{ch}' chưa escape ngoài math mode trong: {cell!r}"
                )
            assert cell.isascii(), f"Ô không phải ASCII: {cell!r}"


def test_escape_checker_catches_raw_underscore():
    """Meta-test: bộ kiểm ở trên phải phân biệt được `\\_` (đúng) và `_` (lỗi)."""
    assert "_" not in _strip_latex_safe_constructs(r"range (0, N\_f]")
    assert "_" in _strip_latex_safe_constructs(r"range (0, N_f]")
    assert "^" not in _strip_latex_safe_constructs(r"$\mathrm{m}^{2}$")
    assert "^" in _strip_latex_safe_constructs(r"m^2")


def test_generated_table_file_is_up_to_date():
    """params_table.tex phải khớp params_master hiện tại (chạy lại script nếu đỏ)."""
    from gen_param_table import build_rows

    table = ROOT / "notebooks" / "params_table.tex"
    if not table.exists():
        pytest.skip("Chưa sinh params_table.tex")

    params = OmegaConf.to_container(OmegaConf.load(MASTER), resolve=True)
    rows, _ = build_rows(params)
    content = table.read_text(encoding="utf-8")
    missing = [r.strip() for r in rows if r.strip() not in content]
    assert not missing, (
        f"{len(missing)} dòng trong params_master chưa có trong params_table.tex "
        f"(chạy `python scripts/gen_param_table.py`). Ví dụ: {missing[:3]}"
    )


# ---------------------------------------------------------------------------
# 3. Đồng bộ params_master <-> code
# ---------------------------------------------------------------------------
def test_code_defaults_match_params_master(params):
    """Default trong EnvironmentConfig phải khớp params_master (nguyên tắc T0.1)."""
    cfg = EnvironmentConfig()
    mismatches = []
    for yaml_key, field in CODE_MAPPING.items():
        assert yaml_key in params, f"params_master thiếu '{yaml_key}'"
        assert hasattr(cfg, field), f"EnvironmentConfig thiếu field '{field}'"
        expected = float(params[yaml_key]["value"])
        actual = float(getattr(cfg, field))
        if abs(expected - actual) > 1e-12 * max(1.0, abs(expected)):
            mismatches.append(f"{yaml_key}={expected:g} != EnvironmentConfig.{field}={actual:g}")
    assert not mismatches, "Lệch giữa params_master và code:\n  " + "\n  ".join(mismatches)


def test_role_based_encryption_costs(params):
    """Phân vai theo đích: Paillier (MEC) đắt hơn mã khối (AUX) — [A] 800 vs [B] 90 cycles/bit."""
    assert params["homomorphic_cycles_per_bit"]["value"] > params["block_cipher_cycles_per_bit"]["value"]


def test_block_length_within_frame_length(params):
    """N_i phải thuộc (0, N_f] — [B] Eq(6) đòi Phi in [0,1]."""
    n_i = params["block_cipher_block_length"]["value"]
    n_f = params["block_cipher_frame_length"]["value"]
    assert 0 < n_i <= n_f, f"N_i={n_i} không thuộc (0, N_f={n_f}]"
