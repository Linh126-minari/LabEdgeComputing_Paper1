# Báo cáo hoàn thành Phase 0 — Chốt nền tảng

**Ngày:** 26/07/2026 · Theo `KEHOACH_DUAN_VEC.md`

## Quyết định đã chốt (được xác nhận 26/07/2026)

| Quyết định | Lựa chọn |
|---|---|
| T0.1 — Số làn `N_l` | **8** theo [B] Table I |
| T0.3 — Mâu thuẫn Paillier [A] vs block cipher [B] | **Hệ 2 lớp**: Paillier (800 cyc/bit) chống MEC tò mò + block cipher (N_i, 90 cyc/bit) chống eavesdropper → c_en = 890 |
| T0.9 — `η_r`, `η_e` | **Bỏ hoàn toàn** (eavesdropping thụ động theo [B]; jamming là mở rộng T3.2) |
| T0.6 — Công thức `E^w` | **`p_w·max(0, T_remote − T_local)`** (chặt chẽ với mô hình song song) |

## Việc đã làm theo từng task

- **T0.1** — Tạo `configs/params_master.yaml` (mọi tham số: value/unit/source/note). `scenario_default.yaml`, `paillier_params.yaml`, `config.yaml` chỉ tham chiếu qua interpolation, không còn giá trị hardcode. Tạo `scripts/gen_param_table.py` sinh `notebooks/params_table.tex`; `.tex` dùng `\input{params_table.tex}`. Sửa defaults trong `EnvironmentConfig` và `train.py`: `f_j=1 GHz`, `f_n=0.5 GHz`, `U/J/N = 10/1/3`.
- **T0.2** — `ρ = 1e-4` vehicles/vehicle (tỉ lệ, không phải 1/m²). Xóa `_resolve_attacker_density()` + fallback diện tích; bỏ chặn `min(n^e, len(attack_vehicles))`. Docstring ghi rõ ngữ nghĩa `d` = bán kính vùng phát người gửi ([B]).
- **T0.3** — Tách `homomorphic` (key 1024, 800 cyc/bit [A]) khỏi `block_cipher` (N_f=64, N_i=32, 90 cyc/bit [B]); xóa `min(n_i, task.size)`; c_en tổng = 890.
- **T0.4** — Xóa `User.generate_task()` (đường sinh task lệch 8×); chỉ `TaskGenerator` sinh task; `_generate_tasks` trong env dùng bits trực tiếp [102400, 1024000]; thêm assertion bắt sai đơn vị; arrival dùng `1−exp(−λΔt)`.
- **T0.5** — Viết lại `channel.py`: path loss có `d₀ = 1 m` tường minh, `δ₀ = 4.75e-2` hiệu chỉnh (SNR ≈ 25 dB @100m, ≈ 6 dB @300m, rate ≈ 8.3 Mbps @100m với B=1MHz). Xóa hack chia 1000. Tạo `scripts/calibrate_channel.py`.
- **T0.6** — Thêm `E_wait = p_w·max(0, T_remote − T_local)` vào tổng năng lượng; `idle_power = 0.05 W` [A] §6 giờ được **sử dụng**. Đồng thời sửa `E_local = κ·c·s·f²` theo [A] (thay 1 W giả định). Cập nhật `.tex`.
- **T0.7** — MEC: trả lại `g_j = b_j·exp(ω)·g_j0`; AUX: `ε_n = load + 1` (tính cả user yêu cầu) → giá AUX rảnh > 0; `λ_res` và `/1e9` giữ lại, ghi rõ MỞ RỘNG trong code + `.tex`.
- **T0.8** — Thay `tanh(U)` bằng `ϖ₁·U`, `ϖ₁ = 1e-5` ([B] Table I). Với N_i=32 mới, `ϖ₁·U` nằm khoảng [0, ~0.8] và có phương sai (kiểm trong test).
- **T0.9** — Xóa mô hình sigmoid (thay `system_model_paper_style.txt` bằng stub deprecation trỏ về `.tex`); xóa `attack_risk_alpha/beta`, `_get_attack_success_probability`, `effective_rate`, `effective_tx_energy`, `attacker_tx_power`, `jamming_radius`, `η_r`, `η_e` (cả trong `convex_solver.py`). `AttackVehicle` chỉ còn để minh họa.
- **T0.10** — `.tex` đồng bộ: Poisson thay Bernoulli; state dim `4 + 2(J+N)` = 12 (bỏ chiều chết: dummy SNR, `d_att^min`, `θ_sec`); action 3 chiều; `ϖ₁·U` thay `w_u·tanh(U)`; `d` trong `n^e` sửa đúng ngữ nghĩa [B]; ghi chú bỏ downlink theo [A] §3.4/[B] §II-B; thêm mục "Remarks": tái tham số hóa kênh, lỗi in Algorithm 2 dòng 14 của [A] (mean → argmax), thuật ngữ Dueling vs Dueling+Double. Thêm assert nhất quán state/action dim vào `train.py`. Xóa ~57 dòng code chết trùng lặp sau `return` trong `step()`.
- **T0.11** — Double DQN (chọn action bằng online net, đánh giá bằng target net; bật/tắt qua `use_double_dqn`); `learning_starts = 2000`; `SmoothL1Loss` (Huber); soft update tùy chọn (`τ = 0.005`); epsilon: linear decay theo tổng step với `exploration_fraction = 0.3` (mặc định), exponential chỉ decay mỗi episode.
- **T0.12** — `tests/test_formulas.py` (giá trị tính tay từ [A]/[B]: n^e(200)=1.396, S(32,64,1)=0.5, U(1e7,32,1)=312500, giá AUX>0, tỉ số exp, kênh, đơn vị task, Double DQN ≠ DQN) và `tests/test_env_invariants.py` (ràng buộc α, state ∈ [0,1], p_att/S ∈ [0,1], rate đơn điệu, không NaN sau 10 000 step). Sửa `tests/test_env_basic.py` dùng `TaskGenerator`.

## Kiểm tra DoD đã thực hiện (bằng grep tĩnh)

- Không còn hardcode `8192/2048/1e10/10e9` trong `src/`, `configs/` (ngoài ghi chú về lỗi cũ và scenario_dense có chủ đích).
- Không còn `sigmoid`, `attack_risk_*`, `η_r/η_e`, `jamming`, `d_km`, `task_size_range`, `2e-6` trong code.
- `idle_power` có ít nhất một chỗ **sử dụng** (`core.py::step`, E_wait).
- Bảng tham số `.tex` sinh từ `params_master.yaml`.

## Kết quả chạy test (26/07/2026)

Lần 1: **32 passed, 2 failed**. Cả hai đều là lỗi trong *test*, không phải trong code sản phẩm:

1. `test_offload_energy_exceeds_tx_plus_enc` — test không tất định: user 0 được đặt ngẫu nhiên trong vùng 1000×1000 và di chuyển thẳng, nên với seed 42 nó không bao giờ vào vùng phủ MEC (bán kính 300 m quanh (250,250)). **Sửa:** ép `user.position = mec.position` mỗi bước, và thay vì chỉ kiểm "có offload được không", giờ kiểm đúng bất đẳng thức DoD `E_total > E_compute_tx + E_enc`. Để làm được điều đó đã **tách thành phần năng lượng vào `info`/`task_history`** (`task_energy_enc`, `task_energy_wait`, `task_energy_compute_tx`) — đây cũng là tiền đề cho yêu cầu log từng thành phần reward ở T1.2. Thêm test bất biến `E_total = E_compute_tx + E_enc + E_wait`.

2. `test_varpi_scaled_U_not_saturated` — test lấy mẫu `rate ~ Uniform[1e6, 9e6]` **tổng hợp**, over-represent rate cao so với phân phối khoảng cách thực, rồi chặn `max ≤ 1.5`; giá trị thật đạt 2.37. **Sửa:** lấy mẫu từ phân phối state **thực** của môi trường và kiểm đúng DoD của T0.8 — phương sai ≠ 0 (lỗi gốc là `tanh` bão hòa) và `median(ϖ₁·U) ∈ (1e-3, 1)` (cùng bậc với các số hạng khác), `p95 < 3`. Bài kiểm này vẫn đỏ nếu ai đó trả lại `tanh` (phương sai = 0) hoặc trả lại `N_i = 2048` (U nhỏ đi ~64 lần).

`ϖ₁` giữ nguyên `1e-5` theo [B] Table I. Phân tích thang: nhánh MEC (`d_i = 300` → `S = 0.124`) cho `ϖ₁·U ≈ 0.09–0.68`; nhánh V2V (`d_i = 200` → `S = 0.395`) cho `≈ 0.43–1.2`; đuôi trên chỉ vượt 1 khi khoảng cách liên kết rất ngắn. Thêm `scripts/calibrate_varpi.py` để in phân vị thật và so sánh các ứng viên `ϖ₁` — mọi ứng viên đều nằm trong dải `ϖ₁ ∈ [0,1]×10⁻⁵` mà [B] khai báo, nên đổi giá trị vẫn trích dẫn được [B].

## Sửa vòng 2 — sinh bảng LaTeX và Hydra override

**Bug 1 — `gen_param_table.py` sinh LaTeX không biên dịch được.** Ba nguyên nhân độc lập:

1. **PyYAML parse `1.0e9` thành CHUỖI.** Regex float của YAML 1.1 đòi dấu ở số mũ (`1.0e+9`), nên `1.0e9`, `0.5e9`, `1.0e6`, `0.8e6`… đi qua `yaml.safe_load` thành `str` → in thô ra bảng thay vì `$10^{9}$`. (Các giá trị như `3e-13` có dấu `-` nên vẫn parse đúng — vì thế lỗi chỉ xuất hiện ở một nửa số dòng.) Hydra không bị vì OmegaConf dùng loader riêng khoan dung hơn. **Sửa:** script dùng `OmegaConf.load` (cùng loader Hydra dùng → thấy đúng giá trị lúc huấn luyện) + lớp `coerce_number` dự phòng, **và** canonical hoá 8 giá trị trong YAML sang dạng `1.0e+9` để mọi consumer YAML khác cũng đúng.
2. **`^` và `_` chưa escape** trong cột đơn vị/nguồn (`W·s^3/cycles^3`, `1/m^2`, `(0, N_f]`) — `^` ngoài math mode là lỗi pdflatex. **Sửa:** hàm `latex_escape` đầy đủ + `UNITS_TEX` render các đơn vị cần số mũ bằng math mode.
3. **Dấu tiếng Việt trong `unit`/`source`** (`MỞ RỘNG`, `tỉ lệ`, `chuẩn hóa`) — `inputenc utf8` của pdflatex không xử lý được mà không thêm package. **Sửa:** chốt quy ước — `unit`/`source` phải ASCII tiếng Anh (vì xuất trực tiếp vào bài), tiếng Việt chuyển hết sang `note` (không bao giờ xuất ra `.tex`). Script **validate và báo lỗi rõ ràng** nếu vi phạm, thay vì âm thầm sinh file lỗi.

Thêm `tests/test_params_master.py` (9 test) chốt cả ba lỗi trên, cộng hai bất biến mới:

- **`params_master.yaml` ↔ defaults trong code phải khớp** (46 tham số) — nếu ai sửa một bên mà quên bên kia thì đỏ. Đây là cái canh nguyên tắc T0.1 mạnh nhất, trước đó chỉ dựa vào kỷ luật thủ công.
- `c_en = 800 + 90` và `N_i ∈ (0, N_f]`.

**Bug 2 — `python train.py general.num_episodes=5` báo "Key not in struct".** `train.py` đọc `cfg.general.get("num_episodes", 100)` nhưng key này chưa từng được khai báo trong `config.yaml`, nên Hydra (struct mode) từ chối override. **Sửa:** khai báo tường minh `num_episodes: 500`, `eval_interval: 10`, `log_interval: 5` trong `general`. Đồng thời **xoá khối `simulation:`** trong `scenario_default.yaml` — nó chứa `num_episodes: 500` trùng lặp và không code nào đọc, đúng loại "một con số ba phiên bản" mà T0.1 tồn tại để diệt.

## Việc còn lại cần bạn chạy

```bash
python -m pytest tests/ -v                # kỳ vọng 44 passed
python scripts/gen_param_table.py         # sinh lại params_table.tex (phải in "Wrote 41 parameters")
python scripts/calibrate_varpi.py         # xem phân vị ϖ₁·U thật, chốt ϖ₁
python scripts/calibrate_channel.py       # bảng SNR/rate (kiểm 2 mục tiêu OK)
python train.py general.num_episodes=5    # assert state/action dim in ra đúng công thức
```

Nếu `calibrate_varpi.py` báo `CẦN HIỆU CHỈNH`, sửa `varpi_1` trong `configs/params_master.yaml` (không sửa code) rồi chạy lại `pytest` — test đồng bộ sẽ xác nhận.

## Lưu ý cho Phase 1

- `exp(ω)` trong định giá MEC chỉ có động lực học sau **T1.1** (deadline động) — hiện `ω ≡ 1`.
- T0.5 thay đổi mọi số liệu delay/energy/U → phải chạy lại toàn bộ baseline trước khi so sánh.
- `γ` nên tăng 0.95–0.99 sau T1.1; `λ_p` hạ xuống [2,10] theo T1.2 rồi thay bằng Constrained RL (T1.3).
