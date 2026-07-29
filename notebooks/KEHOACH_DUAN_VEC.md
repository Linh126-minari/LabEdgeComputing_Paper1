# Kế hoạch triển khai dự án: Tối ưu Offloading Bảo mật trong Vehicular Edge Computing

**Phiên bản:** 1.0 · **Ngày:** 26/07/2026

---

## 0. Cách dùng tài liệu này

### Ký hiệu nguồn

| Ký hiệu | Bài báo |
|---|---|
| **[A]** | Xue, Yao, Wang — *A dynamic pricing scheme for secure offloading and resource allocation based on the internet of vehicles*, Ad Hoc Networks 161 (2024) 103545 |
| **[B]** | Xiao, Zhao, Feng, Liu, Pei, Shi — *Joint Optimization of Security Strength and Resource Allocation for Computation Offloading in VEC*, IEEE TWC 22(12), 2023 |
| **[T]** | `notebooks/LabEdgeComputing.tex` — bản tóm tắt hiện tại của dự án |
| **[C]** | Code hiện tại trong `src/` |

### Cấu trúc mỗi đầu việc

Mỗi task gồm: **Mục tiêu** → **File liên quan** → **Nội dung chi tiết** → **Hướng dẫn thực hiện** → **Tiêu chí hoàn thành (DoD)** → **Phụ thuộc** → **Ước lượng** (ngày-người).

### Mức ưu tiên

- **P0 — Chặn:** không sửa thì mọi số liệu đầu ra đều vô nghĩa. Làm trước tất cả.
- **P1 — Tính hợp lệ khoa học:** không có thì bài không qua được phản biện.
- **P2 — Đóng góp:** thứ tạo ra novelty để công bố.
- **P3 — Hoàn thiện:** tăng chất lượng, có thể cắt nếu thiếu thời gian.

### Nguyên tắc xuyên suốt

1. **Một nguồn chân lý duy nhất cho tham số.** Mọi con số chỉ tồn tại ở một chỗ (YAML), `.tex` sinh ra từ đó.
2. **Không sửa công thức mà không ghi nguồn.** Mỗi hàm tính toán có docstring trỏ tới `[A] Eq(x)` hoặc `[B] Eq(y)`, hoặc ghi rõ `MỞ RỘNG: không có trong bài gốc`.
3. **Không thêm feature mới khi Phase 0 chưa xong.** Xây trên nền sai chỉ nhân thêm công phải làm lại.
4. **Mỗi thay đổi công thức đi kèm một unit test.** Xem T0.12.

---

## 1. Bảng tổng hợp toàn bộ đầu việc

| ID | Đầu việc | Ưu tiên | Ngày-người | Phụ thuộc |
|---|---|---|---|---|
| **PHASE 0 — Chốt nền tảng** | | | **~12** | |
| T0.1 | Bảng tham số hợp nhất (single source of truth) | P0 | 1.5 | — |
| T0.2 | Sửa mật độ attacker `ρ` và đơn vị | P0 | 0.5 | T0.1 |
| T0.3 | Sửa `N_f`/`N_i`, tách block length khỏi key length | P0 | 0.5 | T0.1 |
| T0.4 | Thống nhất công thức kích thước task | P0 | 0.5 | T0.1 |
| T0.5 | Tái tham số hóa mô hình kênh truyền | P0 | 1.5 | T0.1 |
| T0.6 | Thêm năng lượng chờ `E^w` | P0 | 0.5 | T0.1 |
| T0.7 | Sửa công thức định giá động | P0 | 1 | T0.1 |
| T0.8 | Sửa thang `U` trong reward | P0 | 0.5 | T0.1, T0.3 |
| T0.9 | Dọn mô hình bảo mật trùng lặp + tham số chết | P0 | 1 | T0.1 |
| T0.10 | Đồng bộ `.tex` ↔ code ↔ log | P0 | 1.5 | T0.1–T0.9 |
| T0.11 | Sửa Double DQN, warm-up, Huber loss | P0 | 1 | — |
| T0.12 | Bộ regression test cho công thức | P0 | 2 | T0.2–T0.8 |
| **PHASE 1 — Sửa formulation** | | | **~16** | |
| T1.1 | Deadline sinh từ thời gian lưu trú | P1 | 3 | Phase 0 |
| T1.2 | Chốt một hàm mục tiêu duy nhất | P1 | 2 | Phase 0 |
| T1.3 | Constrained RL (Lagrangian) thay `λ_p=100` | P1 | 4 | T1.2 |
| T1.4 | Sửa ngữ nghĩa MDP (vòng chờ task, discount theo thời gian) | P1 | 2 | Phase 0 |
| T1.5 | Đa user thực sự chiếm tài nguyên (background load) | P1 | 3 | Phase 0 |
| T1.6 | Action masking cho hành động không khả thi | P1 | 1 | T1.4 |
| T1.7 | Hiệu chỉnh chế độ bảo mật (sweep `ρ`, `d`, `N_f`) | P1 | 1 | T0.2, T0.3 |
| **PHASE 2 — Baseline & Evaluation** | | | **~19** | |
| T2.1 | Baseline heuristic (all-local/MEC/AUX/random/greedy) | P0 | 2 | T1.2 |
| T2.2 | Baseline Algorithm 1 [A] — user selection | P1 | 1 | T2.1 |
| T2.3 | Baseline CMSO [A] bằng cvxpy | P1 | 3 | T1.2 |
| T2.4 | Baseline GBD [B] | P2 | 5 | T1.2 |
| T2.5 | Baseline DRL: DQN, DDQN, PPO | P1 | 3 | T0.11 |
| T2.6 | Evaluation harness (multi-seed, CI, metrics) | P0 | 3 | T2.1 |
| T2.7 | Ablation kiểu RTCF/RCOR/RTBL/RVOD [B] | P1 | 1.5 | T2.6, T3.1 |
| T2.8 | Sanity check: tái lập xu hướng hình của [A] và [B] | P1 | 2 | T2.6 |
| **PHASE 3 — Đóng góp mới** | | | **~22** | |
| T3.1 | Đưa block length `N_i` vào action space | P2 | 4 | T1.7 |
| T3.2 | Attacker hiện thực hóa thay kỳ vọng mật độ | P2 | 3 | T1.7 |
| T3.3 | Multi-agent + max-min fairness | P2 | 8 | T1.5 |
| T3.4 | So sánh HE vs AES có chi phí receiver + ciphertext expansion | P2 | 4 | T1.2 |
| T3.5 | (Tùy chọn) Action liên tục / BDQ | P3 | 5 | T3.1 |
| **PHASE 4 — Dataset & sim-to-real** | | | **~10** | |
| T4.1 | Script preprocess dữ liệu Rio bus | P1 | 3 | — |
| T4.2 | Lớp `TraceMobility` | P1 | 2 | T4.1, T1.1 |
| T4.3 | Nghiên cứu khoảng cách sim-to-real | P2 | 4 | T4.2, T2.6 |
| **PHASE 5 — Viết bài** | | | **~10** | |
| T5.1 | Cấu trúc bài + định vị novelty | P1 | 2 | Phase 3 |
| T5.2 | Bảng notation + bảng tham số sinh tự động | P1 | 1.5 | T0.10 |
| T5.3 | Phần Limitations + phê bình có căn cứ 2 bài gốc | P2 | 2 | Phase 3 |
| T5.4 | Dựng hình và bảng kết quả | P1 | 3 | Phase 2, 3 |
| T5.5 | Kiểm tra khả năng tái lập (repro check) | P1 | 1.5 | tất cả |

**Tổng:** ~89 ngày-người. Với 1 người làm ~5 ngày/tuần → khoảng 18 tuần; cắt các mục P3 và T2.4 → khoảng 13–14 tuần.

---

## PHASE 0 — CHỐT NỀN TẢNG

> Toàn bộ Phase này là **sửa lỗi truy được về việc chép sai tham số/công thức từ 2 bài gốc**. Không có đóng góp khoa học nào ở đây, nhưng không làm thì mọi kết quả sau đều không dùng được.

---

### T0.1 — Bảng tham số hợp nhất (single source of truth)

**Mục tiêu.** Xóa tình trạng mỗi con số có 3 phiên bản khác nhau (`.tex` / config / log).

**File liên quan.** Tạo mới `configs/params_master.yaml`; sửa `configs/env/*.yaml`, `configs/agent/*.yaml`, `configs/security/*.yaml`.

**Nội dung chi tiết.**
Hiện tại tồn tại các mâu thuẫn sau (đã đối chiếu với bài gốc):

| Tham số | `[T]` | Config/`[C]` | Bài gốc | Kết luận |
|---|---|---|---|---|
| `f_j` (CPU MEC) | 10 GHz | 1 GHz | **1 GHz** [A] §6 | `.tex` **sai** → sửa `.tex` |
| `f_n` (CPU AUX) | 5 GHz | 0.5 GHz | **0.5 GHz** [A] §6 | `.tex` **sai** → sửa `.tex` |
| `U` / `J` / `N` | 5 / 2 / 3 | 10 / 1 / 3 | 10 / 1 / 3 [A] Fig.3 | chọn **10/1/3**, sửa `.tex` |
| State dim | 20 | `10+2(J+N)` | — | tính lại sau khi chốt `J`,`N` |
| Action dim | 3 | docstring ghi 4, log ghi `(4,)` | — | chốt **3**, sửa docstring + log |
| Task arrival | Bernoulli `λΔt` | Poisson | Poisson [A] | chốt **Poisson**, sửa `.tex` |
| `ρ` | 2e-6 | 2e-6 | **1e-4 xe/xe** [B] Table I | xem T0.2 |
| `N_f` / `N_i` | 8192 / 2048 | 8192 / 2048 | **64** [B] Table I | xem T0.3 |
| `N_l` (số làn) | 3 | 3 | 8 [B] Table I | chọn 3 hoặc 8, ghi rõ lý do |
| `v_max` | 25 m/s | 25 m/s | 24 m/s [B] / 25 m/s [A] | dùng 25 (theo [A]), ghi chú |

**Hướng dẫn thực hiện.**

1. Tạo `configs/params_master.yaml` với **mọi** tham số vật lý, mỗi entry có 4 field:
   ```yaml
   mec_cpu_frequency:
     value: 1.0e9
     unit: "Hz"
     source: "[A] Section 6"
     note: "Paper A: 'The computational power of the MEC server is 1 GHz'"
   ```
2. Các file config khác **chỉ tham chiếu**, không lặp lại giá trị. Với Hydra dùng interpolation:
   ```yaml
   # configs/env/scenario_default.yaml
   defaults:
     - /params_master@_here_
   mec_server:
     cpu_frequency: ${mec_cpu_frequency.value}
   ```
3. Viết `scripts/gen_param_table.py` xuất bảng LaTeX từ `params_master.yaml` → chèn vào `.tex` bằng `\input{}`. Từ giờ không bao giờ gõ tay số vào `.tex`.
4. Với mỗi tham số **không** có trong bài gốc (`attack_rate_penalty`, `attack_energy_penalty`, `attack_risk_alpha/beta`, `weight_secure_capacity`, `attacker_jamming_radius`, `attacker_tx_power`), đặt `source: "MỞ RỘNG (không có trong [A]/[B])"` — đây là danh sách đầu vào cho T0.9.

**DoD.** `grep -r "1e9\|10e9\|8192\|2048" configs/ src/ *.tex` không trả về giá trị hardcode nào ngoài `params_master.yaml`. Bảng tham số trong `.tex` sinh tự động.

**Ước lượng.** 1.5 ngày.

---

### T0.2 — Sửa mật độ attacker `ρ` và đơn vị

**Mục tiêu.** Sửa lỗi sai 50× và sai **bản chất đơn vị** khiến `p_att ≈ 3%` (vô nghĩa).

**File liên quan.** `src/security/overhead_models.py`, `src/env/core.py::_resolve_attacker_density`, `configs/params_master.yaml`.

**Nội dung chi tiết.**
[B] Table I ghi rõ: `ρ = 10⁻⁴` **vehicles/vehicle** — *tỉ lệ attacker trên tổng số xe*, không phải mật độ trên m². Code hiện tại dùng `2e-6` và có fallback `len(attack_vehicles) / area` (đơn vị 1/m²) — tức đang hiểu sai thành mật độ không gian.

Hệ quả: với `ρ=2e-6`, `N_l=3`, `v_max=25` → `γ = 3/75 = 0.04`, `n^e = π·300²·2e-6·0.04 ≈ 0.023` → `S = 0.25^0.023 ≈ 0.968` → `p_att ≈ 3.2%`. Toàn bộ nhánh bảo mật của bài bị triệt tiêu.

Với tham số đúng của [B] (`ρ=1e-4`, `N_l=8`, `v_max=24` → `γ=0.111`): `n^e = 3.49e-5 · d²`. Ở `d=200 m` → `n^e ≈ 1.40`; ở `d=300 m` → `n^e ≈ 3.14`.

**Lưu ý quan trọng — cần hiệu chỉnh, không chỉ thay số:** `n^e` phụ thuộc `d²` nên cực nhạy. Vùng có ý nghĩa của mô hình rất hẹp (xem T1.7). Đừng thay số rồi bỏ qua.

**Hướng dẫn thực hiện.**
1. Trong `params_master.yaml`:
   ```yaml
   attacker_density:
     value: 1.0e-4
     unit: "vehicles/vehicle (tỉ lệ, KHÔNG phải 1/m^2)"
     source: "[B] Table I"
   ```
2. Xóa hoàn toàn hàm `_resolve_attacker_density()` và nhánh fallback `len(attack_vehicles)/area`. Hai cách hiểu đơn vị không được trộn.
3. Bỏ dòng chặn `n_attackers_expected = min(n_e, len(self.attack_vehicles))`. Trong `[B]`, `n^e` là **kỳ vọng** từ mật độ, không phải đếm thực thể; chặn nó bằng số attacker mô phỏng là trộn hai mô hình. (Việc dùng số attacker *thực tế* là một mở rộng riêng — xem T3.2, và phải là lựa chọn tường minh qua config.)
4. Thêm docstring: `Ref: [B] Eq(8): n^e_i = π d_i² ρ γ, γ = N_l/(3 v_max). ρ là tỉ lệ attacker/xe.`
5. **Ghi rõ ngữ nghĩa của `d`:** [B] định nghĩa "*d_i is the coverage range radius of the sending vehicle i*" — bán kính vùng phát của **người gửi**, không phải khoảng cách tới node đích. Code hiện tại (truyền `coverage_radius` / `v2v_range`) **đúng theo [B]**. Cần sửa `[T]` vì `.tex` đang viết `π d_{u,ν}²` với `d` là khoảng cách tới node — đó là chỗ sai, không phải code.

**DoD.** Unit test khẳng định `n^e(d=200m) ∈ [1.3, 1.5]` với tham số [B]. `grep -r "2e-6" src/ configs/` rỗng.

**Ước lượng.** 0.5 ngày.

---

### T0.3 — Sửa `N_f`/`N_i`, tách block length khỏi key length

**Mục tiêu.** Sửa lỗi trộn **hai khái niệm mật mã khác nhau** từ hai bài khác nhau.

**File liên quan.** `configs/security/paillier_params.yaml`, `src/security/overhead_models.py`, `src/env/core.py`.

**Nội dung chi tiết.**
- `[B]` Table I: `N_f = 64` bits — **maximum block length của mã khối đối xứng** (64/128 bit là chuẩn cho block cipher).
- `[A]`: `key_length = 1024` bits — **độ dài khóa Paillier** (mã hóa đồng cấu bất đối xứng).
- `[C]` hiện dùng `N_f = 8192`, `N_i = 2048` — con số 1024/2048 đến từ khóa Paillier của [A], nhưng được nhét vào công thức block length của [B]. **Hai đại lượng này không cùng đơn vị ngữ nghĩa.**

Hệ quả kép:
1. `U = R/(8·N_i)·log₂(8·N_i)·S` bị chia cho `8·2048 = 16384` → sai thang hoàn toàn.
2. `N_i/N_f = 2048/8192 = 0.25` cố định → `Φ` cố định.

**Hướng dẫn thực hiện.**
1. Tách rõ hai tham số trong config, **không** để cùng một block:
   ```yaml
   # Mã hóa đồng cấu (Paillier) — theo [A]
   homomorphic:
     key_length_bits: 1024        # source: [A] Section 3.3
     cycles_per_bit: 800          # source: [A] Section 6 (c^en_m)

   # Mã khối đối xứng — theo [B], dùng cho lượng hóa security strength
   block_cipher:
     frame_length_bits: 64        # N_f, source: [B] Table I
     block_length_bits: 32        # N_i, biến quyết định — xem T3.1
     cycles_per_bit: 90           # source: [B] Table I (q^en_i)
   ```
2. Trong `_compute_security_metrics`, xóa dòng `n_i = min(..., task.size)`. Block length không bị chặn bởi kích thước task — task lớn được chia thành nhiều block.
3. **Xử lý mâu thuẫn khái niệm giữa hai bài** — chọn một trong hai và ghi rõ vào bài:
   - **Phương án 1 (khuyến nghị):** Coi đây là hệ **hai lớp**: mã hóa đồng cấu Paillier (`c_en=800`) bảo vệ nội dung khỏi *MEC server tò mò*, và mã khối lớp liên kết (`N_i`, `q^en=90`) bảo vệ khỏi *eavesdropper trên kênh*. Đây chính là hợp nhất hai mô hình đe dọa và **là một luận điểm novelty**, cần một đoạn giải thích trong bài.
   - **Phương án 2:** Chỉ giữ một lược đồ, khi đó phải bỏ đóng góp #2 của [A] hoặc bỏ lượng hóa của [B].
4. Nếu chọn Phương án 1, `c_en` tổng = `800 + 90 = 890` cycles/bit, và phải ghi rõ trong bảng tham số.

**DoD.** Test: `U` với `R=1e7`, `N_i=32`, `S=1` cho giá trị bậc `1e5`, không phải `1e3`. Config không còn chỗ nào lẫn `key_length` với `block_length`.

**Ước lượng.** 0.5 ngày.

---

### T0.4 — Thống nhất công thức kích thước task

**Mục tiêu.** Sửa lệch 8× giữa hai đường sinh task trong cùng repo.

**File liên quan.** `src/env/entities.py::User.generate_task`, `src/env/traffic.py::TaskGenerator`.

**Nội dung chi tiết.**
- `traffic.py`: `task_size_min = 102400`, `max = 1024000` (bits) → 100–1000 **kilobit**.
- `entities.py`: `uniform(100,1000) * 1024 * 8` = 819 200 – 8 192 000 bits → 100–1000 **KB**.
- **Lệch 8×.** [A] §6 ghi "the task size is between 100 kb and 1000 kb" → kilo**bit** → bản `traffic.py` đúng.
- Lưu ý [B] Table I dùng `D_i ∈ [100, 900] KB` — **kilobyte**, khác [A]. Phải chọn một và ghi rõ.

**Hướng dẫn thực hiện.**
1. Xóa hoàn toàn `User.generate_task()`. Chỉ `TaskGenerator` được phép sinh task (một đường duy nhất).
2. Trong `params_master.yaml` ghi cả đơn vị và nguồn:
   ```yaml
   task_size_min: {value: 102400, unit: "bits (=100 kbit)", source: "[A] Section 6"}
   task_size_max: {value: 1024000, unit: "bits (=1000 kbit)", source: "[A] Section 6"}
   ```
3. Thêm assertion trong `TaskGenerator.__init__`: `assert 1e4 < task_size_min < 1e7` để bắt lỗi đơn vị sớm.
4. Ghi chú trong bài: chọn thang của [A]; nếu chạy thêm cấu hình theo [B] (`[100,900]` KB) thì làm thành scenario riêng để so sánh.

**DoD.** `grep -rn "1024 \* 8" src/` rỗng. Test: 1000 task sinh ra đều nằm trong `[102400, 1024000]`.

**Ước lượng.** 0.5 ngày.

---

### T0.5 — Tái tham số hóa mô hình kênh truyền

**Mục tiêu.** Loại bỏ hack `d_km = d/1000` và làm cho SNR nhất quán về đơn vị.

**File liên quan.** `src/env/channel.py`.

**Nội dung chi tiết.**
Bộ tham số của [A] (`δ₀=1e-13`, `W₀=3e-13`, `θ=4`, `p=0.2 W`) **tự nó không nhất quán** khi `d` tính bằng mét:

```
d = 100 m:  SNR = 0.2 · 100⁻⁴ · 1e-13 / 3e-13 = 6.7e-9  →  rate ≈ 0
```

trong khi Fig. 3 của [A] vẽ xe trải trên ±500 m và hệ vẫn hoạt động. Tức là **[A] thiếu định nghĩa khoảng cách tham chiếu cho `δ₀`**. Hack `d/1000` trong code là *triệu chứng* của việc chép nguyên bộ tham số hỏng: nó nhân SNR lên `10¹²` (với `θ=4`) một cách không ghi chép.

**Đây là chỗ dự án có thể làm tốt hơn bài gốc — cần nói ra trong bài, không che.**

**Hướng dẫn thực hiện.**
1. Viết lại theo dạng path loss có khoảng cách tham chiếu tường minh:
   ```python
   def calc_rate(self, dist, tx_power, bandwidth):
       """
       r = B · log2(1 + p · δ₀ · (d/d₀)^(-θ) / W₀)

       Ref: [A] Eq(9),(17). MỞ RỘNG: [A] không định nghĩa khoảng cách
       tham chiếu d₀ cho δ₀, khiến bộ tham số gốc cho SNR ≈ 0 ở d ≥ 100 m.
       Ta khai báo tường minh d₀ = 1 m và hiệu chỉnh lại δ₀.
       """
       d = max(dist, self.d0)
       path_gain = self.delta_0 * (d / self.d0) ** (-self.theta)
       snr = tx_power * path_gain / self.W0
       return bandwidth * np.log2(1 + snr)
   ```
2. **Hiệu chỉnh `δ₀`** để SNR nằm trong dải thực tế. Mục tiêu: SNR ≈ 25–30 dB ở `d=50 m`, ≈ 5–10 dB ở `d=300 m`. Viết `scripts/calibrate_channel.py` sweep `δ₀` và in bảng `(d, SNR_dB, rate_Mbps)` cho `d ∈ {10, 50, 100, 200, 300, 500}`. Chọn `δ₀` rồi khóa lại.
3. Kiểm tra tính hợp lý: rate ở `d=100 m` với `B=1 MHz` nên ở bậc **vài Mbps**. Với task 1 Mbit và deadline 1 s, `t_up` phải chiếm phần đáng kể nhưng không phải 100% của deadline — nếu không thì offloading luôn thắng hoặc luôn thua và bài toán trở nên tầm thường.
4. Ghi vào bài một câu: *"Bộ tham số kênh của [A] không xác định khoảng cách tham chiếu; chúng tôi tái tham số hóa với `d₀=1 m` và hiệu chỉnh `δ₀` để SNR nằm trong dải thực tế, xem Bảng X."*

**Cảnh báo.** Task này thay đổi **mọi** số liệu delay/energy/`U` phía sau. Làm xong phải chạy lại toàn bộ baseline.

**DoD.** Không còn `d_km` trong code. `scripts/calibrate_channel.py` xuất bảng SNR/rate và bảng này có mặt trong bài. Test: rate đơn điệu giảm theo `d`; rate ở `d=100 m` ∈ [1, 20] Mbps.

**Ước lượng.** 1.5 ngày.

---

### T0.6 — Thêm năng lượng chờ `E^w`

**Mục tiêu.** Bổ sung một thành phần năng lượng có trong [A] nhưng bị bỏ trong code.

**File liên quan.** `src/env/core.py` (hàm tính energy), `configs/params_master.yaml`.

**Nội dung chi tiết.**
[A] Eq(13) và Eq(19): khi node xa xử lý task, xe user vẫn tiêu thụ **công suất chờ** `p_w`:

```
E^w = p_w · t^exe = p_w · α^j s c / (β f_j)      [A] Eq(13)  — nhánh MEC
E^w = p_w · t^exe = p_w · α^n s c / f^n_m        [A] Eq(19)  — nhánh AUX
```

Config đã có `idle_power: 0.05` (khớp [A] §6: "the idle power is 0.05 W") nhưng **không hàm nào dùng nó**. Thiếu số hạng này làm offloading rẻ hơn thực tế một cách hệ thống — thiên lệch trực tiếp vào kết luận chính của bài.

**Hướng dẫn thực hiện.**
1. Thêm vào tổng năng lượng:
   ```python
   E_total = E_local + E_tx_eff + E_enc + E_wait
   # E_wait = p_w · (thời gian tính toán tại node xa)
   ```
2. Chú ý ngữ nghĩa với mô hình song song hiện tại: `[T]` cho các nhánh chạy **song song** (`T = T_enc + max{...}`). Khi đó xe chờ trong khoảng `max{...} - T_local` chứ không phải toàn bộ `t^exe`. Chọn một trong hai và ghi rõ:
   - `E^w = p_w · max(0, T_remote - T_local)` — chặt chẽ với mô hình song song (**khuyến nghị**);
   - `E^w = p_w · T_remote` — trung thực với [A] nhưng [A] không có nhánh song song.
3. Cập nhật `[T]` phần "Energy, Cost, and Security Aggregation" cho khớp.

**DoD.** Test: với `α_l=0` (offload hết), `E_total > E_tx + E_enc`. `grep -rn "idle_power" src/` cho ra ít nhất một chỗ **sử dụng**, không chỉ đọc config.

**Ước lượng.** 0.5 ngày.

---

### T0.7 — Sửa công thức định giá động

**Mục tiêu.** Khôi phục cơ chế định giá — đóng góp #3 của [A] — hiện đang bị vô hiệu.

**File liên quan.** `src/env/core.py::_compute_mec_cost`, `_compute_aux_cost`, `src/env/entities.py::AuxVehicle.get_dynamic_price`.

**Nội dung chi tiết.**
Công thức gốc [A] §3.5:
```
g_j = b_j · exp(ω_m) · g_j0            (MEC)
g_n = b_n · ln(1 + ε_n) · ω_m · g_n0   (AUX)
```
Ba lỗi trong code:

1. **Mất `exp(ω)`:** code dùng `base_price * urgency * ...` (tuyến tính theo `ω`) thay vì `exp(ω)`. Vì `ω = 1/τ` và `τ` cố định 1.0 → `ω ≡ 1` → cả hai đều thành hằng số. Sửa `exp(ω)` chỉ có ý nghĩa **sau khi** làm T1.1 (deadline động).
2. **`ε_n = 0 ⇒ g_n = 0`:** `ln(1+0) = 0` nên xe AUX rảnh có giá **đúng bằng 0** → agent luôn chọn AUX. Đây là lỗi có sẵn trong [A]: Algorithm 1 chọn `n* = min_n[g_n]`, nên một AUX rảnh luôn thắng.
3. **`λ_res = 1 + 0.1·level` và phép chia `/1e9`** trong code **không có trong [A]** — là hệ số tự thêm, chưa ghi chép.

**Hướng dẫn thực hiện.**
1. Trả lại `exp(ω)` cho MEC, giữ đúng dạng [A].
2. Sửa `ε_n` để **tính cả user đang yêu cầu**: `ε_n ≥ 1` → `ln(2) ≈ 0.69 > 0`. Ghi chú: *"MỞ RỘNG so với [A]: `ε_n` bao gồm cả user đang phát yêu cầu, nếu không một AUX rảnh sẽ có giá 0 và luôn được chọn — một suy biến của Algorithm 1 trong [A]."*
3. **Cân nhắc gộp cost và energy như [A] Eq(22).** [A] không tách "cost" khỏi "energy" — năng lượng được **quy đổi thành tiền** qua `λ^l = 1 cents/J`, `λ^n = 0.8 cents/J`. Đây là lý do tổng chi phí của [A] khác 0 dù `g_n = 0`. Nếu bám [A], hàm mục tiêu chỉ có **một** đơn vị (cents) và bài toán sạch hơn nhiều — liên quan trực tiếp tới T1.2.
4. Với `λ_res` và `/1e9`: hoặc bỏ, hoặc giữ và ghi rõ `MỞ RỘNG: hệ số chuẩn hóa để cost nằm trong dải [0, C_max]`.
5. Đảm bảo `ε_n` được cập nhật thực sự — cần T1.5, nếu không `ε_n` vẫn gần như tĩnh.

**DoD.** Test: `g_n(ε=1) > 0`; `g_j(ω=2) / g_j(ω=1) ≈ e`. Có comment trỏ `[A] §3.5` cho từng công thức.

**Ước lượng.** 1 ngày.

---

### T0.8 — Sửa thang `U` trong reward

**Mục tiêu.** Cứu số hạng secure information capacity đang **bão hòa thành hằng số**.

**File liên quan.** `src/env/core.py::_compute_reward`.

**Nội dung chi tiết.**
Code dùng `np.tanh(secure_capacity)`. Với `R ≈ 9e6`, `N_i = 2048`:
```
U = 9e6/(8·2048) · log2(8·2048) · S ≈ 549 · 14 · S ≈ 7690
tanh(7690) = 1.0 (bão hòa hoàn toàn)
```
→ số hạng này là **hằng số `+0.2`**, không có gradient, không đóng góp gì vào việc học.

**[B] đã cho sẵn đáp án:** Table I ghi `ϖ₁ ∈ [0,1] × 10⁻⁵`. Tức [B] cũng phải scale `U` xuống `1e-5` vì đúng lý do này. Đây là cách sửa **có thể trích dẫn**.

**Hướng dẫn thực hiện.**
1. Thay:
   ```python
   # CŨ: secure_capacity_norm = float(np.tanh(max(secure_capacity, 0.0)))
   # MỚI — Ref: [B] Table I, ϖ₁ ∈ [0,1] × 1e-5
   secure_capacity_norm = self.config.varpi_1 * secure_capacity
   ```
   với `varpi_1 = 1e-5` (khai báo trong `params_master.yaml`, `source: "[B] Table I"`).
2. Sau khi làm T0.3 (`N_i = 32`), `U` sẽ đổi thang đáng kể — **hiệu chỉnh lại `ϖ₁`** sao cho số hạng này nằm cùng bậc với các số hạng khác (khoảng `[0, 1]`). Viết một script in phân phối `U` trên 1000 sample rồi chọn `ϖ₁`.
3. Đồng thời sửa `[T]`: công thức reward đang ghi `w_u·tanh(U_t)` → thay bằng `ϖ₁·U_t`.

**DoD.** Test: `U_norm` có phương sai khác 0 trên 100 state ngẫu nhiên (hiện tại phương sai = 0). Histogram của `U_norm` nằm trong `[0, ~1]`.

**Ước lượng.** 0.5 ngày.

---

### T0.9 — Dọn mô hình bảo mật trùng lặp và tham số chết

**Mục tiêu.** Xóa tình trạng repo chứa **hai mô hình bảo mật loại trừ nhau**.

**File liên quan.** `system_model_paper_style.txt`, `src/env/core.py`, `configs/params_master.yaml`, `[T]`.

**Nội dung chi tiết.**
Hai mô hình đang cùng tồn tại:
- **Mô hình sigmoid** (trong `system_model_paper_style.txt`): `p_i = sigmoid(α(1 - d_i/r_i))·exp(-β·sec_level)`, `p_att = 1 - Π(1-p_i)`. **Không có trong [A] hay [B]** — do dự án tự đặt.
- **Mô hình [B] Eq(5)–(9)**: frame vulnerability → đang được code dùng.

Tham số chết cần xử lý: `attack_risk_alpha`, `attack_risk_beta` (chỉ thuộc mô hình sigmoid), `attacker_tx_power`, `attacker_jamming_radius` (chỉ dùng trong wrapper `_get_attack_success_probability`).

Ngoài ra `η_r`, `η_e` (`attack_rate_penalty=0.35`, `attack_energy_penalty=0.45`) và các công thức `R_eff = R(1-η_r p_att)`, `E_eff = E(1+η_e p_att)` **không có trong bài nào**. Đây là phát minh riêng và cần quyết định dứt khoát.

**Hướng dẫn thực hiện.**
1. Xóa mô hình sigmoid khỏi `system_model_paper_style.txt`, hoặc chuyển file này sang `docs/deprecated/`. Chỉ giữ **một** mô tả hệ thống, và nó phải khớp `.tex`.
2. Xóa `attack_risk_alpha`, `attack_risk_beta`, và hàm `_get_attack_success_probability` (wrapper compat không còn ai gọi).
3. **Quyết định về `η_r`, `η_e`** — chọn một:
   - **Bỏ** (khuyến nghị nếu muốn trung thực với [B]): trong [B], `p_att` chỉ ảnh hưởng *security strength* `S` và *secure capacity* `U`, **không** làm giảm rate hay tăng năng lượng. Eavesdropping là **thụ động** — nó không gây suy giảm kênh. Việc để nó giảm rate là mô hình **jamming**, mà [B] không có.
   - **Giữ và biện minh:** nếu muốn mô hình attacker chủ động (jamming), phải mô hình nó như **nhiễu trong mẫu số SINR**, không phải hệ số nhân tuyến tính vào rate. Khi đó đây là một mở rộng đáng viết, và nên gộp vào T3.2.
4. Với `attacker_tx_power` và `attacker_jamming_radius`: chỉ giữ nếu chọn phương án jamming ở bước 3; nếu không thì xóa cùng với việc bỏ thực thể `AttackVehicle` khỏi mô hình tính toán (vẫn có thể giữ để vẽ hình minh họa).

**DoD.** `grep -rn "sigmoid\|attack_risk_alpha\|attack_risk_beta" src/ configs/ docs/` rỗng. Repo có đúng một mô tả hệ thống.

**Ước lượng.** 1 ngày.

---

### T0.10 — Đồng bộ `.tex` ↔ code ↔ log

**Mục tiêu.** Loại bỏ tình trạng docstring và log mô tả một hệ thống khác với hệ thống đang chạy.

**File liên quan.** `[T]`, `src/env/core.py` (docstring class), `scripts/train.py`.

**Nội dung chi tiết.** Danh sách sai lệch phải sửa:

| Chỗ | Hiện tại | Đúng |
|---|---|---|
| Docstring `VehicularEdgeEnvironment` | `Action: [alpha_l, alpha_j, alpha_n, resource]` (4 chiều) | `[alpha_l_level, target, resource_level]` (3 chiều) |
| Log train | `action_dim=(4,)`, `state_dim=15` | phải khớp giá trị thực sau khi chốt `J`,`N` |
| `[T]` §MDP | `n^e = π d_{u,ν}²` với `d` = khoảng cách tới node | `d` = **bán kính vùng phát của người gửi** ([B]) |
| `[T]` §State Transition | "attacker dynamics dynamically modify `p_att`" | **Không đúng** với mô hình [B]: `p_att` là hằng số theo nhánh. Xóa câu này, hoặc làm T3.2 để nó thành đúng |
| `[T]` state security | `[d_att^min, θ_sec, p_att^max]` | `d_att^min` và `p_att^max` là **chiều chết**. Bỏ, hoặc làm T3.2 |
| `[T]` §Reward | `w_u·tanh(U_t)` | `ϖ₁·U_t` (xem T0.8) |
| `[T]` §Task generation | Bernoulli `λΔt` | Poisson (theo code và [A]) |
| `[T]` §Cost | thiếu `λ_res`, thiếu `/1e9` | ghi đủ hoặc bỏ hệ số |
| `[T]` §Delay | không có downlink | ghi rõ *"bỏ qua backhaul theo [A] §3.4 và [B] §II-B"* — cả hai bài đều bỏ qua, nên đây là **hợp lệ**, chỉ cần trích dẫn |

**Hướng dẫn thực hiện.**
1. Sửa `.tex` theo bảng trên.
2. Thêm vào `scripts/train.py` một bước **assert nhất quán** khi khởi động:
   ```python
   assert env.observation_space.shape[0] == 10 + 2*(cfg.num_mec + cfg.num_aux), \
       f"State dim không khớp công thức trong bài: {env.observation_space.shape}"
   log.info(f"State dim = {env.observation_space.shape[0]} "
            f"(công thức: 10 + 2(J+N) = 10 + 2({J}+{N}))")
   ```
3. Rà toàn bộ docstring trong `src/env/core.py` — nhiều docstring còn mô tả phiên bản action 4 chiều cũ.
4. Ghi chú riêng về `[A]` Algorithm 2 dòng 14: bài gốc viết chọn action bằng `a_t = (1/|A|)·Σ_{a'} Q(s,a')` — **đó là lấy trung bình, rõ ràng là lỗi in**; code dùng `argmax` đúng. Ghi vào `.tex` như một sửa lỗi có ích.

**DoD.** Chạy `python scripts/train.py` in ra state/action dim khớp chính xác với công thức trong `.tex`. Không còn docstring nào nói action 4 chiều.

**Ước lượng.** 1.5 ngày.

---

### T0.11 — Sửa Double DQN, warm-up, Huber loss

**Mục tiêu.** Làm agent khớp với mô tả và ổn định hóa huấn luyện.

**File liên quan.** `src/agents/policy.py`.

**Nội dung chi tiết.**
1. **`train_step` hiện là DQN thuần**, không phải Double DQN: dùng `next_q_values.max(dim=1)[0]` từ target network.
2. **Warm-up quá ngắn:** `is_ready(self.batch_size)` cho phép train ngay sau 64 sample (chuẩn: 1 000–10 000).
3. Loss function chưa xác định; với thang reward hiện tại (`λ_p=100`) MSE cho loss ≈ 95 — cần Huber.

**Hướng dẫn thực hiện.**
1. Double DQN (3 dòng):
   ```python
   with torch.no_grad():
       # Chọn action bằng online net, đánh giá bằng target net
       next_actions = self.q_network(next_states_tensor).argmax(dim=1, keepdim=True)
       next_q = self.target_network(next_states_tensor).gather(1, next_actions).squeeze(1)
       target_q = rewards_tensor + self.gamma * next_q * (1 - dones_tensor)
   ```
2. Thêm `learning_starts: 2000` vào config; đổi điều kiện thành `is_ready(self.learning_starts)`.
3. Đổi loss sang `nn.SmoothL1Loss()`.
4. Thêm soft target update (tùy chọn): `τ = 0.005` mỗi step, thay vì hard copy mỗi 100 step. Giữ cả hai qua config để so sánh.
5. **Kiểm tra `update_epsilon()` được gọi ở đâu** — nếu gọi mỗi *step* với `decay=0.995` thì `ε` sụp xuống `0.01` sau ~920 step (chưa hết một episode). Nếu gọi mỗi *episode*, sau 500 episode `ε ≈ 0.08`. Phải xác định và ghi rõ; khuyến nghị dùng linear decay theo tổng số step với `exploration_fraction = 0.3`.
6. **Quan trọng:** giữ nguyên `[A]` gọi phương pháp là **Dueling DQN** (DDRP). Nếu `.tex` hay bài viết gọi là "Dueling DDQN" thì phải nói rõ đó là **mở rộng của dự án** so với [A], vì [A] chỉ dùng dueling. Đồng thời DDQN vẫn phải giữ làm baseline riêng (T2.5) — khi đó phương pháp đề xuất phải là Dueling+Double và baseline là Double thuần, để ablation có nghĩa.

**DoD.** Test: với batch giả lập, target Q của Double DQN khác target của DQN thuần. Loss trung bình sau 5000 step < 10 (hiện ~95).

**Ước lượng.** 1 ngày.

---

### T0.12 — Bộ regression test cho công thức

**Mục tiêu.** Đóng băng các công thức đã sửa để không tái phát lỗi.

**File liên quan.** Tạo mới `tests/test_formulas.py`, `tests/test_env_invariants.py`.

**Nội dung chi tiết.** Mỗi công thức từ hai bài có một test với giá trị số tính tay.

**Hướng dẫn thực hiện.**
1. `tests/test_formulas.py` — kiểm tra từng công thức so với giá trị tính tay:
   ```python
   def test_expected_attackers_paper_B():
       """[B] Eq(8): n^e = π d² ρ γ, γ = N_l/(3 v_max)"""
       m = SecurityOverheadModel(SecurityOverheadConfig(
           attacker_density=1e-4, lane_count=8, max_speed_mps=24))
       # γ = 8/72 = 0.1111; n^e = π·200²·1e-4·0.1111 = 1.396
       assert abs(m.expected_attackers(200.0) - 1.396) < 0.01

   def test_security_strength_paper_B():
       """[B] Eq(7): S = (N_i/N_f)^(n^e)"""
       # N_i=32, N_f=64, n^e=1.0  ->  S = 0.5
       assert abs(m.security_strength(32, 64, 1.0) - 0.5) < 1e-6

   def test_secure_capacity_paper_B():
       """[B] Eq(9): U = R/(8N)·log2(8N)·S"""
       # R=1e7, N=32, S=1  ->  1e7/256 · log2(256) = 39062.5 · 8 = 312500
       assert abs(m.secure_information_capacity(1e7, 32, 1.0) - 312500) < 1

   def test_aux_price_nonzero():
       """[A] §3.5 + MỞ RỘNG T0.7: ε_n ≥ 1 nên g_n > 0"""
       assert aux.get_dynamic_price(urgency=1.0) > 0
   ```
2. `tests/test_env_invariants.py` — bất biến của môi trường:
   - `α_l + α_j + α_n == 1` và `α_j · α_n == 0` (ràng buộc [A] 23b–23d) trên 1000 action ngẫu nhiên;
   - state luôn nằm trong `[0,1]`;
   - `delay > 0`, `energy > 0`, `cost >= 0`, `p_att ∈ [0,1]`, `S ∈ [0,1]`;
   - rate đơn điệu giảm theo khoảng cách;
   - không có `NaN`/`inf` sau 10 000 step.
3. Thêm CI (GitHub Actions hoặc script `make test`) chạy toàn bộ test trước mỗi commit.

**DoD.** `pytest` xanh toàn bộ. Cố ý đưa lại `ρ=2e-6` → test đỏ.

**Ước lượng.** 2 ngày.

---

## PHASE 1 — SỬA FORMULATION

---

### T1.1 — Deadline sinh từ thời gian lưu trú

**Mục tiêu.** Biến mobility từ chỗ trang trí thành yếu tố chi phối bài toán. **Đây là task có tỉ lệ giá trị/công sức cao nhất của cả dự án.**

**File liên quan.** `src/env/core.py`, `src/env/entities.py`, `src/env/traffic.py`.

**Nội dung chi tiết.**
Cả hai bài đều dẫn xuất deadline từ hình học chuyển động:

- **[A] Eq(6)–(8):** `x = 2√(y² − z²)` (chiều dài đoạn đường trong vùng phủ, `y` = bán kính, `z` = khoảng cách RSU tới lề đường); `t^w_m = (x − v_m t_q)/v_m` (thời gian lưu trú còn lại); `u_m(t) = √(z² + (x/2 − v_m t^w_m)²)` (khoảng cách tới MEC theo thời gian).
- **[B] Eq(2):** `τ_i = (L − l_i)/v_i`, và nói thẳng: *"The residence time can be used as the maximum delay for task execution to ensure integrity and timeliness."*

Code hiện dùng `τ = 1.0 s` **cố định**. Hệ quả lan tỏa:
- `ω_t = 1/τ ≡ 1` → chiều state chết, và làm `exp(ω)` trong định giá thành hằng số (T0.7);
- mobility hầu như không ảnh hưởng gì → phần "vehicular" của bài mất ý nghĩa;
- không có coupling thời gian → bài toán suy biến gần thành contextual bandit, khiến `γ=0.9` là hợp lý nhưng cũng khiến DRL không hơn được greedy ngoài nhiễu.

**Hướng dẫn thực hiện.**
1. Thêm vào `MECServer` tham số `curb_distance` (`z` trong [A]) và hàm tính chiều dài đoạn đường trong vùng phủ:
   ```python
   def road_segment_length(self):
       """[A] Eq(6): x = 2√(y² − z²), y = coverage_radius, z = curb_distance"""
       return 2.0 * np.sqrt(max(self.coverage_radius**2 - self.curb_distance**2, 0.0))
   ```
2. Thêm hàm tính thời gian lưu trú còn lại:
   ```python
   def residence_time(self, user, mec):
       """
       [A] Eq(7) / [B] Eq(2): thời gian còn lại của user trong vùng phủ.
       Trả về +inf nếu ngoài vùng phủ (khi đó deadline do AUX/local quyết định).
       """
       if not mec.is_covered(user):
           return np.inf
       x = mec.road_segment_length()
       traveled = self._distance_traveled_inside(user, mec)  # ~ v·t_q
       return max((x - traveled) / max(user.velocity, 1e-3), 0.0)
   ```
3. **Định nghĩa deadline hiệu dụng:**
   ```
   τ_u(t) = min( τ_app ,  t^w_MEC(t) ,  t^w_V2V(t) )
   ```
   trong đó `τ_app` là deadline của ứng dụng (giữ 1.0 s theo [A] làm giới hạn trên), `t^w_MEC` theo [A] Eq(7), và `t^w_V2V` là thời gian hai xe còn trong tầm V2V (tính từ vận tốc tương đối). Ghi rõ trong bài: *"MỞ RỘNG: [A] và [B] chỉ xét lưu trú với hạ tầng cố định; ta mở rộng cho liên kết V2V với thời gian tồn tại liên kết tính từ vận tốc tương đối."* → **đây là một đóng góp nhỏ nhưng sạch.**
4. Cập nhật state: thay `τ̄ = τ/τ_max` bằng deadline động, và `ω_t = 1/max(τ_u(t), 0.1)` giờ trở thành feature thật.
5. Bật lại `exp(ω)` trong định giá MEC (T0.7 bước 1).
6. **Tăng `γ`** lên `0.95–0.99`: giờ đã có coupling thời gian thật (xe sắp ra khỏi vùng phủ → deadline ngắn dần), horizon dài trở nên có nghĩa. Chạy sweep `γ ∈ {0.9, 0.95, 0.99}` và báo cáo.
7. Thêm metric mới: **tỉ lệ task thất bại do mất liên kết** (khác với thất bại do quá deadline ứng dụng) — đây là con số thuyết phục để đưa vào bài.

**DoD.** Histogram `τ_u(t)` có phương sai khác 0 và tương quan âm với vận tốc. `ω_t` có phương sai khác 0. Có metric "link-loss failure rate".

**Ước lượng.** 3 ngày.

---

### T1.2 — Chốt một hàm mục tiêu duy nhất

**Mục tiêu.** Thay hàm mục tiêu hiện tại (không thuộc bài nào) bằng một formulation có thể bảo vệ.

**File liên quan.** `src/env/core.py::_compute_reward`, `[T]` §Reward.

**Nội dung chi tiết.**
- **[A]:** `min Σ_m C_m` — `C_m` là **tiền** (cents), năng lượng quy đổi qua `λ^l=1`, `λ^n=0.8` cents/J; delay là **ràng buộc cứng** (23e).
- **[B]:** `max min_i O_i` với `O_i = (1−x_i)ϖ₁U_i − x_iϖ₂T^c_i`; delay (C2) và energy (C3) là **ràng buộc**.
- **[C]:** weighted sum 5 số hạng chuẩn hóa riêng lẻ + `λ_p=100`, không ràng buộc.

Vì mục tiêu không giống bài nào, **không thể** tái lập Fig.4–9 của [A] hay Fig.6–11 của [B], và không thể tuyên bố đóng góp của bên nào.

Dấu vết rõ nhất: `λ_p = 100` là hệ quả của việc **thay ràng buộc cứng bằng penalty tay** — và nó nuốt ~98% tín hiệu reward (mọi số hạng khác nằm trong `[0,1]`, bonus `+1`). Log cho reward episode ≈ −14 245 ⇒ ~142 lần vi phạm/1000 step.

**Hướng dẫn thực hiện.**
Chọn **một** trong ba phương án, ghi rõ vào bài:

**Phương án 1 — Mục tiêu kinh tế của [A] + bảo mật của [B] (khuyến nghị).**
```
min  E[ Σ_t ( C_u(t) − ϖ₁ U_u(t) ) ]
s.t. T_u(t) ≤ τ_u(t)        (ràng buộc cứng, [A] 23e)
     E_u(t) ≤ E_max         (ràng buộc cứng, [B] C3, E_max = 0.4 J)
```
với `C_u(t)` theo [A] Eq(22) — **năng lượng đã nằm trong `C_u`** qua `λ`, nên chỉ có một đơn vị (cents). Ưu điểm: mục tiêu một chiều, dễ giải thích, giữ được cả hai đóng góp, và trực tiếp dẫn tới T1.3.

**Phương án 2 — Max-min fairness của [B] + tầng AUX của [A].**
```
max min_u O_u,  O_u = ϖ₁U_u − ϖ₂C_u
```
Ưu điểm: giữ được kết quả headline của [B] (fairness, Fig.7–8). Nhược điểm: **bắt buộc** phải làm T1.5 và T3.3 trước, vì max-min vô nghĩa với một user.

**Phương án 3 — Giữ weighted sum** nhưng chuẩn hóa lại thang và biện minh bằng lý thuyết đa mục tiêu. Yếu nhất về mặt phản biện; chỉ chọn nếu thiếu thời gian.

**Các bước cụ thể (theo Phương án 1):**
1. Cài `C_u(t)` đúng [A] Eq(22), gồm cả `λ^l κ f² α_l s c` (năng lượng local quy tiền) và `E^w` từ T0.6.
2. Thêm `λ^l = 1.0`, `λ^n = 0.8` cents/J vào `params_master.yaml` (`source: "[A] Section 6"`).
3. Thêm `E_max = 0.4` J (`source: "[B] Table I"`).
4. Giữ tạm `λ_p` nhưng hạ xuống `λ_p ∈ [2, 10]`, chờ T1.3 thay hoàn toàn.
5. **Log riêng từng thành phần reward** mỗi step (`C_local`, `C_mec`, `C_aux`, `U`, `penalty`) để thấy được trade-off. Đây là điều kiện tiên quyết để debug mọi thứ phía sau.

**DoD.** Reward có một đơn vị duy nhất. Log CSV có cột cho từng thành phần. Không thành phần nào chiếm > 60% biên độ reward.

**Ước lượng.** 2 ngày.

---

### T1.3 — Constrained RL thay `λ_p = 100`

**Mục tiêu.** Xử lý ràng buộc deadline/energy đúng như hai bài gốc, và tạo một đóng góp phương pháp thật.

**File liên quan.** Tạo mới `src/agents/constrained_policy.py`; sửa `src/env/core.py`.

**Nội dung chi tiết.**
Ánh xạ trực tiếp: (23e) của [A] và (C2),(C3) của [B] đều là **ràng buộc cứng**. Weighted-sum + penalty tay không đảm bảo thỏa ràng buộc, nên so sánh với GBD/CVX (vốn luôn thỏa) là **không công bằng về mặt formulation**. Constrained RL sửa đúng điểm này.

**Hướng dẫn thực hiện.**
1. Dùng Lagrangian relaxation với multiplier học được (RCPO / Lagrangian-DQN):
   ```
   L(θ, λ_T, λ_E) = E[ Σ_t r_t ] − λ_T·(E[viol_T] − δ_T) − λ_E·(E[viol_E] − δ_E)
   ```
   trong đó `viol_T = 1[T_u > τ_u]`, `viol_E = max(0, E_u − E_max)`, và `δ_T`, `δ_E` là ngưỡng cho phép (ví dụ `δ_T = 0.05` — chấp nhận 5% vi phạm).
2. Cập nhật multiplier theo dual ascent, chậm hơn policy:
   ```python
   # cập nhật mỗi K episode (K ~ 10), lr_lambda << lr_policy
   self.lambda_T = max(0.0, self.lambda_T + self.lr_lambda * (viol_rate_T - self.delta_T))
   ```
3. Reward truyền vào agent: `r_t − λ_T·viol_T − λ_E·viol_E`, với `λ` **tự điều chỉnh** thay vì hằng số 100.
4. Log `λ_T(t)`, `λ_E(t)` và tỉ lệ vi phạm theo episode — biểu đồ hội tụ của multiplier là một hình đẹp cho bài.
5. Baseline so sánh bắt buộc: cùng agent với `λ_p` cố định ∈ `{1, 10, 100}` → chứng minh Lagrangian tự tìm được điểm tốt hơn mọi giá trị cố định. Đây chính là bằng chứng cho đóng góp.

**DoD.** Tỉ lệ vi phạm deadline hội tụ về gần `δ_T`. Cost thấp hơn phiên bản `λ_p` cố định tốt nhất ở cùng mức vi phạm. Có hình hội tụ `λ`.

**Ước lượng.** 4 ngày.

---

### T1.4 — Sửa ngữ nghĩa MDP

**Mục tiêu.** Loại bỏ vòng chờ task làm sai lệch discount và làm bẩn replay buffer.

**File liên quan.** `src/env/core.py::step`.

**Nội dung chi tiết.** Ba vấn đề trong `step()`:
1. Vòng `while not self.task_queue[user.id]` tự tăng `current_time` và `step_count` một lượng **không xác định trước** khi ra quyết định.
2. Có thể `return state, 0.0, done, {"error": "no_task_timeout"}` — transition reward 0 này **vẫn vào replay buffer**.
3. `γ` áp theo *số quyết định*, không theo *thời gian trôi*: hai transition cách nhau 0.1 s và 5 s bị discount như nhau.

**Hướng dẫn thực hiện.**
Chọn một trong hai (khuyến nghị Phương án A):

**Phương án A — Event-driven MDP với discount theo thời gian (semi-MDP).**
1. `step()` trả về `info["elapsed_time"] = Δt` thực tế.
2. Trong `train_step`, dùng discount lũy theo thời gian:
   ```python
   gamma_eff = self.gamma ** (elapsed_time / self.base_dt)
   target_q = rewards + gamma_eff * next_q * (1 - dones)
   ```
   → cần lưu `elapsed_time` vào replay buffer (thêm một field).
3. **Không lưu** transition khi không có task; thay vào đó chỉ tiến thời gian nội bộ.

**Phương án B — Fixed-slot MDP với no-op.**
1. Mỗi step luôn là `dt = 0.1 s`. Nếu không có task, action là no-op, reward = 0, **và transition này được lưu bình thường** (nó là một phần hợp lệ của MDP).
2. Đơn giản hơn, nhưng buffer chứa nhiều transition rỗng → tăng arrival rate `λ` để bù.

**Chung cho cả hai phương án:**
3. Bỏ hẳn nhánh `return ..., 0.0, done, {"error": ...}`.
4. Đảm bảo `episode_length` đếm **số quyết định**, không đếm cả step chờ — nếu không, tổng reward giữa các run không so sánh được.
5. Chuẩn hóa `reset()` và `step()` theo Gymnasium API (`terminated`, `truncated` riêng), vì code đang trả về `done` kiểu Gym cũ trong khi import `gymnasium`.

**DoD.** Không có transition nào trong buffer có `info["error"]`. Tổng reward/episode có phương sai giữa các seed < 20% (hiện tại phụ thuộc mạnh vào số step chờ ngẫu nhiên).

**Ước lượng.** 2 ngày.

---

### T1.5 — Đa user thực sự chiếm tài nguyên

**Mục tiêu.** Khôi phục cạnh tranh tài nguyên — điều kiện tiên quyết cho cơ chế định giá của [A].

**File liên quan.** `src/env/core.py`.

**Nội dung chi tiết.**
`step()` dùng cứng `self.users[0]`. Config đặt `num_users = 10` nhưng 9 user còn lại chỉ di chuyển và sinh task, **không bao giờ chiếm tài nguyên** MEC/AUX. Hệ quả:
- `ε_n ≈ 0` → định giá động của [A] vô hiệu (xem T0.7);
- `U_j(t) ≈ 0` → ba chiều state resource là hằng số;
- fairness của [B] không định nghĩa được.

**Hướng dẫn thực hiện.**
Ba mức, chọn theo thời gian có:

**Mức 1 — Background load (0.5 ngày, làm ngay).**
Sinh tải ngẫu nhiên có tương quan thời gian (Markov-modulated) cho mỗi MEC/AUX:
```python
# ε_n(t+1) = clip(ε_n(t) + Poisson(arrival) − Binomial(departure), 0, ε_max)
```
Không phải mô hình thật nhưng làm state resource biến động và `ε_n > 0`.

**Mức 2 — 9 user còn lại chạy heuristic (2 ngày, khuyến nghị).**
Cho `users[1..9]` dùng chính Algorithm 1 của [A] (T2.2) + greedy resource, và **thực sự** gọi `_allocate_resources`. Ưu điểm: `ε_n` là số user cạnh tranh **thật**, đúng nghĩa [A]; agent học trong môi trường có đối thủ; vẫn là single-agent nên không tăng độ phức tạp huấn luyện.

**Mức 3 — Multi-agent hoàn chỉnh.** Xem T3.3.

**Các bước cho Mức 2:**
1. Tách logic thực thi task thành `_execute_task(user, action) -> metrics` không phụ thuộc `users[0]`.
2. Trong `step()`: agent quyết định cho `users[0]`; vòng lặp qua `users[1:]` gọi heuristic rồi `_execute_task`.
3. `ε_n` = số user đã chọn AUX `n` trong slot hiện tại (**bao gồm** `users[0]`, theo T0.7).
4. Ràng buộc `Σ_m β_m ≤ 1` ([A] 23j) phải được **thực thi**: nếu tổng tài nguyên yêu cầu vượt 100%, chia tỉ lệ hoặc từ chối. Hiện tại chưa có kiểm tra này.
5. Áp ràng buộc `[A] §3.4`: *"the service of one auxiliary vehicle is limited to one user"* — mỗi AUX phục vụ tối đa 1 user/slot. Đây là ràng buộc rõ ràng trong bài mà code chưa có.
6. Thêm metric: `ε_n` trung bình, độ lợi dụng MEC, số lần bị từ chối do hết tài nguyên.

**DoD.** `ε_n` có phương sai khác 0 và tăng theo `num_users`. `U_j(t)` có phương sai khác 0. Test khẳng định `Σβ ≤ 1` và mỗi AUX ≤ 1 user/slot.

**Ước lượng.** 3 ngày (Mức 1 + Mức 2).

---

### T1.6 — Action masking

**Mục tiêu.** Ngăn agent lãng phí một nửa action space vào hành động không khả thi.

**File liên quan.** `src/agents/policy.py`, `src/env/core.py`.

**Nội dung chi tiết.**
Hiện tại khi user ngoài vùng phủ MEC, nhánh MEC **âm thầm fallback về local**. Vấn đề: agent chọn `d_u=0` (MEC) và `d_u=1` (AUX) cho kết quả **giống nhau** khi cả hai đều ngoài tầm → tín hiệu học bị nhiễu, và ở nhiều state một nửa của 200 action là vô nghĩa.

**Hướng dẫn thực hiện.**
1. Env cung cấp mask:
   ```python
   def action_mask(self) -> np.ndarray:
       """True = action khả thi. Shape = (10, 2, 10) flatten -> (200,)"""
       mec_ok = any(m.is_covered(self.users[0]) for m in self.mec_servers)
       aux_ok = any(self._in_v2v_range(self.users[0], a) for a in self.auxiliary_vehicles)
       # nếu cả hai đều không khả thi, chỉ cho phép α_l = 1 (level 9)
   ```
2. Trong `select_action`, đặt `Q[~mask] = -inf` trước `argmax`; khi explore, sample chỉ trong tập khả thi.
3. Trong `train_step`, mask cả `next_q` để target không dùng action không khả thi. Cần lưu `next_mask` vào replay buffer.
4. **Ablation bắt buộc:** có/không masking, cùng seed, cùng số step. Đây là một đóng góp nhỏ nhưng dễ chứng minh và dễ bảo vệ.

**DoD.** Không lần nào agent chọn nhánh remote khi ngoài tầm. Ablation cho thấy cải thiện tốc độ hội tụ.

**Ước lượng.** 1 ngày.

---

### T1.7 — Hiệu chỉnh chế độ bảo mật

**Mục tiêu.** Đảm bảo số hạng bảo mật hoạt động trong vùng **không suy biến** — và phát hiện này tự nó là một kết quả đáng công bố.

**File liên quan.** Tạo mới `scripts/calibrate_security.py`.

**Nội dung chi tiết.**
Mô hình [B] cực nhạy với `n^e`, và **nghiệm đóng Eq(39) của [B] suy biến ở phần lớn dải tham số**. Phân tích:

`N_i* = round(2^{1/((1−n^e)ln2) − 3})` khi `0 ≤ n^e < 1`, `= N_f` khi `n^e ≥ 1`.

| `n^e` | `1/((1−n^e)·ln2)` | `N_i*` | Nhận xét |
|---|---|---|---|
| 0.1 | 1.60 | `2^(-1.40) ≈ 0` | **suy biến** — không mã hóa |
| 0.3 | 2.06 | `2^(-0.94) ≈ 1` | gần suy biến |
| 0.52 | 3.00 | `2^0 = 1` | ngưỡng dưới của vùng có nghĩa |
| 0.7 | 4.81 | `2^1.81 ≈ 4` | có nghĩa |
| 0.84 | 9.01 | `2^6.01 ≈ 64` | đạt `N_f` |
| ≥ 0.85 | — | `N_f` (bão hòa) | **bão hòa** |

⇒ **Vùng thực sự có nghĩa của nghiệm đóng [B] chỉ là `n^e ∈ [0.52, 0.84]`** — một dải hẹp bất ngờ.

Tệ hơn: với tham số Table I của [B] (`ρ=1e-4`, `N_l=8`, `v_max=24`, `L=100 m` ⇒ bán kính 50 m), `n^e = 3.49e-5 · 50² ≈ 0.087` → **suy biến**. Và Fig.6 của [B] sweep `ρ ∈ [2e-4, 9e-4]` cho `n^e ∈ [0.17, 0.78]` — phần lớn dưới 0.52. Trong khi ở tầm V2V 200 m của dự án, `n^e ≈ 1.4` → **bão hòa phía kia**.

**Đây là một quan sát phân tích mạnh** và là động lực trực tiếp cho T3.1 (học `N_i` thay vì dùng nghiệm đóng).

**Hướng dẫn thực hiện.**
1. Viết `scripts/calibrate_security.py` sweep `(ρ, d, N_l, v_max, N_f)` và xuất heatmap của: `n^e`, `S`, `p_att`, `U`, `N_i*` theo Eq(39).
2. Xác định vùng tham số cho `p_att ∈ [0.1, 0.6]` và `n^e ∈ [0.4, 1.2]` — tức bảo mật thực sự là đánh đổi, không phải hằng số.
3. Chọn bộ tham số vận hành và ghi vào `params_master.yaml` **kèm lý do hiệu chỉnh**.
4. **Viết phát hiện này thành một tiểu mục trong bài** (Section Analysis): nghiệm đóng của [B] chỉ hữu ích trên `n^e ∈ [0.52, 0.84]`; ngoài dải đó nó cho "không mã hóa" hoặc "mã hóa tối đa" bất kể điều kiện khác. Kèm hình `N_i*` vs `n^e` với vùng suy biến/bão hòa tô màu.
5. Nếu cần mở rộng dải, cân nhắc `N_f = 128` (chuẩn AES) thay vì 64 — vẫn hợp lệ về mật mã và mở rộng không gian quyết định cho T3.1.

**DoD.** Có heatmap và bảng chọn tham số. `p_att` trong mô phỏng có phương sai khác 0 và trung bình ∈ `[0.1, 0.6]`. Có hình vùng suy biến của Eq(39).

**Ước lượng.** 1 ngày.

---

## PHASE 2 — BASELINE & EVALUATION [ĐÃ HOÀN THÀNH — Xem BAO_CAO_PHASE2.md]

> **Không có baseline thì không có evaluation.** `.tex` liệt kê 9 baseline nhưng repo hiện không có code cho bất kỳ cái nào. Đây là công việc ưu tiên P0 về mức độ, chỉ xếp sau Phase 0/1 về thứ tự vì baseline phải chạy trên formulation đã chốt.

---

### T2.1 — Baseline heuristic

**Mục tiêu.** Có ngay tập baseline tối thiểu để mọi số liệu có ý nghĩa so sánh.

**File liên quan.** Tạo mới `src/baselines/heuristic.py`, `src/baselines/base.py`.

**Nội dung chi tiết.** Năm baseline mà `.tex` đã hứa:

| Tên | Chính sách |
|---|---|
| All-Local | `α_l = 1` |
| All-MEC | `α_l = 0`, `d_u = 0`; fallback local nếu ngoài vùng phủ |
| All-AUX | `α_l = 0`, `d_u = 1`; fallback local nếu ngoài tầm V2V |
| Random | `α_l`, `d_u`, `l_r` uniform |
| Greedy | duyệt toàn bộ 200 action, chọn action tối thiểu chi phí **tức thời** |

**Hướng dẫn thực hiện.**
1. Định nghĩa interface chung để mọi baseline và agent thay thế nhau được:
   ```python
   class Policy(ABC):
       @abstractmethod
       def select_action(self, state, env=None, training=False) -> np.ndarray: ...
   ```
   Cho phép truyền `env` vì Greedy cần truy vấn trạng thái thật (không chỉ state vector đã chuẩn hóa).
2. Greedy: quan trọng phải làm đúng — nó thường là baseline mạnh nhất trong loại bài này. Với 200 action, brute-force mỗi step là khả thi. Dùng đúng hàm tính chi phí của env (đừng cài lại) để so sánh công bằng.
3. Random dùng seed riêng để tái lập được.
4. Chạy tất cả trên **cùng chuỗi task và cùng quỹ đạo mobility** — cần cố định seed của env riêng biệt với seed của policy. Đây là chi tiết dễ sai và làm mọi so sánh vô nghĩa.

**DoD.** 5 baseline chạy hết 1000 step không lỗi. All-Local có cost tiền = 0 (không có phí offload). Greedy tốt hơn Random về mọi metric.

**Ước lượng.** 2 ngày.

---

### T2.2 — Baseline Algorithm 1 của [A]

**Mục tiêu.** Cài đúng thuật toán chọn mục tiêu offload của bài gốc.

**File liên quan.** `src/baselines/paper_a_alg1.py`.

**Nội dung chi tiết.**
[A] Algorithm 1: nếu `d_{m,j} ≤ d_max` → offload MEC; ngược lại tính `g_n = b_n·ln(1+ε_n)·ω_m·g_n0` cho từng AUX và chọn `n* = min_n[g_n]`.

**Hướng dẫn thực hiện.**
1. Cài đúng theo pseudocode, không "cải tiến".
2. Cần T0.7 (`ε_n ≥ 1`) trước, nếu không mọi AUX đều có giá 0 và `min` trở thành chọn bừa.
3. Kết hợp với một chính sách `α` cố định (ví dụ `α_l = 0.5`) hoặc với CMSO (T2.3) để thành baseline hoàn chỉnh.
4. Ghi rõ trong bài: đây là **Algorithm 1 của [A] nguyên bản**, dùng làm điểm tham chiếu cho phần user selection.

**DoD.** Với user trong vùng phủ MEC, luôn chọn MEC (khớp pseudocode). Với user ngoài vùng phủ, chọn AUX có `g_n` nhỏ nhất.

**Ước lượng.** 1 ngày.

---

### T2.3 — Baseline CMSO của [A] bằng cvxpy

**Mục tiêu.** Có baseline **gần tối ưu** cho nhánh AUX — bắt buộc vì [A] chứng minh bài toán này lồi.

**File liên quan.** `src/baselines/cmso.py`.

**Nội dung chi tiết.**
[A] Algorithm 3 + Eq(32)–(37):
- `(α_l)^ub = t_m f_m/(s_m c_m)` — Eq(32);
- `(α_l)^lb = 1 − t_m R^up_{m,n}(t)/s_m` — Eq(33);
- `f*_m = α_l s_m c_m/t_m` — Eq(34);
- Eq(37) chứng minh `∂²C/∂α_l² ≥ 0` → **lồi** → giải bằng CVX.

Đây là nghiệm gần tối ưu từng slot cho nhánh AUX. Nếu DRL không thắng được nó ở mục tiêu dài hạn, phải giải thích tại sao (thường là: CMSO tối ưu myopic từng slot, DRL tối ưu tích lũy — cần chỉ ra bằng số liệu).

**Hướng dẫn thực hiện.**
1. `pip install cvxpy`.
2. Cài hàm mục tiêu Eq(35) rồi giải theo `α_l` trong `[(α_l)^min, (α_l)^max]` với `(α_l)^max = min{(α_l)^ub, 1}`, `(α_l)^min = max{0, (α_l)^lb}`.
3. **Kiểm tra tính lồi bằng số** trước khi tin: sample `α_l ∈ [0,1]`, tính `C`, kiểm tra đạo hàm cấp hai ≥ 0. Eq(37) của [A] giả định `c_m > c^en_m` (1000 > 800 — đúng với tham số của họ), nhưng nếu ta gộp `c_en = 890` theo T0.3 thì vẫn đúng; nếu đổi tham số phải kiểm lại.
4. Xử lý trường hợp `(α_l)^min > (α_l)^max` (bài toán bất khả thi) — [A] không nói rõ; ghi lại là một điểm chưa đầy đủ của bài gốc và chọn fallback all-local.
5. Ghép với T2.2 thành baseline **"CSO scheme của [A]"** hoàn chỉnh: Algorithm 1 + CMSO cho AUX + DDRP (T2.5) cho MEC.

**DoD.** CMSO cho `α_l*` nằm trong biên Eq(32)–(33). Kiểm tra số khẳng định tính lồi. Chi phí ≤ Greedy trên nhánh AUX.

**Ước lượng.** 3 ngày.

---

### T2.4 — Baseline GBD của [B]

**Mục tiêu.** Có upper/lower bound để báo cáo **optimality gap** của policy DRL. Baseline mạnh nhất có thể có.

**File liên quan.** `src/baselines/gbd.py`.

**Nội dung chi tiết.**
[B] Algorithm 1/2/3: tách thành primal (lồi, cho continuous vars qua KKT Eq(22),(24),(26),(28) + subgradient Eq(30),(31)) và master (MILP, cho `x` và `N`, với `N*` dạng đóng Eq(39)). Primal cho **upper bound**, master cho **lower bound**.

**Cảnh báo về công sức.** Đây là task đắt nhất Phase 2 (5 ngày) và [B] dùng mô hình rate khác (`R = α√p` kiểu TCP thay vì Shannon). Phải chuyển GBD sang mô hình Shannon của [A] — Eq(22) (nghiệm đóng của `p*`) sẽ phải dẫn lại. Nếu thiếu thời gian, **hạ ưu tiên xuống P3 và bỏ**, dùng CMSO (T2.3) làm baseline optimization duy nhất.

**Hướng dẫn thực hiện.**
1. Bước 1: cài **chỉ Eq(39)** (nghiệm đóng của block length) làm baseline nhẹ cho T3.1 — chỉ nửa ngày và đã đủ để so sánh phần đóng góp chính. **Làm bước này trước, đánh giá lại rồi mới quyết định có làm tiếp GBD đầy đủ.**
2. Nếu làm đầy đủ: cài Algorithm 1 ([B]) với subgradient projection, kiểm tra hội tụ của `λ_i` (tái lập Fig.2 của [B] là một sanity check tốt).
3. Cài Algorithm 2 cho nghiệm bất khả thi (slack variables) và Algorithm 3 (vòng ngoài GBD).
4. Báo cáo `gap = |UB − LB|/LB` và so DRL với UB.

**DoD.** Tối thiểu: Eq(39) chạy được, dùng làm baseline cho T3.1. Đầy đủ: GBD hội tụ với `gap < 5%` trên bài toán tĩnh 8 xe (khớp cấu hình [B]).

**Ước lượng.** 0.5 ngày (chỉ Eq 39) hoặc 5 ngày (đầy đủ).

---

### T2.5 — Baseline DRL

**Mục tiêu.** Ablation kiến trúc agent như `.tex` đã hứa.

**File liên quan.** `src/agents/dqn.py`, `src/agents/ddqn.py`, `src/agents/ppo.py`.

**Nội dung chi tiết.** DQN thuần (MLP, không dueling), DDQN (double, không dueling), PPO (actor-critic cho MultiDiscrete). Phương pháp đề xuất = Dueling + Double (sau T0.11).

**Hướng dẫn thực hiện.**
1. Tái sử dụng `DRLAgent`, chỉ tham số hóa `use_dueling` và `use_double` → 4 tổ hợp, bảng ablation 2×2 sạch sẽ.
2. PPO: dùng `MultiDiscrete` gốc (không flatten) — đây là ưu điểm thực sự của PPO và làm so sánh có nội dung. Cân nhắc `stable-baselines3` để tiết kiệm thời gian, miễn là wrapper env đúng chuẩn Gymnasium (cần T1.4).
3. **Cùng budget:** cùng số env step, cùng số seed, cùng khoảng tune hyperparameter. Nếu tune phương pháp đề xuất nhiều hơn baseline thì kết quả không dùng được.

**DoD.** Bảng 2×2 (dueling × double) + PPO, mỗi ô 5 seed, có mean±CI.

**Ước lượng.** 3 ngày.

---

### T2.6 — Evaluation harness

**Mục tiêu.** Hạ tầng đánh giá tin cậy. Không có nó thì mọi con số đều là một lần chạy may mắn.

**File liên quan.** Tạo mới `scripts/evaluate.py`, `src/utils/metrics.py`.

**Nội dung chi tiết.**
Hiện trạng: log cho thấy các run `num_episodes=2`, một run 100 episode chỉ tới episode 5, `random_seed=42` cố định một seed. Không có learning curve, không có CI.

Metric cần thu (khớp với những gì hai bài báo cáo):

| Metric | Nguồn đối chiếu |
|---|---|
| Chi phí kinh tế trung bình (cents) | [A] Fig.4–9 |
| Secure information capacity `U` | [B] Fig.6–10 |
| Delay thực thi trung bình / p95 | cả hai |
| Năng lượng trung bình (J) | cả hai |
| Tỉ lệ thỏa deadline | [A] 23e / [B] C2 |
| Tỉ lệ vi phạm ràng buộc năng lượng | [B] C3 |
| Tỉ lệ thất bại do mất liên kết | **mới** (T1.1) |
| Jain's fairness index / min `O_u` | [B] Fig.7 |
| Phân bố quyết định (%local/%MEC/%AUX) | [A] Fig.11 tương ứng |
| `ε_n` trung bình, độ lợi dụng MEC | [A] §6 |

**Hướng dẫn thực hiện.**
1. `scripts/evaluate.py --policies dueling_ddqn,dqn,ppo,greedy,cmso,all_local --seeds 5 --episodes 200`.
2. Xuất **CSV thô** (mỗi hàng một episode/seed/policy) — không xuất số đã tổng hợp. Tổng hợp là bước riêng để không phải chạy lại khi đổi cách trình bày.
3. Tính mean ± 95% CI (`t`-distribution với `n=5`, hoặc bootstrap nếu phân phối lệch).
4. **Env seed tách khỏi policy seed:** mọi policy phải gặp cùng chuỗi task và cùng quỹ đạo. Thêm test khẳng định điều này.
5. Learning curve: reward, từng thành phần reward, tỉ lệ vi phạm, `λ_T`, `ε` theo episode.
6. Sweep tự động cho các hình: `num_users ∈ {1..10}` (khớp [A] Fig.8 và [B] Fig.8), `task_size ∈ {100..1000} kbit` ([A] Fig.4–7), `ρ` ([B] Fig.6), `N_f` ([B] Fig.10), `N_l ∈ {4,6,8}` ([B] Fig.6).

**DoD.** Một lệnh sinh ra toàn bộ CSV cho mọi hình trong bài. Chạy lại hai lần với cùng seed cho kết quả bit-identical.

**Ước lượng.** 3 ngày.

---

### T2.7 — Ablation kiểu RTCF/RCOR/RTBL/RVOD

**Mục tiêu.** Chứng minh **từng biến** trong "joint optimization" thực sự đóng góp — đúng mẫu mà [B] dùng.

**File liên quan.** `src/baselines/random_variable_ablation.py`.

**Nội dung chi tiết.**
[B] §IV-A dùng một mẫu ablation rất đáng học: random hóa **một** biến, tối ưu phần còn lại:
- RTCF: random task computation frequency;
- RCOR: random computation offloading resources;
- RTBL: random transmission block length;
- RVOD: random vehicular offloading decision.

Áp cho dự án (sau T3.1):
- **R-α:** random tỉ lệ phân chia, tối ưu phần còn lại;
- **R-target:** random chọn MEC/AUX;
- **R-β:** random mức tài nguyên;
- **R-N:** random block length.

Đây chính là bằng chứng mà thiết kế hiện tại đang thiếu: hiện chỉ có 1 trong 6 biến của [B] được tối ưu, nên "joint" là tuyên bố chưa được kiểm chứng.

**Hướng dẫn thực hiện.**
1. Với mỗi biến, freeze nó thành uniform random, huấn luyện lại agent trên các biến còn lại.
2. Báo cáo mức suy giảm so với full joint → cột "đóng góp biên của từng biến".
3. Nếu một biến cho suy giảm không đáng kể (< CI), **nói thẳng trong bài** — thành thật ở đây làm bài mạnh hơn, và giúp cắt biến để đơn giản hóa mô hình.

**DoD.** Bảng ablena 4 dòng + full, mỗi dòng 5 seed. Ít nhất 2 biến cho suy giảm có ý nghĩa thống kê.

**Ước lượng.** 1.5 ngày.

---

### T2.8 — Sanity check: tái lập xu hướng của [A] và [B]

**Mục tiêu.** Chứng minh môi trường mô phỏng đúng về mặt định tính trước khi tin vào kết quả mới.

**File liên quan.** `scripts/reproduce_papers.py`.

**Nội dung chi tiết.**
Không cần trùng số tuyệt đối (mô hình đã khác), nhưng **xu hướng phải đúng**, nếu không là dấu hiệu môi trường sai:

| Xu hướng | Nguồn | Kỳ vọng |
|---|---|---|
| Task size ↑ → chi phí ↑ | [A] Fig.4 | đơn điệu tăng |
| Task size ↑ → tỉ lệ local ↑ | [A] Fig.4 | "user tends to process more tasks locally" |
| Có mã hóa → chi phí cao hơn chút | [A] Fig.6,7 | chênh nhỏ, cùng dạng |
| `num_users` ↑ → chi phí ↑ nhẹ | [A] Fig.8 | tăng nhẹ |
| `ρ` ↑ → `U` ↓ | [B] Fig.6 | đơn điệu giảm |
| `N_l` ↑ → `U` ↓ | [B] Fig.6 | 8 làn < 6 làn < 4 làn |
| `N_f` ↑ → `U` ↓ | [B] Fig.10 | đơn điệu giảm |
| Kênh tốt hơn → tỉ lệ offload ↑ | [B] Fig.11 | tăng |
| Data size ↑ → tỉ lệ local ↑ | [B] Fig.11 | tăng |

**Hướng dẫn thực hiện.**
1. Script chạy từng sweep và **tự động assert dấu của xu hướng** (ví dụ Spearman correlation có dấu đúng, `p < 0.05`).
2. Xu hướng nào **không** tái lập được → điều tra ngay, đừng bỏ qua. Đây là công cụ phát hiện lỗi mạnh nhất mà dự án đang thiếu.
3. Đưa 2–3 hình tái lập vào bài (hoặc phụ lục) như phần validation của môi trường — reviewer rất coi trọng điều này.

**DoD.** ≥ 7/9 xu hướng tái lập được với dấu đúng và ý nghĩa thống kê. Mỗi xu hướng thất bại có ghi chú giải thích.

**Ước lượng.** 2 ngày.

---

## PHASE 3 — ĐÓNG GÓP MỚI

> Đây là phần tạo ra giá trị công bố. Ba task T3.1, T3.2, T3.4 rẻ và sạch; T3.3 đắt nhưng là chỗ hai bài **thực sự** gặp nhau.

---

### T3.1 — Đưa block length `N_i` vào action space

**Mục tiêu.** Khôi phục biến quyết định **cốt lõi** của [B] mà code bỏ mất, và tạo novelty rõ ràng nhất.

**File liên quan.** `src/env/core.py`, `src/agents/policy.py`, `configs/`.

**Nội dung chi tiết.**
`N_i` có **đánh đổi nội tại**: `S = (N_i/N_f)^{n^e}` tăng theo `N_i`, nhưng `U = (R/8N_i)·log₂(8N_i)·S` có hệ số `1/(8N_i)` giảm → tồn tại cực đại nội (đúng Eq(39) của [B]). Đây là biến duy nhất trong bài có cấu trúc đánh đổi đẹp như vậy.

**Luận điểm novelty (viết nguyên vào bài):**

> [B] giải `N_i*` dạng đóng (Eq.39) cho `n^e` **tĩnh**, không có deadline động và không có tầng AUX. Hơn nữa, như phân tích ở T1.7, nghiệm đóng đó chỉ không suy biến trên `n^e ∈ [0.52, 0.84]`. Khi `n^e`, kênh, tải, và deadline (từ thời gian lưu trú) **cùng biến thiên theo thời gian**, nghiệm đóng không còn tối ưu. Ta để agent học `N_i` đồng thời với `α` và `β`, và so sánh trực tiếp với Eq(39) làm baseline.

**Hướng dẫn thực hiện.**
1. Thêm chiều action thứ tư: `N_i ∈ {8, 16, 32, 64}` (nếu `N_f = 64`) hoặc `{16, 32, 64, 128}` (nếu `N_f = 128`). Action space: `10 × 2 × 10 × 4 = 800` — vẫn hoàn toàn khả thi với dueling DQN.
2. **Nối `N_i` với chi phí:** cho chi phí mã hóa phụ thuộc `N_i` để "bảo mật mạnh hơn thì đắt hơn":
   ```
   c_en(N_i) = c_en_base · (1 + κ_N · log2(N_i/N_min))
   ```
   Ghi rõ `MỞ RỘNG: [A] và [B] đều dùng c_en hằng số; ta mô hình sự phụ thuộc vào block length để đóng vòng trade-off security ↔ economic cost.` **Đây là vòng phản hồi mà cả hai bài đều thiếu** và là phần đáng giá nhất của task này.
3. Thêm `N_i` hiện tại vào state (chuẩn hóa `log₂(N_i)/log₂(N_f)`).
4. Baseline so sánh: (a) `N_i` cố định = `N_f` (bảo mật tối đa); (b) `N_i` cố định = `N_f/2`; (c) **Eq(39) của [B]** tính lại mỗi slot theo `n^e` hiện tại; (d) học được (đề xuất).
5. Hình chủ lực: `U` và cost theo `n^e`, so sánh (c) vs (d) — kỳ vọng (d) thắng rõ ở vùng suy biến/bão hòa của (c).

**DoD.** Agent chọn `N_i` khác nhau theo state (phân phối không dồn vào một giá trị). Thắng cả 3 baseline (a)(b)(c) về mục tiêu tổng hợp, với ý nghĩa thống kê.

**Ước lượng.** 4 ngày.

---

### T3.2 — Attacker hiện thực hóa thay kỳ vọng mật độ

**Mục tiêu.** Biến điểm yếu hiện tại (state bảo mật là chiều chết) thành một đóng góp.

**File liên quan.** `src/env/core.py`, `src/security/overhead_models.py`.

**Nội dung chi tiết.**
[B] dùng `n^e = π d² ρ γ` — một **kỳ vọng** từ mật độ, không phụ thuộc vị trí attacker cụ thể. Dự án đã mô phỏng từng `AttackVehicle` có vị trí và mobility, nhưng chỉ dùng để tính `d_att^min` trong state (một feature không ảnh hưởng gì).

Thay kỳ vọng bằng **số attacker thực sự nằm trong vùng phát của người gửi tại thời điểm `t`**:
```
n^e_realized(t) = |{ k ∈ K : ||x_u(t) − x_k(t)|| ≤ d_u }|
```
→ `S` trở thành ngẫu nhiên, biến thiên theo không gian và thời gian, và agent **học được hành vi tránh attacker** (chọn AUX ở phía đối diện, giảm phần offload khi attacker gần, tăng `N_i`).

**Luận điểm novelty:** ablation sạch "expected-density model ([B]) vs realized-attacker model (ours)", định lượng việc mô hình kỳ vọng đánh giá sai rủi ro bao nhiêu phần trăm, và ở chế độ tham số nào sai lệch lớn nhất.

**Hướng dẫn thực hiện.**
1. Thêm flag config `security.attacker_model: "expected" | "realized"` — giữ cả hai để ablation.
2. Cài `n^e_realized`; lưu ý `n^e` giờ là **số nguyên** nên `S = (N_i/N_f)^{n^e}` nhảy bậc → tạo động lực học thú vị hơn nhiều.
3. Cần đủ attacker để có ý nghĩa thống kê: `num_attack_vehicles = 5–10` thay vì 2, và mobility của attacker phải thực sự chạy.
4. Bây giờ `d_att^min` và `p_att^max` trong state **trở thành feature sống** → và câu trong `[T]` ("attacker dynamics dynamically modify `p_att`") **trở thành đúng**. Ghi rõ đây là mở rộng so với [B], không phải mô tả [B].
5. Nếu chọn phương án jamming ở T0.9 bước 3, gộp vào đây: mô hình attacker chủ động như nhiễu trong mẫu số SINR:
   ```
   SINR = p_u·g_u / (W_0 + Σ_k p_k·g_k·1[k đang jam])
   ```
   Đúng vật lý hơn nhiều so với hệ số nhân `(1 − η_r p_att)`.
6. Hình chủ lực: sai lệch giữa `p_att` kỳ vọng và `p_att` hiện thực theo `ρ`; và tỉ lệ task bị rò rỉ dưới hai mô hình.

**DoD.** `p_att` có phương sai khác 0 theo thời gian. Ablation expected vs realized cho khác biệt có ý nghĩa. Agent thể hiện hành vi tránh attacker (tương quan giữa `d_att^min` và `α_l` khác 0).

**Ước lượng.** 3 ngày.

---

### T3.3 — Multi-agent + max-min fairness

**Mục tiêu.** Đóng góp mạnh nhất: chỗ giao thật sự của hai bài mà chưa ai làm.

**File liên quan.** Tạo mới `src/agents/marl.py`; sửa `src/env/core.py` thành multi-agent env.

**Nội dung chi tiết.**
[A] là DDRP đơn user cho từng nhánh; [B] là GBD **tập trung**. Chưa có formulation **phi tập trung, đa agent** với cơ chế định giá làm tín hiệu điều phối.

Ý tưởng cốt lõi: **cơ chế định giá của [A] chính là tín hiệu điều phối cho MARL.** `ε_n` tăng → `g_n` tăng → user tự phân tán sang AUX khác hoặc MEC. Đây là một *market-based coordination* nổi lên tự nhiên từ chính công thức của [A], mà [A] không khai thác vì họ chỉ xét một user tại một thời điểm.

Đồng thời khôi phục kết quả headline của [B]: fairness (Fig.7–8) — vốn không định nghĩa được với một user.

**Hướng dẫn thực hiện.**
1. Chuyển env sang multi-agent: `step(actions: Dict[int, np.ndarray]) -> Dict`. Dùng chuẩn PettingZoo (parallel API) để dùng được thư viện sẵn có.
2. Bắt đầu bằng **Independent DQN với tham số chia sẻ** (đơn giản, thường đủ mạnh): một mạng, mỗi agent forward với state cục bộ của mình. Chỉ nâng lên QMIX/MADDPG nếu IDQN không hội tụ.
3. State cục bộ: giữ như hiện tại + thêm `ε_n` và độ lợi dụng MEC (tín hiệu giá). Không cho agent thấy state của agent khác (phi tập trung).
4. **Reward — chọn một, ghi rõ:**
   - Cá nhân: `r_u = −C_u + ϖ₁U_u` (ích kỷ, cạnh tranh);
   - Max-min theo [B]: `r_u = min_{u'} O_{u'}` (chia sẻ, hợp tác);
   - Kết hợp: `r_u = (1−ξ)·O_u + ξ·min_{u'}O_{u'}` — sweep `ξ` → hình "hiệu quả vs công bằng" rất thuyết phục.
5. Metric fairness: Jain's index, `min_u O_u`, độ lệch chuẩn giữa các user.
6. Baseline bắt buộc: GBD tập trung ([B], T2.4) làm **giới hạn trên**; báo cáo "giá của sự phi tập trung" (price of decentralization). Đây là con số có giá trị.
7. Kiểm tra hiện tượng đáng quan tâm: định giá động có tự phát ngăn được **thundering herd** (mọi user dồn vào một AUX) không? Nếu có, đó là kết quả đẹp; nếu không, chỉ ra tại sao và đề xuất sửa cơ chế giá.

**DoD.** IDQN hội tụ với 10 agent. Jain's index cao hơn baseline greedy phi phối hợp. Có hình trade-off theo `ξ`. Có số "price of decentralization" so với GBD.

**Ước lượng.** 8 ngày.

---

### T3.4 — So sánh HE vs AES đầy đủ

**Mục tiêu.** Khai thác một điểm yếu thực sự của [A] — có khả năng **đảo ngược kết luận Fig.9** của họ.

**File liên quan.** `src/security/crypto_engine.py`, `src/env/core.py`.

**Nội dung chi tiết.**
Đóng góp #2 của [A] là *"chỉ mã hóa ở sender, không giải mã ở receiver"* — chính là lý do CSO thắng AES trong Fig.9. Code hiện **không mô hình chi phí phía receiver** nên **không thể tái lập kết quả headline này**.

Hai bài đã cho sẵn hai con số để dựng baseline: `c_en = 800` cycles/bit (Paillier, [A] §6) vs `q^en = 90` cycles/bit (block cipher, [B] Table I) — chênh gần **9×**.

**Hai điểm phê bình có căn cứ đối với [A]:**

1. **Paillier làm phồng ciphertext ~2×.** Plaintext mod `h` → ciphertext mod `h²` (xem [A] Step 2: `Y = i^X η^h mod h²`). [A] ngầm giả định kích thước dữ liệu truyền không đổi, trong khi [B] **nói rõ** block cipher *"does not increase the transmitted data size"*. Mô hình đúng sự phồng này sẽ tăng `t_up` và năng lượng truyền **~2×** cho nhánh HE. Vì Fig.9 của [A] cho CSO thắng AES chỉ **2.1%** (ở 100 kb) đến **12.3%** (ở 500 kb), một hệ số 2× trên thành phần truyền **rất có thể đảo ngược kết luận**. Đây là một kết quả đáng công bố.

2. **Paillier chỉ đồng cấu cộng.** Chính Eq(40) của [A] chỉ chứng minh *additive* homomorphism. Nên tuyên bố "MEC tính toán trực tiếp trên dữ liệu mã hóa" cho task VEC tổng quát (nhận dạng bản đồ, điều hướng — đều cần phép nhân và hàm phi tuyến) là **overclaim**. Nên hoặc giới hạn phạm vi về task dạng tổng hợp/aggregation, hoặc mô hình thêm chi phí mỗi phép toán trên ciphertext (HE arithmetic đắt hơn plaintext hàng bậc).

**Hướng dẫn thực hiện.**
1. Thêm `crypto_scheme: "paillier" | "aes" | "none"` vào config, mỗi lược đồ có profile:
   ```yaml
   paillier:
     enc_cycles_per_bit: 800        # [A] §6
     dec_cycles_per_bit: 0          # [A]: không giải mã ở receiver
     ciphertext_expansion: 2.0       # MỞ RỘNG: mod h -> mod h²
     supports_computation: "additive_only"
   aes:
     enc_cycles_per_bit: 90         # [B] Table I (q^en)
     dec_cycles_per_bit: 90         # MỞ RỘNG: [A] bỏ qua chi phí này
     ciphertext_expansion: 1.0      # [B]: "does not increase data size"
     supports_computation: "none"
   ```
2. Áp `ciphertext_expansion` vào `t_up` và `E_tx`: `s_tx = expansion · s_off`.
3. Áp `dec_cycles_per_bit` vào thời gian xử lý tại MEC/AUX cho nhánh AES.
4. Hình chủ lực: **tái lập Fig.9 của [A]** với và không có ciphertext expansion → chỉ ra ngưỡng task size mà kết luận đảo chiều.
5. Bổ sung một phân tích độ nhạy: tại giá trị `expansion` nào thì HE hết lợi thế? Câu trả lời dạng ngưỡng rất dễ trình bày.

**DoD.** Tái lập được xu hướng Fig.9 khi `expansion=1` (khớp giả định [A]), và cho thấy đảo chiều ở `expansion=2`. Có hình phân tích ngưỡng.

**Ước lượng.** 4 ngày.

---

### T3.5 — (Tùy chọn) Action liên tục / BDQ

**Mục tiêu.** Bỏ sai số rời rạc hóa; làm so sánh với SAC/TD3/PPO có nội dung.

**Nội dung chi tiết.**
Hiện `α_l ∈ {0, 1/9, ..., 1}` (10 mức). Cả [A] (`α ∈ (0,1)` liên tục) và [B] (`p, f^c, f^en, f^e` liên tục) đều dùng biến liên tục — nên rời rạc hóa là một nguồn sai lệch so với hai bài, đáng đo lường.

**Hướng dẫn thực hiện.**
1. Cách rẻ: sweep số mức rời rạc `∈ {5, 10, 20, 50}` → đo sai số rời rạc hóa so với CMSO (nghiệm liên tục). Nếu chênh nhỏ, **báo cáo và dừng** — đã đủ để biện minh cho thiết kế rời rạc. Chỉ nửa ngày.
2. Nếu chênh lớn: dùng Branching Dueling Q-Network (mỗi chiều một head, tránh nổ tổ hợp) hoặc chuyển sang SAC với action liên tục cho `α`, `β` và rời rạc cho `target`, `N_i` (hybrid action space).

**DoD.** Có bảng sai số rời rạc hóa; hoặc có agent action liên tục thắng bản rời rạc.

**Ước lượng.** 0.5 ngày (chỉ sweep) hoặc 5 ngày (đầy đủ).

---

## PHASE 4 — DATASET & SIM-TO-REAL

---

### T4.1 — Script preprocess dữ liệu Rio bus

**Mục tiêu.** Biến notebook Colab thành pipeline tái lập được, và **sửa các lỗi dữ liệu đang có**.

**File liên quan.** Tạo mới `scripts/preprocess_riobus.py`, `configs/data/riobus.yaml`.

**Nội dung chi tiết.**
Hiện trạng: notebook `25_11_data.ipynb` dùng PySpark trên Colab, path hardcode vào Google Drive shortcut ID, trích 150 trajectory (1 xe/tuyến ngày 25/01/2019) rồi vẽ hình. Ba lỗi phải sửa:

1. **Cột `time` bị hỏng.** Output in ra `date = 01-25-2019` nhưng `time = 2025-11-25 08:50:39` — phần ngày trong timestamp đã bị thay bằng **ngày xử lý**. Mọi việc sắp xếp/nội suy xuyên ngày sẽ sai âm thầm.
2. **Sai đơn vị vận tốc.** Cột `speed` của Rio bus là **km/h** (mẫu: 37, 44, 54), env dùng m/s (15–25 m/s = 54–90 km/h). Không convert thì bus "thực" chạy nhanh gấp 3.6×.
3. **Lấy mẫu thưa và không đều.** GPS bus Rio có khoảng cách mẫu hàng chục giây, env dùng `dt = 0.1 s`. Không nội suy thì "mobility thực" là chuỗi bước nhảy phi vật lý.

**Hướng dẫn thực hiện.**
1. Ghép `date` + `time` thành **một** timestamp ngay ở bước ingest:
   ```python
   df = df.withColumn("ts", F.to_timestamp(
       F.concat_ws(" ", F.col("date"), F.date_format("time", "HH:mm:ss")),
       "MM-dd-yyyy HH:mm:ss"))
   ```
   Sau đó **assert** năm nằm trong 2019, để lỗi này không tái phát.
2. Convert `speed`: `speed_mps = speed_kmh / 3.6`. Assert `0 ≤ speed_mps ≤ 35`.
3. Làm sạch: bỏ record `speed=0` kéo dài (bus đỗ bến), bỏ timestamp trùng, bỏ toạ độ ngoài bounding box Rio (`lat ∈ [-23.1, -22.7]`, `lon ∈ [-43.8, -43.1]`), bỏ jump vận tốc phi lý (> 40 m/s giữa hai mẫu liên tiếp).
4. **Chọn một hành lang giới hạn** khớp sim area: chọn 1–2 km dọc một tuyến (ví dụ tuyến 371 đã có sẵn `GPS_Line_371`), chiếu lat/lon → mét trong hệ ENU cục bộ:
   ```python
   from pyproj import Transformer
   tr = Transformer.from_crs("EPSG:4326", "EPSG:32723", always_xy=True)  # UTM 23S cho Rio
   x, y = tr.transform(lon, lat)
   ```
5. **Nội suy về `dt = 0.1 s`:** dùng nội suy tuyến tính trên `(x, y)` theo `ts`; kiểm tra vận tốc suy ra từ nội suy khớp với cột `speed` (sai lệch < 20%) — đây là kiểm tra chất lượng nội suy quan trọng.
6. Xuất **parquet gọn** (vài MB) vào `data/processed/riobus_corridor.parquet` với schema `[vehicle_id, t_rel, x_m, y_m, v_mps]`. Commit file này (hoặc dùng DVC) để simulator **không cần Spark**.
7. Chuyển toàn bộ sang script + config; xóa path hardcode Google Drive.

**DoD.** `python scripts/preprocess_riobus.py --config configs/data/riobus.yaml` chạy end-to-end. Parquet đầu ra < 20 MB. Assert năm=2019 và `v ≤ 35 m/s` đều pass. Vận tốc nội suy khớp cột `speed`.

**Ước lượng.** 3 ngày.

---

### T4.2 — Lớp `TraceMobility`

**Mục tiêu.** Nối dữ liệu thực vào simulator — hiện tại dataset hoàn toàn không được dùng.

**File liên quan.** `src/env/mobility.py`, `configs/env/*.yaml`.

**Nội dung chi tiết.**
Config ghi `mobility: type: linear`; `mobility.py` chỉ có `LinearMobility` và `RandomWaypointMobility`, không có lớp nào đọc trace.

**Hướng dẫn thực hiện.**
1. Cài lớp mới cùng interface với `LinearMobility` (`step(position, velocity, dt) -> position`) để thay thế được qua config:
   ```python
   class TraceMobility:
       """
       Mobility từ trace GPS thực (Rio bus). Mỗi thực thể gắn với một vehicle_id
       trong parquet; step() tra cứu vị trí tại thời điểm t (đã nội suy về dt).
       """
       def __init__(self, trace_df, vehicle_id, loop=True): ...
       def step(self, position, velocity, dt): ...
       def current_velocity(self): ...   # lấy từ trace, không phải tham số
   ```
2. **Vận tốc phải lấy từ trace**, không dùng `uniform(15,25)`. Đây là điểm khác biệt chính so với mô hình tổng hợp.
3. Gán vai trò: một số `order` làm user vehicle, một số làm auxiliary, một số làm attacker. Chọn theo tiêu chí có ý nghĩa (ví dụ: xe cùng tuyến và gần nhau về thời gian có nhiều cơ hội V2V).
4. **Đặt RSU tại vị trí ven đường thực** trên hành lang đã chọn, với `curb_distance` (`z` trong [A] Eq(6)) đo từ bản đồ. Điều này làm T1.1 (residence time) trở nên thực sự có căn cứ.
5. Xử lý biên: khi trace hết, hoặc loop lại, hoặc kết thúc episode. Ghi rõ lựa chọn vì nó ảnh hưởng thống kê.
6. Thêm scenario config `configs/env/scenario_riobus.yaml` song song với `scenario_synthetic.yaml`, để chạy so sánh được.

**DoD.** Chạy được với `mobility.type: trace`. Quỹ đạo vẽ ra khớp hình trong notebook. Phân phối vận tốc từ trace khác rõ so với `uniform(15,25)`.

**Ước lượng.** 2 ngày.

---

### T4.3 — Nghiên cứu khoảng cách sim-to-real

**Mục tiêu.** Cho dataset một lý do tồn tại về mặt khoa học — và tạo một đóng góp mà reviewer khó bác vì nó là dữ liệu.

**File liên quan.** `scripts/sim2real_study.py`.

**Nội dung chi tiết.**
Cả hai bài dẫn xuất thời gian lưu trú từ **hình học lý tưởng hóa**:
- [A]: đường thẳng 1D, `x = 2√(y²−z²)`, vận tốc hằng số;
- [B]: truncated Gaussian với `μ=13 m/s`, `σ=5`, `v ∈ [2,24] m/s`.

Với trace thực ta tính được **phân phối thời gian lưu trú thực nghiệm** trong footprint RSU thật, và trả lời hai câu hỏi có giá trị:

1. Mô hình lưu trú lý tưởng hóa của [A]/[B] sai lệch bao nhiêu so với thực tế — không chỉ trung bình mà cả **hình dạng phân phối** (bus dừng bến tạo đuôi dài mà mô hình Gaussian không có)?
2. Policy huấn luyện trên mô hình lý tưởng **suy giảm bao nhiêu** khi triển khai trên trace thực?

**Hướng dẫn thực hiện.**
1. So sánh phân phối: vận tốc thực (Rio bus) vs truncated Gaussian của [B] (`μ=13, σ=5, [2,24]`). Kiểm định KS, vẽ CDF chồng nhau. **Dự đoán:** bus dừng đỗ nhiều → khối lượng lớn ở gần 0, lệch mạnh so với Gaussian — và đó chính là phát hiện.
2. So sánh thời gian lưu trú: công thức [A] Eq(7) vs đo thực nghiệm từ trace.
3. **Thí nghiệm transfer chính:**

   | Huấn luyện trên | Đánh giá trên | Ý nghĩa |
   |---|---|---|
   | Synthetic | Synthetic | baseline nội bộ |
   | Synthetic | Trace thực | **sim-to-real gap** |
   | Trace thực | Trace thực | giới hạn trên |
   | Trace thực | Synthetic | kiểm tra chiều ngược |

   Báo cáo bảng 2×2 với mọi metric của T2.6.
4. Nếu gap lớn: thử domain randomization (randomize vận tốc/mật độ khi huấn luyện) và đo mức thu hẹp gap. Đây là kết quả có tính ứng dụng.
5. **Nêu rõ limitation:** bus không đại diện cho lưu lượng xe nói chung (tuyến cố định, dừng đỗ nhiều, phương sai vận tốc thấp, tập trung ở hành lang bus). Nếu có thời gian, bổ sung trace SUMO trên bản đồ OSM cùng khu vực để so chéo. Nêu limitation làm bài mạnh hơn, không yếu hơn.

**DoD.** Bảng transfer 2×2 hoàn chỉnh với CI. Hình so sánh CDF vận tốc và thời gian lưu trú. Kết luận định lượng về sim-to-real gap.

**Ước lượng.** 4 ngày.

---

## PHASE 5 — VIẾT BÀI

---

### T5.1 — Cấu trúc bài + định vị novelty

**Mục tiêu.** Tuyên bố đóng góp khớp chính xác với những gì code làm được.

**Nội dung chi tiết.**
Vấn đề hiện tại của `[T]`: tiêu đề ghép tiêu đề [B] + "Dynamic Pricing" của [A], nhưng trong 6 biến joint optimization của [B] thì code chỉ tối ưu 1 (`f^e`), còn `N_i`, `p_i`, `f^c`, `f^en` đều đóng băng. Reviewer sẽ mở ra và tìm đúng những biến đó.

**Hướng dẫn thực hiện.**
1. **Chỉ tuyên bố những gì đã cài.** Sau Phase 3, danh sách đóng góp khả thi:
   - (C1) Hợp nhất mô hình đe dọa hai lớp: HE chống MEC tò mò ([A]) + lượng hóa block-cipher chống eavesdropper ([B]) — lần đầu, trong kiến trúc ba tầng có AUX;
   - (C2) Deadline sinh từ thời gian lưu trú mở rộng cho liên kết V2V (T1.1);
   - (C3) Học block length trong môi trường động, với phân tích chỉ ra nghiệm đóng của [B] suy biến ngoài `n^e ∈ [0.52, 0.84]` (T1.7 + T3.1);
   - (C4) Mô hình attacker hiện thực hóa thay kỳ vọng mật độ, kèm định lượng sai lệch (T3.2);
   - (C5) Constrained RL thay penalty tay, khớp ràng buộc cứng của cả hai bài (T1.3);
   - (C6) Định giá động làm tín hiệu điều phối trong MARL, và "giá của sự phi tập trung" so với GBD (T3.3);
   - (C7) Chỉ ra ciphertext expansion của Paillier có thể đảo ngược kết luận HE-vs-AES của [A] (T3.4);
   - (C8) Đánh giá sim-to-real trên trace GPS thực (T4.3).
2. **Chọn 3–4 đóng góp làm trục chính**, còn lại xuống mục phụ. Bài có 8 đóng góp ngang hàng luôn bị đọc là không có đóng góp nào. Khuyến nghị trục chính: **C3 + C4 + C5** (một cụm mạch lạc về "bảo mật thích ứng dưới ràng buộc cứng"), hoặc **C6 + C2** nếu làm được MARL.
3. Related work phải nêu rõ [A] và [B] và **khoảng trống** giữa chúng: [A] có kiến trúc + kinh tế nhưng bảo mật chỉ là chi phí mã hóa; [B] có lượng hóa bảo mật nhưng không có AUX, không có kinh tế, và giải tĩnh tập trung.
4. Bảng so sánh với related work có các cột: có AUX? có kinh tế? lượng hóa bảo mật? block length là biến? động/tĩnh? phi tập trung? dữ liệu thực?

**DoD.** Danh sách đóng góp có ánh xạ 1-1 tới task đã hoàn thành. Có bảng so sánh related work.

**Ước lượng.** 2 ngày.

---

### T5.2 — Bảng notation + bảng tham số sinh tự động

**Mục tiêu.** Không bao giờ để bài và code lệch nhau lần nữa.

**Hướng dẫn thực hiện.**
1. `scripts/gen_param_table.py` (từ T0.1) xuất `tables/params.tex`, `.tex` dùng `\input{tables/params.tex}`.
2. Bảng notation: mỗi ký hiệu có cột **Nguồn** ([A] Eq / [B] Eq / MỞ RỘNG). Cột này rất có giá trị với reviewer vì nó cho thấy chính xác đâu là phần mới.
3. Thêm bước CI kiểm tra `tables/params.tex` được sinh lại từ config hiện tại và không có diff — nếu ai sửa config mà quên sinh lại, CI đỏ.

**DoD.** Không có số nào gõ tay trong `.tex`. CI kiểm tra tính đồng bộ.

**Ước lượng.** 1.5 ngày.

---

### T5.3 — Phần Limitations + phê bình có căn cứ

**Mục tiêu.** Chủ động xử lý những điểm reviewer sẽ tấn công, và biến một số thành đóng góp.

**Nội dung chi tiết.** Nội dung đã sẵn từ các task trước:

**Phê bình có căn cứ đối với hai bài gốc** (đưa vào Related Work hoặc Discussion):
- [A] không định nghĩa khoảng cách tham chiếu cho `δ₀`, khiến bộ tham số cho SNR ≈ 0 ở `d ≥ 100 m` (T0.5);
- [A] có suy biến `ε_n = 0 ⇒ g_n = 0`, nên Algorithm 1 luôn chọn AUX rảnh (T0.7);
- [A] Algorithm 2 dòng 14 in sai (trung bình thay vì argmax) (T0.10);
- [A] bỏ qua ciphertext expansion của Paillier, có thể đảo ngược kết luận Fig.9 (T3.4);
- [A] overclaim về "tính toán trên dữ liệu mã hóa" khi Paillier chỉ đồng cấu cộng (T3.4);
- [B] nghiệm đóng Eq(39) suy biến ngoài `n^e ∈ [0.52, 0.84]`, và tham số Table I của chính họ (`d=50 m` ⇒ `n^e ≈ 0.087`) nằm trong vùng suy biến (T1.7);
- [B] không xét tầng AUX, không xét chi phí kinh tế.

**Limitations của chính dự án** (nêu thẳng):
- bỏ qua downlink/backhaul (theo [A] §3.4 và [B] §II-B — **cả hai bài đều bỏ qua**, nên trích dẫn được);
- bỏ qua nhiễu đồng kênh (theo [A]: "we assume that there is no signal interference between the vehicles");
- kênh không có fading nhỏ (nếu chưa làm) — nêu là hướng tương lai;
- dữ liệu bus không đại diện cho lưu lượng xe chung;
- mã hóa ở chế độ mô phỏng (`mode: simulation`), không chạy Paillier thật.

**Hướng dẫn thực hiện.** Mỗi điểm phê bình phải kèm **bằng chứng số** (một hình hoặc một con số), không chỉ nhận định. Đó là ranh giới giữa phê bình có giá trị và phàn nàn.

**DoD.** Mỗi điểm phê bình có bằng chứng định lượng. Phần limitations không né tránh điểm nào đã biết.

**Ước lượng.** 2 ngày.

---

### T5.4 / T5.5 — Dựng hình và kiểm tra tái lập

**Hướng dẫn thực hiện.**
1. Mọi hình sinh từ CSV của T2.6 bằng script (`scripts/make_figures.py`), không sửa tay.
2. Mỗi hình có error bar/vùng CI. Hình không có CI sẽ bị hỏi.
3. Repro check: clone repo sạch → `make setup && make test && make experiments && make figures` phải tái tạo mọi hình.
4. Ghi `requirements.txt` khóa version, ghi seed vào từng file kết quả.
5. Viết `README.md` với: cách chạy, cách lấy dữ liệu Rio bus, thời gian chạy dự kiến, phần cứng dùng.

**Ước lượng.** 3 + 1.5 ngày.

---

## 2. Lịch trình gợi ý (14 tuần, 1 người)

| Tuần | Nội dung chính |
|---|---|
| 1–2 | Phase 0 toàn bộ (T0.1 → T0.12). **Không làm gì khác cho tới khi test xanh.** |
| 3 | T1.1 (residence-time deadline), T1.7 (hiệu chỉnh bảo mật) |
| 4 | T1.2 (chốt mục tiêu), T1.4 (MDP), T1.6 (masking) |
| 5 | T1.5 (đa user), T2.1 (baseline heuristic), T2.6 (harness) — bắt đầu |
| 6 | T1.3 (constrained RL), T2.6 hoàn tất |
| 7 | T2.2, T2.3 (CMSO), T2.5 (baseline DRL) |
| 8 | T2.8 (sanity check tái lập) → **cổng kiểm soát: nếu xu hướng không tái lập, dừng và điều tra** |
| 9–10 | T3.1 (block length), T3.2 (attacker hiện thực hóa), T2.7 (ablation) |
| 11 | T4.1, T4.2 (dataset + TraceMobility) |
| 12 | T3.4 (HE vs AES), T4.3 (sim-to-real) |
| 13 | T5.1, T5.2, T5.4 (viết + hình) |
| 14 | T5.3, T5.5 (limitations + repro), đệm |

**Nếu có thêm 3–4 tuần:** chèn T3.3 (MARL) vào tuần 11–14 và đẩy phần còn lại. Nếu MARL là trục chính của bài thì nên làm sớm hơn (tuần 9) và cắt T3.4.

**Cổng kiểm soát bắt buộc:**
- **Sau tuần 2:** `pytest` xanh toàn bộ, không còn số hardcode. Không đạt → không đi tiếp.
- **Sau tuần 8:** ≥ 7/9 xu hướng của hai bài tái lập được. Không đạt → môi trường còn sai, quay lại sửa.
- **Sau tuần 10:** ít nhất một đóng góp có kết quả thắng baseline với ý nghĩa thống kê. Không đạt → xem lại định vị bài.

---

## 3. Rủi ro và phương án dự phòng

| Rủi ro | Dấu hiệu sớm | Phương án |
|---|---|---|
| Hiệu chỉnh kênh (T0.5) làm mọi baseline phải chạy lại | Xảy ra chắc chắn | Làm T0.5 **sớm nhất có thể**, trước khi có bất kỳ kết quả nào cần giữ |
| Chế độ bảo mật vẫn suy biến sau hiệu chỉnh | `p_att` phương sai ~0 sau T1.7 | Đổi `N_f` sang 128; hoặc chuyển sang mô hình hiện thực hóa (T3.2) sớm hơn kế hoạch |
| DRL không thắng được Greedy/CMSO | T2.8 cho DRL ≈ Greedy | Dấu hiệu bài toán thiếu coupling thời gian → tăng cường T1.1 và T1.5; nếu vẫn vậy, **báo cáo trung thực** và định vị bài là "phân tích khi nào DRL đáng dùng trong VEC" — vẫn là bài có giá trị |
| GBD (T2.4) quá đắt | Quá 5 ngày mà chưa hội tụ | Cắt xuống chỉ Eq(39); dùng CMSO làm baseline optimization duy nhất |
| MARL (T3.3) không hội tụ | IDQN phân kỳ với > 5 agent | Giảm số agent; chuyển sang parameter sharing + curriculum (tăng dần số agent); hoặc cắt T3.3 và dùng T1.5 Mức 2 |
| Dữ liệu Rio bus không dùng được (thiếu độ dày về không gian/thời gian) | Sau T4.1: không đủ xe đồng thời trong bán kính V2V | Bổ sung SUMO trên OSM Rio; hoặc chuyển dataset sang mục "validation của mô hình lưu trú" thay vì driver của mô phỏng |
| Phạm vi phình ra, không kịp deadline | Hết tuần 10 mà chưa xong Phase 2 | Cắt theo thứ tự: T3.5 → T2.4 → T3.3 → T4.3. **Không bao giờ cắt Phase 0 hoặc T2.6.** |

---

## 4. Checklist trước khi submit

**Tính đúng đắn**
- [ ] `pytest` xanh; mọi công thức có test với giá trị tính tay
- [ ] Không số hardcode ngoài `params_master.yaml`
- [ ] Mọi công thức có docstring trỏ `[A] Eq(x)` / `[B] Eq(y)` / `MỞ RỘNG`
- [ ] `.tex`, config, docstring, log **thống nhất tuyệt đối**
- [ ] Không có `NaN`/`inf` sau 100k step
- [ ] Không state feature nào có phương sai = 0

**Tính hợp lệ khoa học**
- [ ] Hàm mục tiêu là **một** và bảo vệ được, ánh xạ rõ về [A] hoặc [B]
- [ ] Ràng buộc deadline/energy được xử lý như ràng buộc, không phải penalty tay
- [ ] ≥ 7/9 xu hướng của hai bài tái lập được
- [ ] Mọi kết quả có ≥ 5 seed và CI 95%
- [ ] Mọi policy được đánh giá trên cùng chuỗi task và cùng quỹ đạo
- [ ] Baseline được tune ngang với phương pháp đề xuất
- [ ] Có baseline optimization-theoretic (CMSO tối thiểu)
- [ ] Có ablation cho từng biến quyết định (T2.7)

**Tính đóng góp**
- [ ] 3–4 đóng góp trục chính, ánh xạ 1-1 tới code đã cài
- [ ] Không tuyên bố tối ưu biến nào đang bị đóng băng
- [ ] Mỗi điểm phê bình bài gốc có bằng chứng số
- [ ] Limitations nêu thẳng, không né

**Tính tái lập**
- [ ] Repo sạch chạy được end-to-end bằng một lệnh
- [ ] `requirements.txt` khóa version; seed ghi vào từng file kết quả
- [ ] `README` có hướng dẫn lấy dữ liệu và thời gian chạy dự kiến
- [ ] Mọi hình sinh từ script, không sửa tay

---

## 5. Phụ lục: Bảng tham số hợp nhất đề xuất (bản nháp cho T0.1)

| Tham số | Ký hiệu | Giá trị | Đơn vị | Nguồn |
|---|---|---|---|---|
| Diện tích mô phỏng | `X×Y` | 1000×1000 | m | [A] Fig.3 (±500 m) |
| Số user | `U` | 10 | — | [A] §6 |
| Số MEC | `J` | 1 | — | [A] §6 |
| Số AUX | `N` | 3 | — | [A] §6 |
| Số attacker | `K` | 5–10 | — | MỞ RỘNG (T3.2 cần đủ mẫu) |
| Vận tốc xe | `v` | [15, 25] | m/s | [A] §6 |
| Bán kính phủ MEC | `L` | 300 | m | [A] §6 |
| Khoảng cách RSU–lề | `z` | cần chọn | m | [A] Eq(6) — **[A] không cho giá trị** |
| Tầm V2V | `r_v2v` | 200 | m | MỞ RỘNG ([A] không nêu rõ) |
| CPU MEC | `f_j` | 1.0 | GHz | [A] §6 |
| CPU AUX | `f_n` | 0.5 | GHz | [A] §6 |
| CPU user (max) | `F_loc` | 2.0 | GHz | [B] Table I |
| CPU edge (max) | `F_e` | 3.0 | GHz | [B] Table I |
| Kích thước task | `s` | [102400, 1024000] | bits | [A] §6 (100–1000 kbit) |
| Cường độ tính toán | `c` | 1000 | cycles/bit | [A] §6 |
| Deadline ứng dụng | `τ_app` | 1.0 | s | [A] §6 (giới hạn trên; thực tế từ T1.1) |
| Tỉ lệ đến task | `λ` | 0.5 | tasks/s | [A]/[C] |
| Băng thông MEC | `B` | 1.0 | MHz | [A] §6 |
| Băng thông V2V | `B_v2v` | 0.8 | MHz | [A] §6 |
| Path loss exponent | `θ` | 4 | — | [A] §6 |
| Nhiễu | `W₀` | 3e-13 | W | [A] §6 |
| Channel gain @ `d₀` | `δ₀` | **cần hiệu chỉnh** | — | [A] §6 + T0.5 |
| Khoảng cách tham chiếu | `d₀` | 1 | m | MỞ RỘNG (T0.5) |
| Công suất phát | `p` | 0.2 | W | [A] §6 |
| Công suất chờ | `p_w` | 0.05 | W | [A] §6 |
| Hệ số điện dung | `κ` | 1e-27 | — | [B] Table I |
| Cycles mã hóa HE | `c_en` | 800 | cycles/bit | [A] §6 |
| Cycles mã hóa block | `q_en` | 90 | cycles/bit | [B] Table I |
| Khóa Paillier | — | 1024 | bits | [A] §3.3 |
| Frame length | `N_f` | **64** (hoặc 128) | bits | [B] Table I |
| Block length | `N_i` | biến quyết định | bits | [B] — T3.1 |
| Mật độ attacker | `ρ` | **1e-4** | xe/xe | [B] Table I |
| Số làn | `N_l` | 3 hoặc 8 | — | [A]=? / [B]=8 — cần chọn |
| Vận tốc tối đa | `v_max` | 25 | m/s | [A] §6 |
| Quy đổi năng lượng (local) | `λ^l` | 1.0 | cents/J | [A] §6 |
| Quy đổi năng lượng (AUX) | `λ^n` | 0.8 | cents/J | [A] §6 |
| Năng lượng tối đa | `E_max` | 0.4 | J | [B] Table I |
| Trọng số `U` | `ϖ₁` | 1e-5 (hiệu chỉnh) | — | [B] Table I |
| Discount | `γ` | 0.95–0.99 | — | MỞ RỘNG (T1.1) |
| Learning rate | — | 1e-3 | — | [C] |
| Batch size | — | 64 | — | [C] |
| Buffer | — | 100 000 | — | [C] |
| Warm-up | — | 2 000 | steps | MỞ RỘNG (T0.11) |

**Ba tham số bắt buộc phải quyết định mà bài gốc không cho:** `z` (khoảng cách RSU–lề, cần cho [A] Eq(6)), `δ₀` (cần hiệu chỉnh, T0.5), `N_l` (hai bài khác nhau).
