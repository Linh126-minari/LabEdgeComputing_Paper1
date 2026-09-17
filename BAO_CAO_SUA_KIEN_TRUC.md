# Báo cáo: Sửa kiến trúc theo đánh giá hai bài báo gốc

**Ngày:** 17/09/2026 · Thực hiện các đề xuất trong `DANH_GIA_KIEN_TRUC_THEO_BAI_BAO.md` · Test: **101 passed** · Hướng dẫn train: `HUONG_DAN_COLAB.md`

## 0. Đính chính so với bản đánh giá
Bản đánh giá §3.2 viết rằng khôi phục cấu trúc mục tiêu của [B] sẽ sửa được thiên lệch. **Nhận định này sai:**
- Trong [B], nhánh local nhận −ϖ₂T (luôn ≤ 0), còn nhánh offload nhận +ϖ₁U (luôn ≥ 0), nên [B] cũng nghiêng về offload.
- Vì vậy đã dùng **phương án (b): chi phí rủi ro rò rỉ**. U của [B] Eq(9) chỉ còn là metric báo cáo.

## 1. Thay đổi mô hình

| # | Nội dung | Trước | Sau |
|---|---|---|---|
| 3.1 | An ninh | Hai lớp chồng nhau (Paillier + mã khối, 890 cycles/bit) trên **mọi** nhánh; chi phí Paillier nhân theo N_i | **Phân vai theo đích.** MEC: Paillier 800 cycles/bit, ciphertext 2×, P_leak = 0. AUX: mã khối 90·(1 + 0.1·log₂(N_i/8)) cycles/bit, P_leak = 1 − (N_i/N_f)^{nᵉ} |
| 3.2 | Mục tiêu | r = −(C − ϖ₁U) | r = −C, với C = [A] Eq(22) + ϖ_s·(bit offload)·P_leak. Local và MEC-HE không bị tính rủi ro |
| 3.3 | CPU | xe 2 / MEC 1 / AUX 0.5 GHz (edge chậm hơn xe) | xe 1 GHz (DVFS) / MEC **3 GHz** ([B] F^e) / AUX 2 GHz |
| 3.3 | κ | 1e-28 (năng lượng local gần bằng 0) | **5e-28** ([B]: 1e-27) |
| 3.4 | Độ khẩn cấp ω | min(1/τ, 3), cần `urgency_cap` | **1/(1 + τ) ∈ (0,1)**; τ = 1 s cho ω = 0.5 (giá trị [A] dùng) |
| 3.5 | λⁿ | nhân năng lượng tính toán của AV | nhân **năng lượng của user** ở nhánh AUX ([A] Eq 20, 22); nhánh MEC dùng λʲ = λˡ |
| 3.6 | Mô hình thời gian | – | Giữ nguyên: mã hóa → local tuần tự; DVFS dạng đóng (đã có từ đợt trước) |
| 3.7 | Action | 3 chiều (N_i không được học) | **4 chiều [α_l, đích, β, N_i]**, 800 action; mask cố định N_i ở nhánh không dùng mã khối |
| 3.7 | State | 4 + 2(J+N) | **5 + 2(J+N)**: thêm nᵉ = số attacker trong tầm phát V2V ([B] giả định người gửi biết nᵉ khi chọn N_i) |
| 3.9 | Attacker | Kỳ vọng [B] Eq(8), hằng số | **Attacker thật** (11 xe/1 km², E[nᵉ] = 0.785 = kỳ vọng [B]); nᵉ thay đổi theo state. Trong kịch bản trace, attacker vẫn là synthetic |
| 3.9 | Vận tốc synthetic | U(15, 25) ([A]) | Gauss cắt cụt μ = 13, σ = 5 trên [2, 24] ([B] Eq 1); lấy mẫu lại mỗi episode |

## 2. Hiệu chỉnh (`scripts/calibrate_energy.py`, 300 state, thử mọi action)

| Cấu hình | Khả thi | Offload được | Offload tối ưu (khi offload được) | Đích AUX | N_i tối ưu |
|---|---|---|---|---|---|
| **Mặc định mới** | **92%** | 40% | **51%** | 95% | **8: 25, 64: 33** |
| CPU/κ cũ (xe 2 / MEC 1 / AUX 0.5, κ 1e-28) | 100% | 51% | **2%** (suy biến: luôn local) | – | – |
| κ = 1e-27 ([B]) | **78%** (22% state không thỏa E_max) | 57% | 60% | 94% | 8: 40, 64: 42 |
| Attacker kỳ vọng | 94% | 57% | 47% | 91% | **64: 70** (hằng số) |
| Giá rò rỉ = 0 | 94% | 57% | 47% | 91% | **8: 70** (hằng số) |

Kết luận: mỗi lựa chọn đều cần thiết để bài toán không suy biến.
- Thiếu CPU/κ mới thì offload vô nghĩa.
- Thiếu attacker thật thì N_i là hằng số.
- Thiếu giá rò rỉ thì luôn chọn N_i nhỏ nhất.

Test `test_calibrated_model_is_non_degenerate` canh giữ điều này.

**Nhận xét cho bài:** MEC hiếm khi được chọn (5–9%), vì Paillier tốn 800 cycles/bit và ciphertext phồng 2×. Đây chính là đánh đổi "MEC an toàn tuyệt đối nhưng đắt ↔ AUX rẻ nhưng rủi ro", nên đáng đưa vào thảo luận. Có thể quét `leakage_price` và mật độ attacker để vẽ vùng MEC trở nên đáng chọn.

**Khác biệt sim-to-real (vận tốc synthetic theo [B]):**
- vận tốc: KS 0.728 (13 so với 4.4 m/s);
- thời gian lưu trú MEC: KS 0.553 (trung vị 31 so với 86 s).

## 3. Thay đổi code

| File | Thay đổi |
|---|---|
| `src/env/core.py` | Viết lại `_execute_task` và thêm `_offload_branch` (phân vai mã hóa, λ theo nhánh, chi phí rủi ro); `_price_urgency`; `_compute_reward`; action 4D + mask; state + nᵉ (`attackers_in_range`); `_sample_velocity` ([B]); attacker luôn synthetic; `evaluate_action`/`snapshot_resources` (dry-run); metrics mới: `avg_cost_security`, `offload_ratio`, `aux_share`, `mec_share`, `avg_block_length_aux` |
| `src/env/multi_agent.py` | State + nᵉ, ω mới |
| `src/baselines/heuristic.py`, `cmso.py`, `paper_a_alg1.py` | Theo `action_space.nvec`. Greedy và CMSO đánh giá bằng `env.evaluate_action` (cùng mô hình chi phí, không chép công thức). AllAUX/CMSO chọn N_i theo [B] Eq(39) |
| `train.py` | `create_env_config` lấy default từ `EnvironmentConfig` (không hardcode lần hai); assert state 5 + 2(J+N), action 3/4 chiều; log `avg_cost_security`, `offload_ratio`, `avg_block_length_aux` |
| `configs/params_master.yaml` | CPU, κ, attacker, vận tốc, `leakage_price`, `price_energy_mec`, `block_cipher_round_factor`. Bỏ `urgency_cap`, `varpi_1`, `encryption_total_cycles_per_bit` |
| `configs/env/*.yaml`, `configs/security/paillier_params.yaml`, `configs/agent/*.yaml` | Khóa mới (`attack.model`, `constraints.leakage_price`, `encryption.scheme`, `use_4d_action`, …) |
| `scripts/calibrate_energy.py` | Viết lại: đo khả thi, tỷ lệ offload, đích, phân bố N_i |
| `scripts/sim2real_study.py` | objective = `avg_cost` (bao gồm rủi ro); vận tốc synthetic theo [B] |
| `scripts/run_experiments.py` | Thời gian đo lại (DQN 5.1/7.3 s, MADDPG 22 s mỗi episode); `--speed-factor` |
| `scripts/plot_convergence.py` (mới) | Đường hội tụ để chọn số episode |
| `notebooks/colab_train.ipynb` (mới), `HUONG_DAN_COLAB.md` (mới) | Quy trình Colab |
| `notebooks/LabEdgeComputing.tex` | Viết lại State/Action, DVFS, Objective, "Role-Based Privacy Protection"; số KS mới |
| `requirements.txt` | Thêm pandas, pyarrow, scipy, pyyaml, cvxpy |
| Đã xóa | `scripts/calibrate_varpi.py` (hiệu chỉnh ϖ₁, không còn dùng) |
| Test | `test_formulas` (rủi ro, ω), `test_energy_dvfs` (không suy biến), `test_params_master` (khóa mới), `test_phase1`/`test_baselines`/`test_phase4`/`test_env_invariants` (action 4D, state 13 chiều) |

`outputs/eval_raw_results.csv` và `eval_summary.csv` bị ghi đè khi smoke-test `scripts/evaluate.py`. Đây là số liệu của mô hình cũ, vốn đã không hợp lệ.

## 4. Còn lại / lưu ý
- **Toàn bộ số liệu cũ** (Abstract, Results trong `.tex`, `TONG_HOP_DANH_GIA_3_THUAT_TOAN.md`) thuộc mô hình cũ và phải thay sau khi train lại.
- `total_reward` trong log train vẫn khác thước đo giữa LagrangianDQN (không trừ penalty) và baseline (có trừ). **Chỉ so sánh bằng `sim2real_study.py`.**
- DuelingDQN vẫn là "DDRP-style single-agent" (điều khiển user 0), không phải DDRP tập trung như [A] (§3.8). Cần ghi rõ trong bài.
- Công suất phát p vẫn cố định (0.2 W); [B] tối ưu p.
- Các script Phase 3 cũ (`compare_crypto.py`, `analyze_discretization.py`, `reproduce_papers.py`) chưa được kiểm tra lại với mô hình mới.
- Code chưa commit. Colab cần `git push` trước (xem `HUONG_DAN_COLAB.md` §0).
