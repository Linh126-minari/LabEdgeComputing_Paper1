# Báo cáo hoàn thành Phase 1 — Sửa formulation

**Ngày:** 26/07/2026 · Theo `KEHOACH_DUAN_VEC.md` · Tiếp nối `BAO_CAO_PHASE0.md`

## Quyết định đã chốt (xác nhận 26/07/2026)

| Quyết định | Lựa chọn |
|---|---|
| T1.2 — Hàm mục tiêu | **PA1**: kinh tế [A] Eq(22) + bảo mật [B]; đơn vị duy nhất (cents); deadline/energy là ràng buộc |
| T1.4 — Ngữ nghĩa MDP | **PA-A**: semi-MDP event-driven, discount theo thời gian γ^(Δt/dt) |
| T1.5 — Đa user | **Mức 1 + Mức 2**: users[1:] chạy Algorithm 1 [A], chiếm tài nguyên thật |
| T1.7 — N_f | **Giữ 64** theo [B]; hiệu chỉnh (d_i) để n^e vào vùng có nghĩa |

## Việc đã làm theo từng task

- **T1.1 — Deadline động.** `MECServer.road_segment_length()` ([A] Eq(6), thêm `curb_distance` z=20m); `residence_time_mec()` tổng quát hóa [A] Eq(7) sang 2D (thời gian ra khỏi đường tròn phủ sóng theo hướng chuyển động — trùng [A] trong trường hợp 1D); `v2v_link_time()` từ vận tốc **tương đối** (MỞ RỘNG so với [A]/[B] — đóng góp nhỏ nhưng sạch, đã ghi vào .tex); `τ_u = clip(min(τ_app, t^w_MEC, t^w_V2V), 0.05, τ_app)`. State dùng τ_eff/ω động (hết chiều chết). AUX giờ di chuyển bền vững bằng `LinearMobility` riêng (bản cũ đổi hướng ngẫu nhiên mỗi step khiến link-lifetime vô nghĩa). Metric mới `link_loss_failure_rate` phân biệt vỡ deadline do mất liên kết vs do ứng dụng. γ mặc định 0.9 → **0.95** (sweep {0.9, 0.95, 0.99} khi chạy thực nghiệm). **MỞ RỘNG bắt buộc phát sinh:** chặn ω = 1/τ tại `urgency_cap = 3` trong định giá `exp(ω)` — [A] chỉ xét τ = 1s; với τ→0.05s thì exp(20) ≈ 5·10⁸ làm giá phát nổ.
- **T1.2 — Hàm mục tiêu duy nhất (PA1).** `C_u = λ^l·(E_local + E_enc + E_tx + E_wait) + λ^n·E_AV + g_j + g_n` (cents, [A] Eq(22); λ^l=1, λ^n=0.8 cents/J [A] §6); reward = −(C_u − ϖ₁·U). Bỏ hoàn toàn weighted-sum 5 trọng số (xóa khỏi config + params_master). E_max = 0.4 J ([B] C3). λ_p tạm hạ 100 → **5** (dải [2,10] theo kế hoạch). Log **từng thành phần** reward: `info["reward_components"]`, các cột năng lượng/chi phí tách riêng, và `outputs/.../reward_components.csv` mỗi episode.
- **T1.3 — Constrained RL.** `src/agents/constrained_policy.py::LagrangianDRLAgent` (RCPO-style): reward định hình r − λ_T·viol_T − λ_E·viol_E; dual ascent mỗi K=10 episode, lr_λ=0.05 ≪ lr_policy, δ_T=0.05, chặn λ_max=100; `lambda_history` lưu ra `lambda_history.json` (hình hội tụ multiplier cho bài). Env thêm `penalty_mode`: `"external"` (Lagrangian — env không trừ penalty, trả cờ vi phạm) vs `"fixed"` (baseline λ_p cố định — dùng cho so sánh bắt buộc λ_p ∈ {1,10,100}). Config `configs/agent/lagrangian_dqn.yaml`; chạy: `python train.py agent=lagrangian_dqn`.
- **T1.4 — Semi-MDP.** `step()` = một QUYẾT ĐỊNH; sau khi thực thi, env tiến thời gian tới task kế tiếp ngay trong step — next_state **luôn** chứa task, thời gian chờ tính vào `info["elapsed_time"]` của transition hiện tại; **không còn** transition rỗng/`{"error": ...}` nào vào buffer; `step_count` đếm quyết định (episode so sánh được giữa các run); Gymnasium API chuẩn (`reset(seed=)` → `(obs, info)`, `step` → 5-tuple terminated/truncated). `ReplayBuffer` lưu thêm `elapsed_time` + `next_mask`; `train_step` dùng `γ_eff = γ^(elapsed/dt)`.
- **T1.5 — Đa user (Mức 2 + Mức 1).** Tách `_execute_task(user, ...)` không phụ thuộc `users[0]`; `users[1:]` chạy Algorithm 1 của [A] (MEC nếu trong vùng phủ, ngược lại AUX rảnh giá g_n nhỏ nhất, α_l=0.5) **mỗi slot** và thực sự chiếm CPU/BW từ pool chung. Ràng buộc thực thi: Σβ ≤ 1 ([A] 23j — cấp phát là % pool còn lại + từ chối khi dưới sàn 1%, đếm `rejected_resource_count`); mỗi AUX ≤ 1 user/slot ([A] §3.4 — điều kiện `current_load == 0`). Mức 1 (Markov-modulated load) là fallback qua config khi tắt Mức 2. Metrics mới: `avg_eps_n`, `mec_utilization`, `rejected_resource_count`, `background_task_count`.
- **T1.6 — Action masking.** `env.action_mask()` (200, C-order khớp `np.ravel_multi_index` của agent): α_l=1 luôn khả thi; nhánh MEC cần vùng phủ + tài nguyên; nhánh AUX cần trong tầm + AUX rảnh + tài nguyên. `select_action`: Q[~mask] = −inf, explore chỉ trong tập khả thi; `train_step` mask cả next_q (Double lẫn thuần) bằng `next_mask` lưu trong buffer. Ablation có/không mask chạy được ngay bằng cách không truyền mask (giữ nguyên interface).
- **T1.7 — Hiệu chỉnh bảo mật.** `scripts/calibrate_security.py`: bảng vùng suy biến của nghiệm đóng [B] Eq(39) (**chỉ có nghĩa trên n^e ∈ [0.52, 0.84]** — với chính tham số Table I của [B] thì n^e ≈ 0.087 → suy biến; ở tầm 200m thì bão hòa), sweep (ρ, d), heatmap PNG (`--plot`), kiểm tra bộ vận hành. Chốt: `security_range_v2i = 200 m` (n^e=1.34, p_att=0.61), `security_range_v2v = 150 m` (n^e=0.75, p_att=0.41) — p_att có phương sai giữa hai nhánh, trung bình ≈ 0.51 ∈ [0.1,0.6]. Phát hiện này đã viết thành tiểu mục trong .tex (Remarks) — động lực trực tiếp cho T3.1.

## Đồng bộ tài liệu

`.tex`: thêm mobility-induced deadline (Eq. effective_deadline), semi-MDP + γ_eff, action masking, objective mới (Eq. objective/total_cost/lagrangian — thay hẳn weighted-sum), multi-user contention, tiểu mục vùng suy biến Eq(39). `params_master.yaml`: thêm 8 tham số mới (z, d_i^V2I/V2V, τ_min, ω_max, λ_p, λ_pE; xóa 5 trọng số cũ); `params_table.tex` cập nhật khớp; test đồng bộ code↔yaml mở rộng tương ứng.

## Test

`tests/test_phase1.py` — 20 test bám DoD từng task: hình học lưu trú tính tay, τ_eff/ω có phương sai, tương quan âm với vận tốc, reward khớp tổng thành phần, all-local cost = λ^l·E, dual ascent tăng/giảm đúng chiều, elapsed_time là bội dt + step_count đếm quyết định, ε_n/U_j có phương sai + tăng theo num_users, AUX ≤ 1 user, Σβ ≤ 1, mask đúng thứ tự flatten + agent không bao giờ chọn action bị mask, vùng suy biến Eq(39), điểm vận hành trong vùng có nghĩa. Các test cũ đã chuyển sang Gymnasium API.

## Việc cần bạn chạy

```bash
python -m pytest tests/ -v                     # kỳ vọng ~65 passed
python scripts/calibrate_security.py           # bảng T1.7 (thêm --plot nếu có matplotlib)
python scripts/gen_param_table.py              # sinh lại bảng tham số
python train.py general.num_episodes=20        # smoke run agent thường (penalty fixed)
python train.py agent=lagrangian_dqn general.num_episodes=20   # smoke run Lagrangian
```

## Lưu ý cho Phase 2

1. **Mọi số liệu cũ đều vô hiệu** — reward đơn vị mới (cents), env semi-MDP: baseline (T2.x) phải chạy trên formulation này.
2. So sánh Lagrangian vs λ_p cố định {1,10,100} (T1.3 DoD) cần evaluation harness của T2.6 (multi-seed).
3. Với κ=1e-27, f_u=2GHz: E_local của task lớn (~2 J) vượt E_max=0.4 J → ràng buộc năng lượng ép offload mạnh; nếu muốn all-local là lựa chọn khả thi cho task lớn, cân nhắc κ=1e-28 (phổ biến trong văn liệu) — quyết định ở T2.8 khi tái lập xu hướng của [A]/[B].
4. `test_generated_table_file_is_up_to_date` sẽ nhắc chạy lại `gen_param_table.py` nếu bảng lệch.
