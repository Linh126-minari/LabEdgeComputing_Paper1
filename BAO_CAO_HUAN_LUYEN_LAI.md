# Báo cáo: Sửa ràng buộc năng lượng, MADDPG thật, giảm thời gian huấn luyện

> **17/09/2026:** Bảng hiệu chỉnh §1.3 (κ = 1e-28) và bảng thời gian §4 đã **lỗi thời**, vì mô hình đã sửa theo `BAO_CAO_SUA_KIEN_TRUC.md` (κ = 5e-28, CPU mới, action 4D). Quy trình train hiện hành: `HUONG_DAN_COLAB.md`.

**Ngày:** 16/09/2026 · Tiếp nối `BAO_CAO_PHASE4_RIOBUS.md` · Test: **101 passed**

## 1. Ràng buộc năng lượng E ≤ E_max

### 1.1 Nguyên nhân viol_E ≈ 100%
Kiểm tra lại PDF gốc:
- [B] Table I: κ = 1e-27, E_i = 0.4 J, **F_i^loc = 2.0 GHz là tần số tối đa**. [B] tối ưu f^c và f^en theo DVFS (ràng buộc C4: 0 ≤ f ≤ F^loc).
- [A] §6: xe 0.5 GHz, Paillier 800 cycles/bit.

Code cũ **luôn chạy CPU xe ở 2 GHz** và mã hóa 2 lớp tốn 890 cycles/bit. Với task 1 Mbit:
- tính local tốn ≈ 4 J;
- mã hóa tốn ≈ 3.6 J;

cả hai đều vượt xa E_max = 0.4 J. Kết quả là **0% state có action nào thỏa ràng buộc** (đo ở §1.3). Đây là lý do λ_E của LagrangianDQN chỉ tăng mãi.

### 1.2 Sửa: DVFS dạng đóng (`src/env/core.py::_dvfs_frequencies`)
CPU xe user chạy **tuần tự**: mã hóa trước, rồi tính local. Các nhánh remote chạy song song sau khi mã hóa xong.

```
min κ(C_en·f_en² + C_l·f_l²)  s.t.  C_en/f_en + C_l/f_l ≤ τ,   C_en/f_en + T_remote ≤ τ,   f ≤ F_loc
```
Nghiệm KKT:
- f_en = f_l = (C_en + C_l)/τ;
- nếu phần mã hóa không kịp cho upload: f_en = C_en/(τ − T_remote), f_l = C_l/(τ − t_en);
- cắt vào khoảng [0.1·F, F]. Nếu không khả thi thì chạy F và ghi nhận vi phạm deadline.

Sửa kèm: trước đây nhánh local và nhánh fallback-local được tính như **chạy song song trên 2 CPU**; giờ tổng cycles chạy trên một CPU.

Cờ cấu hình: `dvfs_enabled` (mặc định `True`), `dvfs_min_fraction` = 0.1.

### 1.3 Hiệu chỉnh (`scripts/calibrate_energy.py`, 300 state, thử mọi action)

| κ | DVFS | E_max | Có action khả thi | Có action kịp deadline | Offload MEC vi phạm E | Min-cost vi phạm E | Action tốt nhất là offload |
|---|---|---|---|---|---|---|---|
| 1e-27 | ✗ | 0.4 | **0%** | 99% | 100% | 100% | – |
| 1e-27 | ✓ | 0.4 | 67% (task lớn: 5%) | 98% | 76% | 33% | 21% |
| 1e-27 | ✓ | 1.0 | 94% | 98% | – | 6% | 23% |
| 1e-28 | ✗ | 0.4 | 95% | 98% | 56% | 13% | 22% |
| **1e-28** | **✓** | **0.4** | **98%** | **98%** | **32%** | 4% | **23%** |
| 1e-28 | ✓ | 0.2 | 98% | 98% | 50% | 11% | 23% |

**Chọn: κ = 1e-28, DVFS bật, E_max = 0.4 J** (giữ đúng giá trị của [B]).
- Ràng buộc năng lượng không làm mất nghiệm: 98% state khả thi, bằng tỷ lệ chỉ xét deadline.
- Ràng buộc vẫn có tác dụng: 32% action offload MEC vi phạm.
- Bài toán không suy biến: 23% trường hợp action tốt nhất là offload.

κ = 1e-28 là giá trị phổ biến trong văn liệu MEC; trong bài cần ghi rõ "calibrated, [B] uses 1e-27". Nếu muốn ràng buộc chặt hơn thì dùng E_max = 0.2. Mọi giá trị nằm ở `configs/params_master.yaml`.

**Lưu ý khi đọc kết quả:** vì DVFS chọn tần số thấp nhất còn kịp deadline, `avg_delay` sẽ tiến sát τ_eff. Delay trung bình không còn là thước đo "càng nhỏ càng tốt"; nên báo cáo tỷ lệ vi phạm deadline thay vì delay.

Test: `tests/test_energy_dvfs.py` (5 test, gồm cả một test hiệu chỉnh tự động).

## 2. MADDPG thật (`src/agents/maddpg.py`)

"MADDPG" cũ chỉ là `DRLAgent` (Dueling Double DQN) với cấu hình khác. Bản mới theo Lowe et al. (NeurIPS 2017):
- **Mỗi user là một agent.** `MultiAgentVehicularEdgeEnv` được viết lại theo sự kiện: mỗi bước, mọi user đang có task quyết định đồng thời, rồi thời gian tiến tới khi có user có task tiếp. Chiết khấu dùng γ^(Δt/dt), giống DQN. Env ghi `task_history`, nên `get_metrics()` hoạt động.
- **Actor phân tán:** π(o_i, id_i), mỗi chiều action một đầu logits, dùng **Gumbel-Softmax straight-through**.
- **Critic tập trung:** Q(o_1..o_N, a_1..a_N, id_i). Tham số chia sẻ giữa các agent đồng nhất.
- **Action masking theo chiều:** target không khả thi bị chặn; nếu không target nào khả thi thì chỉ cho phép α_l = 1.
- **Penalty cố định**, giống DuelingDQN. `reward_mode`: individual | max_min ([B]) | combined.
- Background user chạy Algorithm 1 bị tắt, vì giờ mọi user đều là agent.

**Lưu ý khi so sánh:**
- Metrics của MADDPG là trung bình trên task của **cả 10 user học**.
- Metrics của DQN/Lagrangian là trên task của **user 0**, với 9 user nền chạy Algorithm 1 [A].

Đây là hai kịch bản khác nhau. Trong bài cần ghi rõ, hoặc đánh giá thêm DQN trong env đa agent (mỗi user dùng policy DQN).

Tích hợp: `train.py` (hàm `build_agent`), `configs/agent/maddpg.yaml`. Test: `tests/test_maddpg.py` (4 test).

## 3. Vấn đề khác phát hiện (chưa sửa, cần bạn quyết định)
- **N_i không nằm trong action space.** Cờ `use_4d_action` được khai báo nhưng không dùng ở đâu; action luôn là `MultiDiscrete[10, 2, 10]`, và `train.py` còn assert đúng 3 chiều. Vì vậy "dynamic security level" (T3.1) chưa bao giờ được học. MADDPG đã hỗ trợ sẵn 4 chiều; bật lên cần sửa `action_space`, assert và tính lại các bảng.
- **Lỗi so sánh reward cũ** (LagrangianDQN log reward chưa trừ penalty) vẫn còn trong `train.py`. Vì vậy **không dùng `total_reward` để so sánh thuật toán**; dùng `objective_cents` + viol_T + viol_E từ `scripts/sim2real_study.py`.

## 4. Giảm thời gian huấn luyện

**Làm rõ:** "54 h + 17.5 h" là **thời lượng giao thông thật trong dữ liệu** (108 + 35 cửa sổ × 30 phút), không phải thời gian train.

### 4.1 Đo thực tế (1000 bước/episode, 1 luồng CPU)

| Run | s/episode | 500 episode (eval 10×5) |
|---|---|---|
| DQN / Lagrangian, synthetic | 3.5 | ~35 phút |
| DQN / Lagrangian, trace Rio | 5.8 | ~57 phút |
| MADDPG, synthetic / trace | 19.5 / 20.0 | ~3.1 giờ |
| MADDPG, `train_freq=4` | 5.5 | ~55 phút |

Log cũ ghi ~12 s/episode vì chạy trên máy Windows khác. Toàn bộ lưới 3 thuật toán × 2 miền × 3 seed chạy tuần tự hết **~29 giờ**.

Profile DQN: ~60% thời gian là cập nhật mạng (forward/backward/Adam), ~25% là di chuyển xe trong env. GPU **không giúp**: mạng MLP quá nhỏ, và nút thắt nằm ở vòng lặp env Python.

### 4.2 Các phương pháp (đã cài sẵn các mục ✓)

| # | Phương pháp | Tác động | Ảnh hưởng khoa học |
|---|---|---|---|
| ✓1 | **Chạy song song nhiều tiến trình** (`scripts/run_experiments.py`, 1 luồng torch/tiến trình, tối đa 8) | 29 h → **3.4 h** | Không ảnh hưởng |
| ✓2 | Giảm tần suất eval trong train: 10×5 → 25×3 episode (mặc định của launcher) | −13% | Không ảnh hưởng (eval cuối làm riêng bằng `sim2real_study.py`) |
| ✓3 | MADDPG `train_freq=4` (cập nhật mạng mỗi 4 bước, như SB3) | MADDPG ~3.5× nhanh hơn → cả lưới **1.9 h** | Nhỏ; nên chạy thử 1 seed để so với `train_freq=1` |
| 4 | Giảm số episode sau khi chạy thử 1 seed. Log cũ: eval reward vào vùng ±2% giá trị cuối ngay từ episode ~10 (nhưng reward cũ bị penalty chi phối, nên phải kiểm tra lại với mô hình mới) | 500 → 200 episode: cả lưới **~0.7 h** | Phải có đường hội tụ minh chứng |
| 5 | Chỉ chạy ma trận sim-to-real 2×2 cho thuật toán đề xuất + 1 baseline; so sánh 3 thuật toán chỉ trên miền Rio | Bớt ~1/3 số run | Không ảnh hưởng tới kết luận chính |
| ✓6 | Launcher bỏ qua run đã có `agent_model.pth` | Không mất công khi bị ngắt | – |
| ✓7 | Env chỉ nạp các cột trace cần thiết, id dạng category (267 → 102 MB) | Chạy được nhiều tiến trình hơn | – |

Không khuyến nghị giảm `episode_length`, vì làm thay đổi horizon và ngữ nghĩa bài toán.

### 4.3 Quy trình đề xuất
```bash
# B1. Chạy thử 1 seed, xem đường hội tụ để chọn số episode (~1 h)
python scripts/run_experiments.py --seeds 0 --maddpg-train-freq 4 --out-root outputs/pilot
python scripts/plot_log_comparisons.py ...   # hoặc đọc outputs/pilot/*/*/seed0/training_history.json

# B2. Chạy đủ lưới với số episode đã chọn (vd 200 → ~0.7 h; 500 → ~1.9 h)
python scripts/run_experiments.py --dry-run --episodes 200 --maddpg-train-freq 4
python scripts/run_experiments.py --episodes 200 --maddpg-train-freq 4

# B3. Đánh giá chung (objective, viol_T, viol_E) trên synthetic và riobus_test, không train lại
for a in lagrangian_dqn dueling_dqn maddpg; do
  python scripts/sim2real_study.py --runs-dir outputs/runs --agent $a --seeds 0 1 2
done
```
Khi chạy nên đóng các ứng dụng nặng: 8 tiến trình dùng khoảng 5–6 GB RAM.

## 5. File thay đổi
- **Mới:**
  - `src/agents/maddpg.py`
  - `scripts/calibrate_energy.py`
  - `scripts/run_experiments.py`
  - `tests/test_energy_dvfs.py`
  - `tests/test_maddpg.py`
- **Sửa:**
  - `src/env/core.py`: DVFS, CPU tuần tự, nạp trace tiết kiệm RAM
  - `src/env/multi_agent.py`: theo sự kiện, `task_history`
  - `src/security/overhead_models.py`: κ mặc định
  - `train.py`: `build_agent`, MADDPG, `eval_episodes`, seed cho torch
  - `scripts/sim2real_study.py`: `--runs-dir`
  - `configs/params_master.yaml`: κ, nguồn F^loc
  - `configs/agent/maddpg.yaml`, `configs/config.yaml`
  - `notebooks/params_table.tex`: sinh lại
  - `notebooks/LabEdgeComputing.tex`: mô tả MADDPG, comment cảnh báo
  - `notebooks/TONG_HOP_DANH_GIA_3_THUAT_TOAN.md`: cảnh báo kết quả cũ không hợp lệ
