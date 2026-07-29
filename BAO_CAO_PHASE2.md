# Báo cáo hoàn thành Phase 2 — Baseline & Evaluation Framework

**Ngày:** 29/07/2026 · Theo `KEHOACH_DUAN_VEC.md` · Tiếp nối `BAO_CAO_PHASE1.md`

## Quyết định & Nội dung hoàn thành Phase 2

Phase 2 tập trung xây dựng toàn bộ các thuật toán baseline so sánh (cả heuristic lẫn thuật toán gốc từ bài báo [A] và [B]), bổ sung hạ tầng ablation cho DRL agent, và phát triển framework đánh giá đa thuật toán/đa seed kèm bộ kiểm tra xu hướng định tính.

---

## Việc đã làm theo từng task

- **T2.1 — Baseline Heuristic policies (`src/baselines/heuristic.py`).**
  - Thực thi lớp cơ sở `BasePolicy` trong `src/baselines/base.py` định nghĩa chuẩn giao diện `select_action(state, env, training)`.
  - `AllLocalPolicy`: Cố định $\alpha_l = 1.0$, tính toán hoàn toàn cục bộ.
  - `AllMECPolicy`: Offload toàn bộ sang MEC server trong vùng phủ sóng ($\alpha_l = 0.0, \text{target} = \text{MEC}$).
  - `AllAUXPolicy`: Offload toàn bộ sang xe phụ rảnh rỗi trong tầm V2V ($\alpha_l = 0.0, \text{target} = \text{AUX}$).
  - `RandomPolicy`: Lấy mẫu ngẫu nhiên đồng đều trong tập các hành động hợp lệ tuân thủ `env.action_mask()`.
  - `GreedyPolicy`: Đánh giá các hành động khả thi để tối thiểu hóa chi phí tức thời $C_u - \varpi_1 U$.

- **T2.2 — Paper [A] Algorithm 1 User Selection (`src/baselines/paper_a_alg1.py`).**
  - Thực thi đúng thuật toán chọn node offload của Paper [A]: Ưu tiên MEC nếu trong vùng phủ sóng $\rightarrow$ Xe phụ AUX rảnh rỗi có đơn giá $g_n$ nhỏ nhất $\rightarrow$ Tính toán cục bộ.

- **T2.3 — Paper [A] CMSO Algorithm 3 (`src/baselines/cmso.py`).**
  - Thực thi thuật toán tối ưu hóa tỷ lệ phân chia cục bộ $\alpha_l^* \in [\alpha_l^{min}, \alpha_l^{max}]$ theo bài báo [A]. Sử dụng thư viện `cvxpy` để giải bài toán tối ưu lồi 1D (kèm cơ chế 1D grid search fallback an toàn khi `cvxpy` không có sẵn hoặc gặp lỗi hội tụ).

- **T2.4 — Paper [B] Eq(39) Closed-Form Security (`src/baselines/gbd.py`).**
  - Thực thi công thức dạng đóng xác định độ dài khối tối ưu $N_i^* = \min(N_f, \lfloor 1 / \sqrt{n_i^e} \rfloor)$ từ Paper [B] cho baseline mã hóa khối.

- **T2.5 — DRL Baselines Ablation Support (`src/agents/`).**
  - Bổ sung `StandardDQNNetwork` (mạng Q MLP thuần không dueling) trong `src/agents/networks.py`.
  - Cập nhật `DRLAgent` trong `src/agents/policy.py` với tham số `use_dueling: bool = True` (gọi `StandardDQNNetwork` khi `False`). Cho phép đánh giá ablation 2x2 đủ 4 phương án: DQN, Double DQN, Dueling DQN, Dueling Double DQN.

- **T2.6 — Metrics Collector & Evaluation Harness (`src/utils/metrics.py` & `scripts/evaluate.py`).**
  - `MetricsCollector`: Thu thập các chỉ số episode gồm chi phí (cents), dung lượng an toàn $U$, trễ thực thi (trung bình & p95), năng lượng tiêu thụ, tỷ lệ vi phạm deadline/năng lượng, tỷ lệ mất liên lạc V2X và chỉ số công bằng Jain's fairness index.
  - `scripts/evaluate.py`: Kịch bản đánh giá chạy đa seed, đa thuật toán trên cùng quỹ đạo di chuyển xe và tiến trình sinh task; xuất dữ liệu thô ra `outputs/eval_raw_results.csv` và bảng tổng hợp trung bình kèm khoảng tin cậy 95% ra `outputs/eval_summary.csv`.

- **T2.7 & T2.8 — Scripts Ablation & Paper Trend Validation (`scripts/reproduce_papers.py`).**
  - Thực thi kiểm tra xu hướng định tính từ bài báo [A] và [B] sử dụng hệ số tương quan Spearman (`scipy.stats.spearmanr`):
    1. Kích thước task $\uparrow \implies$ Chi phí $\uparrow$ (Spearman $\rho = 1.0000, p < 0.001$).
    2. Mật độ kẻ tấn công $\rho_{att} \uparrow \implies$ Dung lượng an toàn $U \downarrow$ (Spearman $\rho = -1.0000, p < 0.001$).

---

## Đồng bộ & Kiểm thử

- **Test Suite (`tests/test_baselines.py` + toàn bộ hệ thống):**
  - Thêm 7 unit tests kiểm tra tính hợp lệ của các baseline, ràng buộc action mask, ưu thế chi phí của Greedy so với Random, khoảng nghiệm CMSO và kết quả dạng đóng GBD Eq(39).
  - Toàn bộ **77/77 tests** đã vượt qua thành công:
    ```bash
    python -m pytest tests/ -v
    ```

---

## Lệnh thực thi kiểm tra cho người dùng

```bash
# 1. Chạy bộ unit/invariant test đầy đủ (77 tests passed)
python -m pytest tests/ -v

# 2. Chạy thử nghiệm khung đánh giá đa thuật toán / đa seed
python scripts/evaluate.py --episodes 10 --seeds 3

# 3. Chạy kiểm tra xu hướng định tính (Paper trends validation)
python scripts/reproduce_papers.py --quick
```

---

## Sẵn sàng cho Phase 3

Phase 2 đã hoàn thành 100% mục tiêu. Tiếp theo dự án sẵn sàng chuyển sang **Phase 3 — Joint Security & Resource Optimization / Security Enhancement Module**:
1. Tối ưu hóa độ dài khối động $N_i^*$ (GBD / Generalized Benders Decomposition) kết hợp mã hóa hai lớp Paillier + Block Cipher.
2. Tích hợp thuật toán DRL đề xuất giải quyết bài toán phối hợp Offloading + Allocating + Security Strength.
