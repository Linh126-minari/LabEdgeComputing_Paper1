# Báo cáo hoàn thành Phase 3 — Novel Contributions & Advanced Extensions

**Ngày:** 29/07/2026 · Theo `KEHOACH_DUAN_VEC.md` · Tiếp nối `BAO_CAO_PHASE2.md`

## Quyết định & Nội dung hoàn thành Phase 3

Phase 3 tập trung xây dựng các đóng góp mới của bài báo (Novel Contributions), mở rộng action space 4 chiều, mô hình hóa kẻ tấn công hiện thực hóa theo không gian, phát triển hạ tầng Multi-Agent VEC (MARL), so sánh chi tiết HE vs AES và phân tích sai số rời rạc hóa.

---

## Việc đã làm theo từng task

- **T3.1 — Đưa block length $N_i$ vào Action Space & Khép kín vòng Trade-off.**
  - Mở rộng action space thành 4 chiều `MultiDiscrete[10, 2, 10, 4]` với $N_i \in \{8, 16, 32, 64\}$ bits.
  - Gắn chi phí mã hóa phụ thuộc $N_i$:
    $$c_{en}(N_i) = c_{en\_base} \cdot (1 + \kappa_N \cdot \log_2(N_i / N_{min}))$$
    khép kín vòng lặp đánh đổi giữa độ an toàn $S = (N_i / N_f)^{n_e}$ và chi phí năng lượng/tính toán mã hóa.

- **T3.2 — Attacker hiện thực hóa (Realized Attacker Model).**
  - Thêm cấu hình `attacker_model: "expected" | "realized"`.
  - Trong chế độ `"realized"`, đếm trực tiếp số lượng xe tấn công `AttackVehicle` thực tế nằm trong bán kính phát $d_u$ của xe gửi, phản ánh biến thiên ngẫu nhiên theo không gian và thời gian.

- **T3.3 — Multi-Agent VEC Environment & MARL (IDQN).**
  - Xây dựng `src/env/multi_agent.py` (`MultiAgentVehicularEdgeEnv`) hỗ trợ chuẩn PettingZoo ParallelEnv cho đa xe người dùng.
  - Xây dựng `ParameterSharedIDQNAgent` (`src/agents/marl.py`) thực thi thuật toán Independent DQN chia sẻ trọng số mạng Q.
  - Hỗ trợ 3 dạng reward: Cá nhân ($r_u$), Max-Min Fairness ($\min_{u'} r_{u'}$ theo Paper [B]), và Kết hợp $(1-\xi) r_u + \xi \min_{u'} r_{u'}$.

- **T3.4 — So sánh HE (Paillier) vs AES đầy đủ (`scripts/compare_crypto.py`).**
  - Thêm thuộc tính `crypto_scheme: "paillier" | "aes" | "none"` và hệ số phồng ciphertext ($2.0\times$ cho Paillier vs $1.0\times$ cho AES).
  - Xây dựng kịch bản `scripts/compare_crypto.py` đánh giá ảnh hưởng của phồng dữ liệu truyền V2I đến trễ và năng lượng, tìm ngưỡng kích thước task mà ưu thế của HE bị đảo ngược.

- **T3.5 — Phân tích sai số rời rạc hóa Action Space (`scripts/analyze_discretization.py`).**
  - Xây dựng kịch bản `scripts/analyze_discretization.py` đo lường sai số rời rạc hóa giữa các mức $\alpha_{levels} \in \{5, 10, 20, 50\}$ so với nghiệm liên tục của CMSO (Paper [A] Alg 3).

---

## Kiểm thử & Xác nhận

- Toàn bộ **82/82 unit & invariant tests** đã vượt qua thành công:
  ```bash
  python -m pytest tests/ -v
  ```
- Các script thực nghiệm Phase 3 đã chạy thành công và lưu kết quả CSV vào thư mục `outputs/`.

---

## Lệnh thực thi cho người dùng

```bash
# 1. Chạy bộ test tự động (82 tests passed)
python -m pytest tests/ -v

# 2. Chạy thử nghiệm so sánh mã hóa HE vs AES
python scripts/compare_crypto.py

# 3. Chạy phân tích sai số rời rạc hóa
python scripts/analyze_discretization.py
```
