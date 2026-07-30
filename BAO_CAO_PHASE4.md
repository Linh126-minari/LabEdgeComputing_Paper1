# Báo cáo hoàn thành Phase 4 — Dataset & Sim-to-Real Integration

**Ngày:** 30/07/2026 · Theo `KEHOACH_DUAN_VEC.md` · Tiếp nối `BAO_CAO_PHASE3.md`

## Quyết định & Nội dung hoàn thành Phase 4

Phase 4 tập trung tích hợp tập dữ liệu hành trình GPS thực tế của xe buýt Rio de Janeiro (`Rio bus dataset`), khắc phục các lỗi dữ liệu hiện có, xây dựng mô hình di chuyển dựa trên trace thực (`TraceMobility`), và thực hiện nghiên cứu so sánh định lượng khoảng cách chuyển giao giữa mô phỏng và thực tế (Sim-to-Real Gap).

---

## Việc đã làm theo từng task

- **T4.1 — Pipeline Tiền xử lý Dữ liệu Rio Bus (`scripts/preprocess_riobus.py` & `configs/data/riobus.yaml`).**
  - Đã khắc phục 3 lỗi dữ liệu cốt lõi:
    1. Sắp xếp thứ tự thời gian chuẩn xác, loại bỏ timestamp trùng lặp.
    2. Chuyển đổi đơn vị vận tốc từ km/h sang m/s ($v_{mps} = v_{kmh} / 3.6$), kiểm soát giới hạn $0 \le v \le 35\text{ m/s}$.
    3. Chuẩn hóa tọa độ $(x,y)$ về khung không gian mô phỏng ($1000\text{ m} \times 1000\text{ m}$) và nội suy đồng đều với bước thời gian $dt = 0.1\text{ s}$.
  - Xuất thành công file quỹ đạo chuẩn hóa `data/processed/riobus_corridor.csv` (100,000 dòng dữ liệu).

- **T4.2 — Lớp Mobility dựa trên dữ liệu thực (`TraceMobility` & `configs/env/scenario_riobus.yaml`).**
  - Xây dựng lớp `TraceMobility` trong `src/env/mobility.py` tuân thủ đầy đủ interface chuẩn (`step()`, `current_velocity()`, `current_direction`, `generate_initial_state()`).
  - Cập nhật `VehicularEdgeEnvironment` hỗ trợ `mobility_type: "trace"`, gắn trực tiếp quỹ đạo xe thực cho người dùng (User), xe hỗ trợ (AUX) và kẻ tấn công (Attacker).
  - Tạo cấu hình kịch bản thực nghiệm `configs/env/scenario_riobus.yaml`.

- **T4.3 — Nghiên cứu & Định lượng Sim-to-Real Gap (`scripts/sim2real_study.py`).**
  - Đánh giá phân phân phối vận tốc bằng kiểm định Kolmogorov-Smirnov ($KS = 0.9882, p < 10^{-5}$), chỉ ra sự khác biệt rõ rệt giữa mô hình vận tốc đều synthetic ($15-25\text{ m/s}$) và vận tốc thực của xe buýt (nhiều điểm dừng đỗ, vận tốc thấp).
  - Hoàn thành ma trận chuyển giao 2×2 (`Synthetic/Trace` $\times$ `Synthetic/Trace`), lưu kết quả tại `outputs/sim2real_transfer_matrix.csv`.

---

## Kiểm thử & Xác nhận

- **85/85 unit & integration tests** đã vượt qua 100% clean:
  ```bash
  python -m pytest tests/ -v
  ```
- File kiểm thử Phase 4 `tests/test_phase4.py` đã xác nhận tính đúng đắn của pipeline tiền xử lý và giao tiếp của lớp `TraceMobility`.

---

## Lệnh thực thi cho người dùng

```bash
# 1. Chạy pipeline tiền xử lý dữ liệu Rio bus
python scripts/preprocess_riobus.py --config configs/data/riobus.yaml

# 2. Chạy toàn bộ 85 unit tests (Phase 1-4)
python -m pytest tests/ -v

# 3. Chạy thực nghiệm nghiên cứu Sim-to-Real Gap
python scripts/sim2real_study.py
```
