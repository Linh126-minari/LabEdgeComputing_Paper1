# Báo cáo Phase 4 (làm lại): Tích hợp dữ liệu GPS thật Rio Bus

**Ngày:** 16/09/2026 · Thay thế phần T4.1–T4.3 của `BAO_CAO_PHASE4.md`

## 0. Vì sao phải làm lại

Bộ dữ liệu cũ `data/processed/data_train/` **không phải dữ liệu thật**. Nó được sinh bởi `scripts/generate_sample_riobus_raw.py`:
- 8 xe chạy thẳng theo góc `i·π/4`, vận tốc lấy ngẫu nhiên đều trong 15–55 km/h;
- rồi bị co giãn tỷ lệ để nhét vào vùng 1 km × 1 km.

Vì vậy các kết quả sau của `BAO_CAO_PHASE4.md` **đều không dùng được**:
- câu "tích hợp dữ liệu GPS thực tế";
- KS = 0.9882;
- ma trận sim-to-real.

Đã xóa bộ sample giả (`data/processed/data_train/`, `riobus_corridor.parquet`) và script sinh ra nó.

## 1. Dữ liệu nguồn

| Mục | Giá trị |
|---|---|
| Nguồn | GPS xe buýt Rio de Janeiro (SMTR / data.rio), `data/unstructured/individual_bus_lines_data/bus_line_*/part-*.csv` (output Spark) |
| Thời gian | 25/01/2019 – 21/03/2019 (~8 tuần) |
| Quy mô | 9,556,337 điểm · 77 tuyến · 2,162 xe |
| Cột | `date, time, order, line, latitude, longitude, speed` (km/h) |
| Lấy mẫu | Trung vị 60–70 s; p90 ≈ 140 s (sau khi tách chuyến) |

Các phần còn lại trong `data/unstructured/` **không dùng**:
- `individual_bus_orders_data/`: chỉ 3 tuyến, trùng với dữ liệu chính.
- `Trajectory/`: chỉ có ảnh PNG.

## 2. Lỗi dữ liệu phát hiện và cách xử lý

| # | Lỗi | Xử lý | Số dòng |
|---|---|---|---|
| 1 | Cột `time` mang ngày giả **2025-11-25** (ngày chạy Spark); ngày thật nằm ở `date` | Ghép `date` + giờ lấy từ `time` | toàn bộ |
| 2 | Mọi cột được đọc thành string | Ép kiểu số | 0 lỗi |
| 3 | Dữ liệu Carnaval 2019 (02–06/03) có giao thông bất thường | Loại các ngày này | 434,620 |
| 4 | `speed` lên tới 971 km/h | Loại điểm có speed > 120 km/h | 111 |
| 5 | Trùng `(order, timestamp)` | Bỏ bản trùng | 1,306 |
| 6 | Nhảy GPS | Bỏ "gai" (vận tốc vào và ra đều > 120 km/h), sau đó lặp để bỏ điểm nhảy còn sót | 16,173 |
| 7 | Khoảng trống lớn (có chỗ tới hàng giờ) | Tách chuyến khi gap > 180 s; bỏ chuyến < 5 điểm | 2,159,229 |
| 8 | `speed` chỉ tương quan 0.62 với chuyển động thật | **Tính lại vận tốc từ tọa độ** | – |

**Sau làm sạch:** 6,944,898 điểm · 1,989 xe · 77 tuyến · 642,130 chuyến.

Lưu ý về dòng 7: tỷ lệ bị loại khá cao (~23%) vì nhiều xe gửi GPS rất thưa (các điểm cách nhau trên 3 phút). Có thể nới `max_trip_gap_s` và `min_trip_points` nếu cần giữ thêm dữ liệu.

## 3. Pipeline (`scripts/preprocess_riobus.py`, config `configs/data/riobus.yaml`)

1. **`clean_raw`**: làm sạch theo bảng trên.
2. **`find_hotspot`**:
   - Chia lưới 1 km, chọn ô có nhiều cặp (xe, giờ) nhất trong giờ cao điểm ngày thường. Đếm cặp (xe, giờ) thay vì số điểm để không thiên vị xe đứng lâu.
   - Kết quả: **lat −22.913086, lon −43.208268** (khu Praça da Bandeira / Tijuca, cửa ngõ vào trung tâm).
   - Chiếu lat/lon ra mét bằng phép chiếu equirectangular cục bộ (sai số < 0.1% trong phạm vi vài km). Hotspot được đặt trùng vị trí MEC #0 của simulator, tức (250, 250).
3. **Cửa sổ thời gian**:
   - Mỗi ngày thường có 15 cửa sổ **30 phút**, bắt đầu mỗi giờ từ 6h đến 20h. Một episode (1000 quyết định) kéo dài ~1700 s mô phỏng, nên cửa sổ phải ≥ 1800 s để trace không quay vòng giữa episode.
   - Trong mỗi cửa sổ, lấy mọi đoạn quỹ đạo thật (không có khoảng trống > 180 s), cắt trong bán kính 1.5 km, dài ≥ 60 s và có đi vào trong bán kính 1 km quanh MEC.
   - Nội suy tuyến tính, lưu ở bước **1 s**.
4. **Gán slot**: xem mục 4.
5. **Chia train/test theo ngày** (không rò rỉ thời gian): train 25/01–28/02, test 07/03–21/03.
6. **Thống kê và hình**: `riobus_stats.json`, `riobus_overview.png`.

Chạy:
```bash
python scripts/preprocess_riobus.py                 # ~1.5 phút
python scripts/preprocess_riobus.py --reuse-clean   # bỏ qua bước làm sạch khi chỉ đổi tham số cửa sổ
```

## 4. Quyết định thiết kế quan trọng: slot

**Vấn đề:** xe buýt chỉ **đi ngang qua** vùng quanh MEC. Trung vị một chuyến là ~11 phút, và ở trong bán kính 800 m chỉ ~2 phút. Khi thử yêu cầu mỗi xe có mặt liên tục suốt một cửa sổ 10 phút, **không cửa sổ nào đạt** (thường chỉ 1–3 xe).

**Giải pháp:** simulator có số thực thể cố định (10 user + 3 AUX + 2 attacker = 15), nên mỗi thực thể là một **slot**:
- Các đoạn quỹ đạo thật được xếp vào slot bằng lập lịch khoảng thời gian (best-fit).
- Khi một xe buýt rời vùng, xe buýt thật kế tiếp nhận slot đó.
- Mọi slot dùng **chung đồng hồ** (`t_rel`), nên khoảng cách V2I/V2V giữa các xe tại mỗi thời điểm là thật.
- Khi slot không có xe: `active = 0`, giữ vị trí cuối, `v = 0`.
- Chỉ giữ cửa sổ có độ phủ trung bình của các slot ≥ 80%.

**Định dạng output** (`riobus_train.parquet`, `riobus_test.parquet`):

| Cột | Ý nghĩa |
|---|---|
| `window_id` | `YYYYMMDD_HHMM` |
| `vehicle_id` | `slot00`…`slot14` |
| `bus_id`, `line` | Xe buýt thật đang giữ slot (rỗng nếu inactive) |
| `t_rel` | Giây, 0…1800, bước 1 s |
| `x_m`, `y_m` | Mét, khung simulator (MEC #0 ở (250, 250)) |
| `v_mps` | Tính từ vị trí |
| `active` | 1 = có xe thật |

`riobus_corridor.csv` (1 cửa sổ) được giữ để tương thích với code cũ.

## 5. Kết quả

| Chỉ số | Train | Test |
|---|---|---|
| Số cửa sổ 30 phút (×15 slot) | 108 (~54 h) | 35 (~17.5 h) |
| Xe buýt khác nhau / tuyến | 518 / 29 | 379 / 28 |
| Độ phủ slot (mean / p10) | 0.85 / 0.69 | 0.85 / 0.71 |
| Số lần đổi xe / slot / cửa sổ | 3.16 | 3.25 |
| Vận tốc (mean / p50 / p90, m/s) | 4.42 / 3.70 / 9.00 | 4.43 / 3.69 / 9.11 |
| Thời gian lưu trú vùng MEC L = 300 m (p10 / p50 / p90, s) | 29 / 86 / 210 | 31 / 92 / 221 |
| Láng giềng trong 150 m / 200 m / 300 m (mean) | 0.36 / 0.55 / 1.00 | 0.36 / 0.55 / 1.01 |

**Nhận xét cho bài báo:**
- **Vận tốc thật khác xa giả định synthetic.** Trace cho trung bình ~4.5 m/s (16 km/h), trong khi synthetic là U(15, 25) m/s theo [A]. KS = 0.977 (p ≈ 0).
- **Thời gian lưu trú MEC thật có trung vị ~86 s.** Con số này lớn gấp ~4 lần synthetic (trung vị 21 s, KS = 0.830), nên ràng buộc deadline do mobility (T1.1) sẽ ít chặt hơn với dữ liệu thật. Đây là một điểm sim-to-real đáng viết.
- **Mật độ láng giềng rất thấp (<1 xe trong 200 m).** Nguyên nhân là dữ liệu chỉ có xe buýt. Giả định số AUX/attacker trong tầm V2V sẽ bị đánh giá thấp. Xem hạn chế 3 ở mục 7.
- Phân phối train và test rất gần nhau, nên chia theo ngày là ổn định.

![overview](data/processed/riobus/riobus_overview.png)

## 6. Thay đổi code

| File | Thay đổi |
|---|---|
| `scripts/preprocess_riobus.py` | Viết lại toàn bộ (mục 2–4) |
| `configs/data/riobus.yaml` | Tham số mới, có chú thích |
| `src/env/mobility.py` | `TraceMobility` **nội suy theo thời gian** (trace 1 s dùng được với dt = 0.1 s); thêm `is_active`; hướng tính từ chuyển động thực tế |
| `src/env/core.py` | Thêm `trace_window_id`: `None` thì mỗi `reset()` chọn ngẫu nhiên một cửa sổ (tái lập được khi truyền seed), hoặc cố định theo id |
| `src/env/core.py` | **Sửa lỗi:** `reset()` trước đây gán vị trí ngẫu nhiên **đè lên trace** |
| `src/env/entities.py` | **Sửa lỗi:** vận tốc của User/AUX **không bao giờ cập nhật** theo trace, nên thời gian lưu trú và thời gian liên kết V2V luôn dùng vận tốc ban đầu |
| `train.py` | **Sửa lỗi:** khối `mobility:` trong `scenario_riobus.yaml` trước đây bị bỏ qua, nên `env=scenario_riobus` thực chất vẫn chạy synthetic. Nay đã đọc `type`, `trace_file`, `trace_window_id` |
| `configs/env/scenario_riobus.yaml` | Trỏ tới `riobus_train.parquet` |
| `scripts/sim2real_study.py` | KS dùng train (chỉ các điểm active); env trace dùng test |
| `tests/test_phase4.py` | Viết lại, 9 test: timestamp, trùng, nhảy GPS, ngày lễ, tách chuyến, phép chiếu, đồng hồ chung và đổi xe trong slot, nội suy thời gian, `is_active`, env reset/cửa sổ, chạy với trace thật |

**Test:** `python -m pytest tests/`, **91 passed**.

## 7. Hạn chế và việc còn lại

1. ~~Sim-to-real không huấn luyện gì.~~ **Đã làm lại** (§8.2); còn phải chạy đầy đủ.
2. **Nội suy thẳng giữa các điểm cách nhau ~60 s**, chưa map-matching theo bản đồ OSM. Hệ quả: tỷ lệ dừng sau nội suy (7%) thấp hơn nhiều so với dữ liệu gốc (31%), và quỹ đạo có thể cắt ngang khối nhà. Hướng khắc phục: map-matching bằng OSRM/Valhalla/`leuvenmapmatching`.
3. **Chỉ có xe buýt, và chỉ 77/~700 tuyến của Rio**, nên mật độ xe bị đánh giá thấp. Hướng khắc phục:
   - lấy thêm tuyến trong khu hotspot từ nguồn gốc data.rio;
   - hoặc dựng kịch bản **SUMO trên bản đồ OSM của khu hotspot**, hiệu chỉnh lưu lượng theo dữ liệu này, rồi dùng xe con làm AUX/attacker.
4. **Slot đổi xe gây nhảy vị trí** (trung bình ~3.2 lần mỗi slot trong cửa sổ 30 phút). ~~Env chưa dùng `is_active`.~~ **Đã xử lý** (§8.1).
5. ~~Trace quay vòng giữa episode.~~ **Đã xử lý**: cửa sổ tăng lên 1800 s. Nếu tăng `episode_length`, cần tăng `window_duration_s` tương ứng.
6. **Cần chạy lại** mọi thí nghiệm dùng `mobility_type="trace"`, và sửa phần Phase 4 trong `.tex`/`BAO_CAO_PHASE4.md` nếu đã viết "dữ liệu thật" dựa trên bộ cũ.

## 8. Cập nhật đợt 2 (16/09/2026)

### 8.1 Slot inactive trong môi trường
- `_mec_feasible` / `_aux_feasible` / `action_mask`:
  - user inactive thì chỉ được tính local;
  - AUX inactive thì không phải đích offload.
- Background user inactive không sinh tải lên MEC/AUX.
- Attacker inactive không được đếm trong chế độ `attacker_model="realized"`.
- `reset()` gán slot có độ phủ cao nhất cho users (users[0] là agent), tiếp theo cho AUX, cuối cùng cho attacker.
- Test mới: `test_inactive_trace_slots_are_not_offload_targets`. Tổng cộng **92 passed**.

### 8.2 `scripts/sim2real_study.py` viết lại
- **KS vận tốc:** 0.728 (synthetic theo [B] μ = 13 m/s vs thật 4.42 m/s). Với U(15, 25) cũ: 0.977.
- **KS thời gian lưu trú MEC:** 0.553 (trung vị synthetic 31 s vs thật 86 s). Với U(15, 25) cũ: 0.830.
- **Ma trận 2×2 thật:**
  - Train `LagrangianDQN` (hoặc agent tùy chọn) trên synthetic và trên `riobus_train`.
  - Đánh giá trên synthetic và `riobus_test`, nhiều seed, có CI95.
- **Metric chung:** `objective = cost_energy + cost_service − ϖ₁·U` (cents/task), kèm viol_T, viol_E, delay, U.
- Đã chạy smoke run (3 episode × 100 step): pipeline thông, nhưng **chưa có số liệu thật**.

Lệnh chạy đầy đủ (ước tính ~1h40 cho mỗi lần train 500 episode trên CPU, dựa theo log cũ):
```bash
python scripts/sim2real_study.py --agent lagrangian_dqn --episodes 500 --seeds 0 1 2
python scripts/sim2real_study.py --agent dueling_dqn   --episodes 500 --seeds 0 1 2
```

### 8.3 Vấn đề phát hiện ở kết quả 3 thuật toán (`TONG_HOP_DANH_GIA_3_THUAT_TOAN.md`)
Đã ghi chú dạng comment ngay trong mục Numerical Results của `.tex`.

1. **"MADDPG" không tồn tại trong code.** `train.py` tạo `DRLAgent` cho mọi agent khác `LagrangianDQN`, nên "MADDPG" thực chất là Dueling Double DQN với vài hyperparameter khác (soft update, …). Điều này giải thích vì sao số liệu của nó gần như trùng với DuelingDQN.
2. **Reward không cùng thước đo.** Với `LagrangianDQN`, env chạy `penalty_mode="external"` và reward được log **chưa trừ penalty**. Hai baseline chạy `penalty_mode="fixed"` và reward được log **đã trừ penalty**. Hơn nữa λ_p = **5** (theo `params_master`), không phải 100 như tài liệu ghi. Vì vậy các con số "−2264 vs −11757" và "+80.7%" không phải là so sánh hợp lệ. Thực tế LagrangianDQN còn có viol_T **cao hơn** (6.24% so với 3%).
3. **Ràng buộc năng lượng gần như không khả thi.** E trung bình ≈ 2.3 J, trong khi E_max = 0.4 J, nên viol_E ≈ 100% ở mọi agent. Điều này giải thích vì sao λ_E chỉ tăng mãi (1.09 → 5.70). Cần chỉnh lại (κ = 1e-28, hoặc E_max, hoặc định nghĩa E) trước khi train lại.
4. **Cả 3 mô hình đều train với mobility `linear` (synthetic).** Log cho thấy `mobility.type: linear`. Hơn nữa, trước đợt sửa này, chạy `env=scenario_riobus` cũng vẫn là synthetic.

### 8.4 Git
- `.gitignore` bỏ qua `data/unstructured/` (755 MB) và toàn bộ `*.parquet` trong `data/processed/riobus/` (clean 90 MB, train 51 MB, test 17 MB). Tạo lại bằng `python scripts/preprocess_riobus.py`, mất ~2 phút.
- Đã bỏ theo dõi 5 file `__pycache__/*.pyc` từng bị commit (`git rm --cached`).
- Chưa commit.

### 8.5 Cửa sổ 30 phút
Đo trong smoke run: 50 quyết định tiêu tốn ~85 s mô phỏng, nên một episode 1000 quyết định dài ~1700 s. Với cửa sổ 10 phút, trace sẽ quay vòng gần 3 lần mỗi episode. Vì vậy cửa sổ được đổi thành 1800 s, bắt đầu mỗi giờ 6h–20h. Kết quả: 108 cửa sổ train, 35 cửa sổ test (bảng §5 đã cập nhật).
