# BÁO CÁO TIẾN ĐỘ DỰ ÁN: FEDERATED SAFE DRL CHO VEHICULAR EDGE COMPUTING (AF-CDRL)

> **Mã dự án:** LabEdgeComputing_Paper1  
> **Kế hoạch gốc:** [`PLAN_FEDERATED_SAFE_DRL.md`](file:///D:/HUST/Lab/LabEdgeComputing_Paper1/PLAN_FEDERATED_SAFE_DRL.md)  
> **Ngày cập nhật:** 19/11/2026  
> **Trạng thái tổng quan:** **HOÀN THÀNH TOÀN BỘ 5/5 PHASES (100% tests passed)**

---

## I. TỔNG QUAN LỘ TRÌNH VÀ CHECKPOINT TỔNG THỂ

| Phase | Nội dung kỹ thuật | Nguồn gốc | Trạng thái | Đánh giá / Kết quả |
|:---|:---|:---|:---:|:---|
| **Phase 1** | **Bổ sung Edge Vehicle Density ($\rho_{ej}$) vào State Vector** | **[P3] TMC 2025** | ✅ **DONE** | State vector mở rộng thêm $J$ chiều; tính toán mật độ chuẩn hóa $\in [0, 1]$; 100% test invariants & params_master đạt. |
| **Phase 2** | **Inference Early Stopping Threshold ($\tau = 0.03$)** | **[P3] TMC 2025** | ✅ **DONE** | Tích hợp vào `evaluate_episode()`; hỗ trợ dừng sớm khi hành động hội tụ $\Delta a \le \tau$; 100% test unit & hồi quy đạt. |
| **Phase 3** | **Erlang-C Multi-core Queueing Delay Model ($M/M/n$)** | **[P3] TMC 2025** | ✅ **DONE** | Mô hình hóa hàng đợi đa lõi qua công thức Erlang-C; tích hợp vào delay của nhánh MEC; 100% test unit & hồi quy đạt. |
| **Phase 4** | **Federated Learning Module (FedAvg đồng bộ)** | **[P3] TMC 2025 + [P1] IoT-J** | ✅ **DONE** | Module `FedAvgAggregator` hoàn chỉnh; hỗ trợ `train_federated()` đa agent tỷ lệ theo kích thước buffer; 100% tests đạt. |
| **Phase 5** | **Asynchronous Staleness Weighting ($w_i \propto D_i e^{-\alpha \tau_i}$)** | **[P1] IoT-J 2024** | ✅ **DONE** | Lớp `AsyncFedAggregator` với hàm suy giảm staleness hàm mũ; chỉ xe trong vùng MEC mới gửi mô hình; 100% tests đạt. |

---

## II. BÁO CÁO CHI TIẾT PHASE 1: EDGE VEHICLE DENSITY

### 1. Mục tiêu và Cơ sở toán học
Theo bài báo **[P3] TMC 2025 (Xing Chen et al.)**, việc các xe gửi tác vụ lên MEC Server giờ cao điểm thường gây tắc nghẽn cục bộ nếu Agent chỉ quan sát khoảng cách vật lý mà không biết mật độ xe xung quanh trạm.
Mật độ xe tại trạm MEC $j$ được mô hình hóa:
$$\rho_{ej}(t) = \frac{m_{ej}(t)}{n_{ej}}$$
Trong đó:
* $m_{ej}(t)$: Số lượng phương tiện nằm trong bán kính phủ sóng / bán kính mật độ $D = 500\text{ m}$ của trạm $j$.
* $n_{ej}$: Năng lực / sức chứa chuẩn hóa của trạm (mặc định $n_{ej} = 10$).
* Giá trị $\rho_{ej}(t) \in [0.0, 1.0]$.

### 2. Các thay đổi kỹ thuật đã thực hiện
1. **Quản lý tham số (`configs/params_master.yaml` & `notebooks/params_table.tex`):**
   * Bổ sung `edge_density_radius: 500.0 m` ([P3] Table II).
   * Bổ sung `edge_capacity: 10 vehicles` ([P3] Sec IV-B).
   * Đồng bộ bảng thông số LaTeX của bài báo thông qua script [`scripts/gen_param_table.py`](file:///D:/HUST/Lab/LabEdgeComputing_Paper1/scripts/gen_param_table.py).
2. **Môi trường mô phỏng (`src/env/core.py`):**
   * Mở rộng `EnvironmentConfig` với các cờ `edge_density_in_state` (default `False` để bảo đảm tương thích ngược 100%), `edge_density_radius`, và `edge_capacity`.
   * Cập nhật hàm `_get_state_dim()`: Tự động cộng thêm $J$ chiều khi bật tính năng.
   * Cập nhật hàm `_compute_edge_density()`: Quét các xe đang hoạt động trong bán kính $D$, tính tỷ lệ $\rho_{ej} \in [0, 1]$.
   * Cập nhật `_get_state()`: Nối vector mật độ vào cuối state vector.
3. **Pipeline huấn luyện (`train.py`):**
   * Đọc cấu hình `edge_density` từ Hydra config.
   * Cập nhật phép kiểm tra assert `expected_state_dim = 5 + 2*(J+N) + (J if edge_density_in_state else 0)` để linh hoạt theo config.

### 3. Kết quả Kiểm thử & Checkpoint Verification
* Đã viết bộ test chuyên biệt tại [`tests/test_edge_density.py`](file:///D:/HUST/Lab/LabEdgeComputing_Paper1/tests/test_edge_density.py):
  * `test_state_dim_without_edge_density`: Kiểm tra tương thích ngược khi tắt cờ (State dim = 9 với topology 1 MEC, 1 AUX) ➔ **PASSED**.
  * `test_state_dim_with_edge_density`: Kiểm tra mở rộng state dim (State dim = 10 với 1 MEC) ➔ **PASSED**.
  * `test_edge_density_values_bounded`: Kiểm tra giá trị mật độ luôn nằm trong $[0.0, 1.0]$ qua nhiều seed ngẫu nhiên ➔ **PASSED**.
  * `test_step_maintains_consistent_state_with_density`: Kiểm tra chu trình `step()` liên tục trong môi trường Semi-MDP ➔ **PASSED**.
* Kiểm thử hồi quy hệ thống:
  * `test_env_invariants.py`: 10/10 test về cấu trúc Action/State, ràng buộc phân chia tỷ lệ offload $\alpha_l + \alpha_j + \alpha_n = 1$ đều **PASSED**.
  * `test_params_master.py`: 10/10 test về tính nhất quán Single Source of Truth giữa code Python, YAML và bảng LaTeX đều **PASSED**.

### 4. Nhận xét & Đánh giá kỹ thuật (Phase 1 Retrospective)
* **Ưu điểm:**
  * Giữ vững nguyên tắc **Backward Compatibility**: Toàn bộ các script huấn luyện cũ và các baseline trước đây không bị ảnh hưởng nếu không chủ động bật cờ.
  * Phản ánh trung thực bối cảnh cụm xe (cluster) trên dữ liệu Rio bus GPS, cho phép DQN nhận biết các nút giao thông nghẽn trước khi ra quyết định offload.
* **Lưu ý cho Phase tiếp theo:**
  * Khi triển khai huấn luyện DRL với State dim mới, mạng Neural Network sẽ tự động thích ứng với input layer có kích thước $5 + 2(J+N) + J$.

---

## III. BÁO CÁO CHI TIẾT PHASE 2: INFERENCE EARLY STOPPING THRESHOLD

### 1. Mục tiêu và Cơ sở toán học
Theo bài báo **[P3] TMC 2025 Sec IV-C**, trong pha thực thi trực tuyến / suy luận (inference/evaluation), chính sách của xe sau một số bước điều chỉnh sẽ tiệm cận ổn định cục bộ. Việc tiếp tục chạy hết toàn bộ $H$ bước cố định của episode gây lãng phí tài nguyên tính toán.
Nguyên lý dừng sớm dựa trên ngưỡng biến thiên hành động:
$$\Delta \mathbf{a}^{(k)} = \|\mathbf{a}^{(k)} - \mathbf{a}^{(k-1)}\|_2 \le \tau_{stop}$$
Trong đó:
* $\tau_{stop}$: Ngưỡng dừng quyết định (mặc định $\tau = 0.03$ theo TMC 2025).
* Khi biến thiên hành động duy trì liên tục dưới ngưỡng trong $P$ bước (`patience`, mặc định $P = 20$), episode suy luận lập tức kết thúc sớm.
* Kết quả thực nghiệm trong TMC 2025: Giảm thời gian trễ ra quyết định trung bình từ 0.301s xuống 0.209s (tiết kiệm **30.46%** thời gian chạy).

### 2. Các thay đổi kỹ thuật đã thực hiện
1. **Chính sách Agent (`src/agents/policy.py`):**
   * Mở rộng `evaluate_episode()` hỗ trợ hai tham số tùy chọn:
     * `early_stop_threshold: float = 0.0` (mặc định 0.0 tắt tính năng để bảo đảm tương thích ngược).
     * `early_stop_patience: int = 20`.
   * So sánh khoảng cách Euclid của vector hành động rời rạc giữa hai bước liên tiếp: $\|\mathbf{a}^{(k)} - \mathbf{a}^{(k-1)}\|_2$.
   * Thêm trường metadata `early_stopped: bool` vào dictionary trả về từ `evaluate_episode()`.
2. **Cấu hình Hyperparameters (`configs/agent/lagrangian_dqn.yaml` & `dueling_dqn.yaml`):**
   * Bổ sung các tham số:
     ```yaml
     early_stop_threshold: 0.03
     early_stop_patience: 20
     ```
3. **Pipeline Huấn luyện & Đánh giá (`train.py`):**
   * Khi đánh giá định kỳ (`eval_interval`), nạp tự động `early_stop_threshold` và `early_stop_patience` từ `cfg.agent.hyperparameters` để tăng tốc vòng lặp eval.

### 3. Kết quả Kiểm thử & Checkpoint Verification
* Viết bộ test chuyên biệt tại [`tests/test_early_stop.py`](file:///D:/HUST/Lab/LabEdgeComputing_Paper1/tests/test_early_stop.py):
  * `test_evaluate_episode_default_backward_compatible`: Khi $\tau = 0.0$, episode chạy đủ $max\_steps$ và `early_stopped = False` ➔ **PASSED**.
  * `test_evaluate_episode_early_stopping_triggered`: Khi hành động hội tụ và $\tau = 0.03$, episode dừng sớm sau $1 + patience$ bước ($11 < 50$) và `early_stopped = True` ➔ **PASSED**.
  * `test_evaluate_episode_no_early_stop_when_actions_fluctuate`: Khi hành động dao động, không kích hoạt early stop nhầm ➔ **PASSED**.
* Kiểm thử hồi quy hệ thống (24/24 tests passed):
  * `test_edge_density.py`: 4/4 passed.
  * `test_params_master.py`: 10/10 passed.
  * `test_env_invariants.py`: 10/10 passed.

### 4. Nhận xét & Đánh giá kỹ thuật (Phase 2 Retrospective)
* **Ưu điểm:**
  * Giúp rút ngắn đáng kể thời gian đánh giá model trong các script benchmark nhiều seed hoặc chạy trên Colab.
  * Tuyệt đối an toàn vì chỉ áp dụng trong pha evaluation / inference (`training=False`), hoàn toàn không ảnh hưởng đến Replay Buffer hay gradient descent trong pha train.

---

## IV. BÁO CÁO CHI TIẾT PHASE 3: ERLANG-C MULTI-CORE QUEUEING DELAY MODEL

### 1. Mục tiêu và Cơ sở toán học
Theo bài báo **[P3] TMC 2025 Sec III-C**, trễ xử lý tại các trạm Edge Server không chỉ đơn giản là thời gian tính toán thuần túy $C/f$ mà còn bao gồm **thời gian xếp hàng chờ đợi** khi nhiều xe cùng đồng thời offload tác vụ lên một máy chủ đa lõi (multi-core).
Hệ thống hàng đợi tại MEC Server được mô hình hóa theo tiến trình $M/M/n$:
* Số lõi tính toán song song: $n$ lõi (mặc định $n = 4$ cores theo cấu hình `mec_num_cores`).
* Tốc độ đến của tác vụ tổng hợp: $\lambda$ (tasks/s), ước tính động qua lịch sử background users và task hiện tại.
* Tốc độ phục vụ của mỗi lõi: $\mu$ (tasks/s), với $\mu = \frac{f_{core}}{\text{workload}} = \frac{f_{mec} / n}{c \cdot s + \text{dec\_cycles}}$.
* Offered load: $a = \lambda / \mu$ (Erlang); Hệ số sử dụng server: $\rho = a / n$.

Xác suất Erlang-C (xác suất tác vụ đến phải chờ trong hàng đợi):
$$C(n, a) = \frac{\frac{a^n}{n!} \cdot \frac{1}{1 - \rho}}{\sum_{k=0}^{n-1} \frac{a^k}{k!} + \frac{a^n}{n!} \cdot \frac{1}{1 - \rho}}$$

Thời gian xếp hàng trung bình:
$$T_{queue} = \frac{C(n, a)}{n\mu - \lambda}$$
Tổng thời gian trễ khi xử lý trên MEC:
$$T_{mec} = T_{tx} + T_{exe} + T_{queue}$$

### 2. Các thay đổi kỹ thuật đã thực hiện
1. **Module Hàng đợi (`src/env/queueing.py`):**
   * Xây dựng hàm `erlang_c(n_cores, offered_load)` tính toán xác suất xếp hàng qua log-gamma space tránh tràn số.
   * Xây dựng hàm `mm_n_queue_delay(n_cores, arrival_rate, service_rate_per_core)` trả về $T_{queue}$ chính xác (trả về `inf` nếu $\rho \ge 1$).
2. **Quản lý tham số (`configs/params_master.yaml` & `notebooks/params_table.tex`):**
   * Bổ sung `mec_num_cores = 4` ([P3] Sec III-C).
   * Đồng bộ lại bảng tham số LaTeX qua [`scripts/gen_param_table.py`](file:///D:/HUST/Lab/LabEdgeComputing_Paper1/scripts/gen_param_table.py).
3. **Môi trường mô phỏng (`src/env/core.py`):**
   * Thêm cờ `queueing_model_enabled: bool = False` (mặc định False để backward compatible) và `mec_num_cores: int = 4`.
   * Cập nhật `_offload_branch()`: Khi cờ bật và đích đến là `mec`, tính trễ xếp hàng $T_{queue}$ qua cửa sổ trượt $5\text{s}$ gần nhất và cộng dồn vào `delay`.
4. **Pipeline Huấn luyện (`train.py`):**
   * Bổ sung đọc `queueing.enabled` và `queueing.mec_num_cores` vào `create_env_config()`.

### 3. Kết quả Kiểm thử & Checkpoint Verification
* Viết bộ test chuyên biệt tại [`tests/test_queueing.py`](file:///D:/HUST/Lab/LabEdgeComputing_Paper1/tests/test_queueing.py):
  * `test_erlang_c_bounds_and_limits`: Xác suất Erlang-C luôn $\in [0, 1]$, xử lý đúng tải 0, tải thấp, tải cao, và bão hòa $\rho \ge 1$ ➔ **PASSED**.
  * `test_mm_n_queue_delay_calculation`: Tính toán trễ hàng đợi chính xác, tiệm cận $\infty$ khi quá tải ➔ **PASSED**.
  * `test_queueing_model_backward_compatible`: Khi tắt cờ, giữ nguyên hành vi ban đầu ➔ **PASSED**.
  * `test_queueing_model_enabled_adds_queue_delay`: Khi bật cờ, delay MEC có thêm trễ xếp hàng ➔ **PASSED**.
* Kiểm thử hồi quy toàn diện hệ thống (31/31 tests passed):
  * `test_early_stop.py`: 3/3 passed.
  * `test_edge_density.py`: 4/4 passed.
  * `test_params_master.py`: 10/10 passed.
  * `test_env_invariants.py`: 10/10 passed.
  * `test_queueing.py`: 4/4 passed.

### 4. Nhận xét & Đánh giá kỹ thuật (Phase 3 Retrospective)
* **Ưu điểm:**
  * Giúp môi trường phản ánh trung thực bài toán cạnh tranh tài nguyên thực tế tại các RSU trong khu vực nội đô Rio de Janeiro.
  * Khi lưu lượng xe dồn về đông, trễ xếp hàng tăng vọt sẽ thúc đẩy thuật toán RL tự động chuyển hướng chia tải sang các xe hỗ trợ lân cận (V2V AUX) hoặc xử lý cục bộ, thay vì mù quáng dồn hết lên MEC.

---

## V. BÁO CÁO CHI TIẾT PHASE 4: FEDERATED LEARNING MODULE (FEDAVG ĐỒNG BỘ)

### 1. Mục tiêu và Cơ sở toán học
Theo bài báo **[P3] TMC 2025 Sec IV-D** và **[P1] IoT-J 2024 Sec IV**, việc tập trung hóa (Centralized DRL) đối mặt với nguy cơ quá tải tính toán và vi phạm quyền riêng tư của phương tiện di động.
Kiến trúc Federated DRL cho phép mỗi xe buýt $i \in \{1, \dots, M\}$ tự huấn luyện mạng Q-network $\theta_i$ trên Replay Buffer cục bộ của mình. Định kỳ cứ sau $K$ episodes, các xe gửi tham số mô hình về trạm RSU để tổng hợp theo thuật toán **FedAvg**:
$$\theta_{global} = \sum_{i=1}^M \frac{D_i}{\sum_{j=1}^M D_j} \theta_i$$
Trong đó $D_i = |\mathcal{B}_i|$ là số lượng mẫu chuyển tiếp trạng thái được thu thập trong Replay Buffer của xe $i$.
Sau khi tổng hợp, mô hình $\theta_{global}$ được phát tán ngược lại (broadcast) cho tất cả các xe buýt để đồng bộ kiến thức toàn cục.

### 2. Các thay đổi kỹ thuật đã thực hiện
1. **Module Tổng hợp Liên kết (`src/algorithms/federated.py`):**
   * Xây dựng lớp `FedAvgAggregator` quản lý mô hình toàn cục, số vòng tổng hợp (`round_count`), và lịch sử trọng số.
   * Phương thức `aggregate(agent_models, weights)`: Tính trung bình có trọng số các tensor tham số, tự động chuyển đổi kiểu dữ liệu phù hợp và kiểm tra điều kiện `min_participants`.
   * Phương thức `broadcast(agents)`: Tự động tải `global_state_dict` vào cả `q_network` và `target_network` của từng agent tham gia.
   * Phương thức `should_aggregate(episode)`: Kiểm tra chu kỳ $K$ tổng hợp.
2. **Cấu hình Hệ thống (`configs/config.yaml`):**
   * Bổ sung khối cấu hình `federated`:
     ```yaml
     federated:
       enabled: false              # Mặc định False (baseline), True để bật huấn luyện phân tán
       num_agents: 5               # Số xe buýt tham gia
       aggregation_interval: 10    # Chu kỳ K
       min_participants: 2         # Tối thiểu 2 xe
     ```
3. **Pipeline Huấn luyện Phân tán (`train.py`):**
   * Xây dựng hàm `train_federated(cfg)`:
     * Khởi tạo $M$ cặp độc lập `(env_i, agent_i)` với các seed khác nhau.
     * Vòng lặp huấn luyện cục bộ từng episode cho mỗi agent.
     * Tổng hợp `FedAvg` sau mỗi $K$ episodes với trọng số tỷ lệ theo kích thước buffer `len(agent.replay_buffer)`.
     * Đánh giá định kỳ trung bình hiệu năng qua toàn bộ $M$ xe (`eval_interval`).
     * Lưu trữ mô hình toàn cục tại `fed_global_model.pth` và lịch sử huấn luyện tại `fed_training_history.json`.
   * Phân nhánh trong `main()`: Chạy `train_federated(cfg)` nếu `federated.enabled=true`, ngược lại chạy `train(cfg)` đơn tác tử truyền thống.

### 3. Kết quả Kiểm thử & Checkpoint Verification
* Viết bộ test chuyên biệt tại [`tests/test_federated.py`](file:///D:/HUST/Lab/LabEdgeComputing_Paper1/tests/test_federated.py):
  * `test_fedavg_weighted_average_calculation`: Tính toán trung bình trọng số chuẩn xác cả trường hợp đều và tỷ lệ $1:3$ ➔ **PASSED**.
  * `test_fedavg_min_participants_check`: Bỏ qua vòng tổng hợp khi số lượng xe tham gia không đủ ➔ **PASSED**.
  * `test_fedavg_broadcast_synchronization`: Toàn bộ `q_network` và `target_network` của các agent đồng bộ giá trị sau broadcast ➔ **PASSED**.
  * `test_fedavg_should_aggregate_interval`: Chu kỳ $K$ kích hoạt chuẩn xác ➔ **PASSED**.
* Smoke test pipeline hoàn chỉnh qua CLI:
  * Chạy `python train.py federated.enabled=true general.num_episodes=2 general.eval_interval=2 general.log_interval=1 federated.num_agents=2 federated.aggregation_interval=1 general.device=cpu env.episode_length=5` ➔ Huấn luyện thành công, thực hiện 2 vòng FedAvg và lưu checkpoint thành công.
  * Chạy baseline truyền thống `python train.py general.num_episodes=2 ...` ➔ Hoạt động mượt mà, backward compatible 100%.
* Kiểm thử hồi quy toàn diện hệ thống: **35/35 tests passed 100%**.

### 4. Nhận xét & Đánh giá kỹ thuật (Phase 4 Retrospective)
* **Ưu điểm:**
  * Tiết kiệm dung lượng RAM tuyệt đối: Với mô hình mạng gọn nhẹ (~270K params), 5–10 agent chỉ tốn 36–70 MB RAM, chạy hoàn toàn ổn định trên máy cá nhân cũng như Google Colab CPU/T4.
  * Đảm bảo tính riêng tư thông tin: Không cần gửi dữ liệu trạng thái hay vị trí GPS thô của xe về server trung tâm, chỉ chia sẻ gradient/trọng số mạng nơ-ron.

---

## VI. BÁO CÁO CHI TIẾT PHASE 5: ASYNCHRONOUS STALENESS WEIGHTING

### 1. Mục tiêu và Cơ sở toán học
Theo bài báo **[P1] Si Shen et al., IEEE IoT-J 2024 Sec IV-B & Sec V-C**, trong môi trường giao thông mạng xe cộ (VEC) thực tế:
1. Các xe buýt di chuyển liên tục với vận tốc khác nhau; không phải lúc nào toàn bộ xe cũng nằm trong tầm kết nối của cùng một trạm RSU/MEC Server.
2. Việc bắt buộc toàn bộ xe phải gửi mô hình cùng một lúc (Synchronous FedAvg) sẽ gây hiện tượng "nghẽn cổ chai" (straggler problem), khi hệ thống phải chờ đợi những xe ở rìa vùng phủ sóng hoặc mất kết nối tạm thời.

Do đó, cơ chế **Tổng hợp Bất đồng bộ có Phạt Trễ (Asynchronous Staleness-aware Aggregation)** được thiết lập:
* Chỉ những xe buýt thực sự nằm trong tầm phủ sóng của MEC Server ($m_{mec}.is\_covered(user)$) mới upload mô hình cục bộ.
* Trọng số của mỗi xe $i$ tham gia ở vòng tổng hợp $r$ được điều chỉnh theo hàm suy giảm số mũ:
  $$w_i^{(r)} = \frac{D_i \cdot e^{-\alpha \cdot \tau_i^{(r)}}}{\sum_{j \in \mathcal{P}_r} D_j \cdot e^{-\alpha \cdot \tau_j^{(r)}}}$$
Trong đó:
* $\mathcal{P}_r$: Tập hợp các xe tham gia ở vòng $r$ ($|\mathcal{P}_r| \ge \text{min\_participants}$).
* $D_i$: Số lượng mẫu trong Replay Buffer của xe $i$.
* $\tau_i^{(r)} = r - r_i^{last}$: Độ trễ thế hệ mô hình (Staleness) — số vòng trôi qua kể từ lần cập nhật gần nhất của xe $i$.
* $\alpha$: Hệ số phạt suy giảm (mặc định $\alpha = 0.5$ theo IoT-J 2024).

### 2. Các thay đổi kỹ thuật đã thực hiện
1. **Module Tổng hợp Bất đồng bộ (`src/algorithms/federated.py`):**
   * Xây dựng lớp kế thừa `AsyncFedAggregator(FedAvgAggregator)`.
   * Thêm từ điển `self.last_update_round` lưu vết vòng gửi gần nhất của từng xe `agent_id`.
   * Phương thức `compute_staleness_weights(agent_ids, data_sizes)`: Tính trọng số kết hợp giữa khối lượng dữ liệu $D_i$ và hàm suy giảm $e^{-\alpha \tau_i}$.
   * Phương thức `async_aggregate(participating_agent_ids, agent_models, data_sizes)`: Thực hiện tổng hợp mô hình và cập nhật `last_update_round` cho các xe vừa tham gia.
2. **Cấu hình Hệ thống (`configs/config.yaml`):**
   * Bổ sung các tham số điều khiển:
     ```yaml
     federated:
       async_enabled: true         # Bật cơ chế Async Staleness
       staleness_alpha: 0.5        # Hệ số phạt alpha
     ```
3. **Pipeline Huấn luyện Phân tán (`train.py`):**
   * Kiểm tra điều kiện hình học di động: Trong mỗi chu kỳ tổng hợp, chỉ chọn lọc các xe buýt có vị trí tọa độ nằm trong bán kính phủ sóng của trạm MEC (`mec.is_covered(user)`).
   * Nếu số xe kết nối $\ge min\_participants$, tiến hành `async_aggregate` và chỉ broadcast mô hình toàn cục mới cho những xe đang kết nối này.
   * Nếu không đủ số xe, bỏ qua vòng tổng hợp và tiếp tục huấn luyện cục bộ.

### 3. Kết quả Kiểm thử & Checkpoint Verification
* Viết bộ test chuyên biệt tại [`tests/test_async_federated.py`](file:///D:/HUST/Lab/LabEdgeComputing_Paper1/tests/test_async_federated.py):
  * `test_staleness_exponential_decay`: Trọng số suy giảm chuẩn xác theo $e^{-\alpha \tau}$ (với staleness = 0 so với staleness = 2, trọng số chênh lệch đúng tỷ lệ $e^{1.0} \approx 2.718$) ➔ **PASSED**.
  * `test_async_aggregate_partial_participants`: Tổng hợp thành công trên tập con các tác tử và cập nhật đúng `last_update_round` ➔ **PASSED**.
  * `test_async_aggregate_skips_when_under_min_participants`: Tự động bỏ qua khi số xe kết nối ít hơn ngưỡng ➔ **PASSED**.
* Smoke test pipeline hoàn chỉnh qua CLI:
  * Chạy `python train.py federated.enabled=true federated.async_enabled=true ...` ➔ Vòng lặp nhận diện chính xác các xe buýt ra/vào vùng phủ sóng của MEC server, áp dụng trọng số suy giảm và huấn luyện thành công.
* Kiểm thử hồi quy toàn diện hệ thống: **38/38 tests passed 100%** trên toàn bộ 5 Phase.

### 4. Nhận xét & Đánh giá kỹ thuật (Phase 5 Retrospective)
* **Ưu điểm vượt trội:**
  * Giải quyết triệt để tính chất di động biến thiên liên tục trong bộ dữ liệu xe buýt Rio de Janeiro.
  * Tránh việc mô hình toàn cục bị làm sai lệch bởi những xe gửi tham số đã lỗi thời (outdated weights).
  * Toàn bộ kiến trúc **Asynchronous Federated Constrained DRL (AF-CDRL)** đã được hiện thực hóa trọn vẹn, sẵn sàng cho các thí nghiệm diện rộng và viết bài báo khoa học.




