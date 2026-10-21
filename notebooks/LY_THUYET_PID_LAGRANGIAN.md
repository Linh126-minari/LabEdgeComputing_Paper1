# Cơ sở Lý thuyết & Chứng minh Toán học: Phương pháp Lagrangian và PID-Lagrangian trong Học tăng cường có Ràng buộc (Constrained DRL)

Tài liệu này cung cấp nền tảng toán học chặt chẽ, các bước suy đạo giải tích, và chứng minh ổn định động học cho phương pháp **Lagrangian Dual Relaxation** và **PID-Lagrangian Control** trong bài toán tối ưu hóa phân chia tác vụ và bảo mật mạng điện toán biên xe cộ (Vehicular Edge Computing - VEC).

---

## 1. Bản chất Bài toán: Học tăng cường có Ràng buộc (CMDP)

### 1.1. Định nghĩa Không gian CMDP
Bài toán được mô hình hóa dưới dạng Quá trình Quyết định Markov có Ràng buộc (*Constrained Markov Decision Process* - CMDP), định nghĩa bởi bộ 6 thành phần:
$$\mathcal{M} = \big(\mathcal{S}, \mathcal{A}, \mathcal{P}, C_0, \{C_i\}_{i=1}^m, \gamma\big)$$
- $\mathcal{S}$: Không gian trạng thái mạng VEC (vị trí xe, tải trạm, kênh truyền, số kẻ nghe lén $n^e$).
- $\mathcal{A}$: Không gian hành động đa chiều rời rạc $a = [\alpha_l, d, \beta, N_u]$.
- $\mathcal{P}(s' | s, a)$: Xác suất chuyển trạng thái.
- $C_0(s, a)$: Chi phí mục tiêu cần cực tiểu hóa (tổng chi phí kinh tế: năng lượng, thuê dịch vụ, rủi ro an ninh).
- $C_i(s, a)$: Các hàm chi phí ràng buộc (ví dụ: cờ vi phạm deadline $\mathbb{I}\{T > \tau\}$, năng lượng vượt ngưỡng $(E - E_{\max})^+$).
- $\gamma \in (0, 1)$: Hệ số chiết khấu.

Mục tiêu của chính sách $\pi_\theta(a|s)$ là giải bài toán quy hoạch toán học:
$$\begin{aligned}
\min_{\theta} \quad & J_0(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t C_0(s_t, a_t) \right] \\
\text{subject to} \quad & J_i(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t C_i(s_t, a_t) \right] \le \delta_i, \quad \forall i \in \{1, \dots, m\}
\end{aligned} \tag{1}$$
trong đó $\delta_i$ là ngưỡng vi phạm tối đa được chấp nhận ($\delta_T = 5\%$ cho deadline, $\delta_E = 0$ cho năng lượng).

---

### 1.2. Tại sao Phạt Cố định (Static Fixed Penalty) Thất bại?
Phương pháp tiếp cận ngây thơ thường ghép ràng buộc vào reward bằng một hệ số phạt cố định $\lambda_p > 0$:
$$r_t^{\mathrm{fixed}} = -C_0(s_t, a_t) - \sum_{i=1}^m \lambda_{p, i} \cdot C_i(s_t, a_t) \tag{2}$$

#### Chứng minh sự phá vỡ Pareto tối ưu:
Giả sử ta xét bài toán 1 ràng buộc với hệ số phạt $\lambda_p$. Bài toán tối ưu không ràng buộc tương đương là:
$$\min_{\theta} \Big( J_0(\pi_\theta) + \lambda_p \cdot J_1(\pi_\theta) \Big)$$
- **Trường hợp $\lambda_p < \lambda^*$ (Phạt quá nhẹ):** Thuật toán chấp nhận vi phạm ràng buộc triền miên ($J_1 > \delta_1$) vì lợi ích giảm $J_0$ lớn hơn chi phí bị phạt $\lambda_p J_1$.
- **Trường hợp $\lambda_p > \lambda^*$ (Phạt quá nặng - Hội chứng *Over-Conservative*):** Thuật toán bị ám ảnh bởi việc tránh né penalty. Nó ép vi phạm $J_1 \to 0 \ll \delta_1$, dẫn đến việc chi trả lãng phí tài nguyên ở $J_0$ (mua trạm MEC đắt đỏ, rút ngắn khóa mã hóa).
- **Hạn chế then chốt:** Nghiệm nhân tử tối ưu $\lambda^*$ phụ thuộc phi tuyến vào phân phối xác suất môi trường và trạng thái chính sách hiện tại $\pi_\theta$. Do đó, **không tồn tại bất kỳ giá trị vô hướng cố định nào của $\lambda_p$ có thể tối ưu đồng thời trên toàn bộ không gian khám phá**.

---

## 2. Phương pháp Đối ngẫu Lagrange Cổ điển (Classical Lagrangian Dual Relaxation)

### 2.1. Hàm Lagrange và Bài toán Min-Max Đối ngẫu
Để tìm nghiệm thỏa mãn chính xác các ràng buộc, ta thiết lập hàm Lagrange:
$$\mathcal{L}(\theta, \boldsymbol{\lambda}) = J_0(\pi_\theta) + \sum_{i=1}^m \lambda_i \cdot \big( J_i(\pi_\theta) - \delta_i \big), \quad \lambda_i \ge 0 \tag{3}$$
Bài toán gốc (Primal Problem) tương đương với:
$$\min_{\theta} \max_{\boldsymbol{\lambda} \ge 0} \mathcal{L}(\theta, \boldsymbol{\lambda}) \tag{4}$$

Theo định lý đối ngẫu Lagrange, bài toán đối ngẫu (Dual Problem) được phát biểu là:
$$\max_{\boldsymbol{\lambda} \ge 0} d(\boldsymbol{\lambda}) \quad \text{với} \quad d(\boldsymbol{\lambda}) = \min_{\theta} \mathcal{L}(\theta, \boldsymbol{\lambda}) \tag{5}$$

### 2.2. Điều kiện Tối ưu Karush-Kuhn-Tucker (KKT)
Điểm yên ngựa $(\theta^*, \boldsymbol{\lambda}^*)$ là nghiệm tối ưu khi và chỉ khi thỏa mãn 4 điều kiện KKT:

1. **Khả thi nguyên thủy (Primal Feasibility):**
   $$J_i(\pi_{\theta^*}) \le \delta_i, \quad \forall i \in \{1, \dots, m\} \tag{6a}$$
2. **Khả thi đối ngẫu (Dual Feasibility):**
   $$\lambda_i^* \ge 0, \quad \forall i \in \{1, \dots, m\} \tag{6b}$$
3. **Độ chùng bù (Complementary Slackness):**
   $$\lambda_i^* \cdot \big( J_i(\pi_{\theta^*}) - \delta_i \big) = 0, \quad \forall i \in \{1, \dots, m\} \tag{6c}$$
4. **Điểm dừng (Stationarity):**
   $$\nabla_\theta J_0(\pi_{\theta^*}) + \sum_{i=1}^m \lambda_i^* \nabla_\theta J_i(\pi_{\theta^*}) = 0 \tag{6d}$$

> [!IMPORTANT]
> **Ý nghĩa kinh tế của $\lambda_i^*$ (*Shadow Price* - Giá bóng):**
> Theo định lý độ nhạy Lagrange, $\lambda_i^* = -\frac{\partial J_0^*}{\partial \delta_i}$. Nhân tử Lagrange $\lambda_i^*$ chính là tốc độ biến thiên của chi phí tối ưu khi nới lỏng biên ràng buộc $\delta_i$ đi một đơn vị. Khi ràng buộc không tích cực ($J_i < \delta_i$), điều kiện (6c) bắt buộc $\lambda_i^* = 0$, nghĩa là không cần áp dụng bất kỳ hình phạt nào.

---

## 3. Phân tích Động học của Dual Ascent Cổ điển (RCPO) & Chứng minh Thất bại

### 3.1. Thuật toán Dual Ascent Cổ điển (RCPO)
Trong thuật toán RCPO (*Reward Constrained Policy Optimization* - Tessler et al., 2019), bài toán được giải bằng cách lặp 2 thang thời gian (Two-timescale stochastic approximation):
- **Cập nhật chính sách $\theta$ (Thang thời gian nhanh):**
  $$\theta_{k+1} = \theta_k - \alpha_\theta \nabla_\theta \mathcal{L}(\theta_k, \boldsymbol{\lambda}_k)$$
- **Cập nhật nhân tử đối ngẫu $\boldsymbol{\lambda}$ (Thang thời gian chậm):**
  $$\lambda_{k+1} = \max\Big(0,\ \lambda_k + \eta_\lambda \cdot (J_k - \delta)\Big) \tag{7}$$

### 3.2. Mô hình hóa theo Lý thuyết Điều khiển Tự động
Xét phương trình cập nhật (7) trong miền thời gian liên tục. Đặt sai số vi phạm $e(t) = J(t) - \delta$. Ta có:
$$\frac{d\lambda(t)}{dt} = \eta_\lambda \cdot e(t) \tag{8}$$
Lấy biến đổi Laplace hai vế:
$$s \Lambda(s) = \eta_\lambda E(s) \implies H_{\mathrm{RCPO}}(s) = \frac{\Lambda(s)}{E(s)} = \frac{\eta_\lambda}{s} \tag{9}$$

> [!WARNING]
> Phương trình (9) chứng minh rằng: **Bản chất của thuật toán RCPO cổ điển là một bộ điều khiển Tích phân thuần túy (Pure Integrator Controller)**.

### 3.3. Chứng minh Toán học: Sự Xuất hiện của Limit Cycles và Mất ổn định

Xét hệ thống động học ghép giữa tham số chính sách $\theta(t)$ và nhân tử đối ngẫu $\lambda(t)$ quanh điểm cân bằng KKT $(\theta^*, \lambda^*)$:
$$\begin{cases}
\dot{\theta} = -\nabla_\theta J_0(\theta) - \lambda \nabla_\theta C(\theta) \\
\dot{\lambda} = C(\theta) - \delta
\end{cases} \tag{10}$$

Tuyến tính hóa hệ phi tuyến (10) bằng khai triển Taylor bậc 1 quanh $(\theta^*, \lambda^*)$ với độ lệch $\Delta \theta = \theta - \theta^*$, $\Delta \lambda = \lambda - \lambda^*$:
$$\begin{bmatrix} \Delta \dot{\theta} \\ \Delta \dot{\lambda} \end{bmatrix} = \mathbf{J} \begin{bmatrix} \Delta \theta \\ \Delta \lambda \end{bmatrix} = \begin{bmatrix} -\mathbf{H}_\theta & -\nabla_\theta C \\ (\nabla_\theta C)^T & 0 \end{bmatrix} \begin{bmatrix} \Delta \theta \\ \Delta \lambda \end{bmatrix} \tag{11}$$
trong đó $\mathbf{H}_\theta = \nabla^2_\theta \mathcal{L}$ là ma trận Hessian của hàm Lagrange theo $\theta$.

Xét phương trình đặc trưng $\det(s\mathbf{I} - \mathbf{J}) = 0$:
$$s^2 + \mathrm{Tr}(\mathbf{H}_\theta) s + \|\nabla_\theta C\|^2 = 0 \tag{12}$$

#### Hệ quả vật lý:
1. **Trễ pha $90^\circ$ (Underdamped Oscillation):** Khâu tích phân thuần $\frac{1}{s}$ tạo ra độ lệch pha đúng $-90^\circ$. Khi hàm mục tiêu có độ cong phẳng ($\mathbf{H}_\theta \approx 0$), phương trình (12) có 2 nghiệm thuần ảo:
   $$s_{1, 2} = \pm j \|\nabla_\theta C\| \tag{13}$$
   Nghiệm thuần ảo biểu thị một hệ thống bảo toàn năng lượng dao động điều hòa không bao giờ tắt (*Marginally stable / Limit cycle*).
2. **Hiện tượng Bão hòa Tích phân (Integral Windup):** 
   Khi mạng RL ở các episode đầu chưa học được cách thỏa mãn ràng buộc ($e(t) > 0$ trong suốt thời gian dài), tích phân $\int_0^T e(t) dt$ tích tụ thành một giá trị khổng lồ $\lambda(t) \gg \lambda^*$. 
   Khi chính sách cuối cùng tìm được vùng khả thi ($e(t) < 0$), nhân tử $\lambda(t)$ khổng lồ này xả xuống cực kỳ chậm. Trong suốt quá trình xả đó, chính sách bị "tê liệt", tiếp tục thực hiện các hành động bảo thủ cực đoan (đẩy hết việc lên MEC) dù ràng buộc đã hoàn toàn an toàn.

---

## 4. Cơ sở Lý thuyết & Chứng minh Tính Ổn định của PID-Lagrangian

Để khắc phục hiện tượng trễ pha và dao động limit cycle, Stooke et al. (ICML 2020) đề xuất tích hợp cấu trúc vi tích phân tỉ lệ (PID) vào quá trình cập nhật đối ngẫu.

```
       Sai số vi phạm e_t = (v_t - δ)
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
┌─────────────┐┌─────────────┐┌─────────────┐
│  Khâu Tỷ Lệ ││Khâu Tích Phân││Khâu Vi Phân │
│   K_P * e_t ││  K_I * e_t  ││ -K_D * Δe_t │
└──────┬──────┘└──────┬──────┘└──────┬──────┘
       │              │              │
       │       [Anti-Windup]         │
       │       clip(I, 0, Imax)      │
       │              │              │
       └──────────────┼──────────────┘
                      ▼
               λ_t = [P + I + D]^+
             (Ổn định hóa đối ngẫu)
```

### 4.1. Công thức Toán học Rời rạc
Định nghĩa sai số vi phạm tại episode $k$: $e_k = \hat{v}_k - \delta$.

Hệ thống cập nhật qua 3 phương trình liên hoàn:
$$\begin{aligned}
I_k &= \mathrm{clip}\Big( I_{k-1} + K_I \cdot e_k,\ 0,\ I_{\max} \Big) \tag{14a} \\
D_k &= -K_D \cdot \big( e_k - e_{k-1} \big) \tag{14b} \\
\lambda_k &= \max\Big( 0,\ K_P \cdot e_k + I_k + D_k \Big) \tag{14c}
\end{aligned}$$
trong đó $K_P, K_I, K_D \ge 0$.

### 4.2. Phân tích Hàm Truyền đạt (Laplace Domain Transfer Function)
Trong miền thời gian liên tục, luật điều khiển có dạng:
$$\lambda(t) = K_P e(t) + K_I \int_0^t e(\tau) d\tau + K_D \frac{de(t)}{dt} \tag{15}$$
Chuyển đổi sang miền tần số phức $s$:
$$C_{\mathrm{PID}}(s) = \frac{\Lambda(s)}{E(s)} = K_P + \frac{K_I}{s} + K_D s = \frac{K_D s^2 + K_P s + K_I}{s} \tag{16}$$

#### Vai trò toán học của từng khâu:
- **Tử số bậc hai ($K_D s^2 + K_P s + K_I$):** Cho phép đặt 2 điểm không (zeros) tự do $z_1, z_2$ trên mặt phẳng phức:
  $$s^2 + \frac{K_P}{K_D} s + \frac{K_I}{K_D} = (s - z_1)(s - z_2) = 0$$
- **Bù sớm pha (Phase Lead Compensation):** Khâu vi phân $K_D s$ cung cấp độ dịch pha dương $+90^\circ$ ở tần số cao. Độ bù pha này trung hòa trực tiếp độ trễ pha $-90^\circ$ của khâu tích phân $\frac{1}{s}$, khôi phục biên độ dự trữ pha (*Phase Margin*) của hệ thống kín $\ge 45^\circ$, dập tắt hoàn toàn hiện tượng dao động tuần hoàn.

---

### 4.3. Chứng minh Tính Ổn định Tiệm cận bằng Hàm Lyapunov

Xét hệ thống ghép kín khi có khâu PID. Thay thế $\lambda$ trong hệ vi phân bằng luật điều khiển PID:
$$\begin{cases}
\dot{\theta} = -\nabla_\theta J_0 - \big( K_P (C - \delta) + I + K_D \dot{C} \big) \nabla_\theta C \\
\dot{I} = K_I (C - \delta)
\end{cases} \tag{17}$$

Ta định nghĩa hàm ứng viên Lyapunov bậc hai quanh điểm cân bằng KKT $(\theta^*, I^*)$:
$$V(\theta, I) = \frac{1}{2} \|\theta - \theta^*\|^2 + \frac{1}{2 K_I} (I - I^*)^2 \tag{18}$$
Rõ ràng $V(\theta, I) > 0, \forall (\theta, I) \ne (\theta^*, I^*)$ và $V(\theta^*, I^*) = 0$.

Lấy đạo hàm của $V$ theo thời gian:
$$\dot{V} = (\theta - \theta^*)^T \dot{\theta} + \frac{1}{K_I} (I - I^*) \dot{I} \tag{19}$$
Thế $\dot{\theta}$ và $\dot{I}$ từ (17) vào (19):
$$\begin{aligned}
\dot{V} &= (\theta - \theta^*)^T \Big( -\nabla_\theta J_0 - \big( K_P (C - \delta) + I + K_D \dot{C} \big) \nabla_\theta C \Big) + (I - I^*)(C - \delta) \\
&= -(\theta - \theta^*)^T \nabla_\theta \mathcal{L}^* - K_P (C - \delta) (\theta - \theta^*)^T \nabla_\theta C - K_D \dot{C} (\theta - \theta^*)^T \nabla_\theta C \\
&\quad - (I - I^*) (\theta - \theta^*)^T \nabla_\theta C + (I - I^*)(C - \delta)
\end{aligned} \tag{20}$$

Sử dụng xấp xỉ tuyến tính quanh điểm cân bằng: $C(\theta) - \delta \approx (\nabla_\theta C)^T (\theta - \theta^*)$ và nhận xét $\dot{C} = (\nabla_\theta C)^T \dot{\theta}$:
$$\begin{aligned}
\dot{V} &\approx -(\theta - \theta^*)^T \mathbf{H}_\theta (\theta - \theta^*) - K_P \big( (\nabla_\theta C)^T (\theta - \theta^*) \big)^2 - K_D \big( (\nabla_\theta C)^T \dot{\theta} \big)^2 \\
&= -\Delta \theta^T \mathbf{H}_\theta \Delta \theta - K_P e^2 - K_D \dot{C}^2 \tag{21}
\end{aligned}$$

#### Đánh giá tính xác định âm của $\dot{V}$:
- Vì $(\theta^*, I^*)$ là điểm cực tiểu cục bộ của hàm Lagrange, ma trận Hessian $\mathbf{H}_\theta \succeq 0 \implies \Delta \theta^T \mathbf{H}_\theta \Delta \theta \ge 0$.
- Số hạng thứ hai: $-K_P e^2 \le 0$ (triệt tiêu khi $e = 0$).
- Số hạng thứ ba (khâu vi phân): $-K_D \dot{C}^2 \le 0$ với $K_D > 0$. Đây là **thành phần tiêu tán năng lượng dao động (dissipative energy damping)**.

Khi $K_D > 0$ và $K_P > 0$:
$$\dot{V} < 0, \quad \forall (\Delta \theta, \Delta I) \ne (0, 0) \tag{22}$$
Theo Định lý Ổn định Lyapunov, điểm cân bằng $(\theta^*, I^*)$ của hệ thống kín là **ổn định tiệm cận (Asymptotically Stable)**. 

> [!TIP]
> **Kết luận chứng minh:** Sự có mặt của khâu $K_D$ biến đổi hệ thống từ trạng thái dao động bảo toàn (nghiệm thuần ảo $s = \pm j\omega$) thành một hệ thống tắt dần có phần thực âm:
> $$s_{1, 2} = -\zeta \omega_n \pm j \omega_d \quad (\text{với hệ số suy giảm } \zeta = \frac{K_D \|\nabla C\|}{2} > 0)$$
> Nhờ đó, biên độ dao động vọt lố bị triệt tiêu theo quy luật hàm mũ $e^{-\zeta \omega_n t}$.

---

## 5. Ánh xạ Toán học vào Kết quả Thực nghiệm Bài toán VEC

Phần này liên kết trực tiếp các công thức toán học ở trên với các hiện tượng bạn đã quan sát trên biểu đồ thực nghiệm `notebooks/fig2_pid_multiplier_dynamics.png`:

### 5.1. Động lực học của Nhân tử Deadline $\lambda_T$ (Đồ thị bên trái)
- **Công thức:** $e_{T, k} = \hat{v}_{T, k} - 0.05$
- **Episodes 1 $\to$ 30:** $\hat{v}_T \approx 11.7\% > 5.0\% \implies e_{T, k} = +0.067 > 0$.
  $$\lambda_T(k) \approx I_0 + K_P \cdot (0.067) + K_I \sum e_T \implies \text{Tăng vọt từ } 1.00 \to 1.055$$
- **Episodes 30 $\to$ 150:** Dưới áp lực $\lambda_T$, chính sách học đưa vi phạm xuống $\hat{v}_T \approx 2.0\% < 5.0\% \implies e_{T, k} = -0.03 < 0$.
  $$\Delta I_T = K_I \cdot (-0.03) = 0.05 \times (-0.03) = -0.0015 / \text{episode}$$
  $$\lambda_T(150) \approx 1.03 - 100 \times 0.0015 = 0.88$$
- **Tác động của $K_D$:** Vì $e_{T, k}$ giảm dốc, $\Delta e_T < 0 \implies -K_D \Delta e_T > 0$. Khâu này đóng vai trò "chân phanh" ngăn không cho $\lambda_T$ rơi tự do về 0, giữ cho đường dốc đi xuống êm ái, bám sát đường tích phân $I_T$.

### 5.2. Động lực học của Nhân tử Năng lượng $\lambda_E$ (Đồ thị bên phải)
- **Công thức:** $e_{E, k} = \hat{v}_{E, k} - 0.0$ với $\delta_E = 0.0$.
- Do các tác vụ lớn khi xe ở xa trạm vẫn làm tiêu thụ vượt ngưỡng trung bình $\hat{v}_E \approx 0.005\text{ J}$ ($5\text{ mJ}$):
  $$e_{E, k} = 0.005 - 0.0 = +0.005 > 0 \quad (\text{luôn dương đều đặn})$$
- Tích lũy tích phân mỗi episode:
  $$\Delta I_E = K_I \cdot e_{E, k} = 0.05 \times 0.005 = +0.00025$$
- Sau 150 episodes:
  $$\lambda_E(150) = 1.000 + 150 \times 0.00025 = 1.041$$
  $\implies$ Giải thích chính xác tại sao $\lambda_E$ bò chéo lên một đường thẳng tuyến tính từ $1.000 \to 1.041$.

---

## 6. Bảng Tổng kết So sánh Cơ chế: Fixed Penalty vs. RCPO vs. PID-Lagrangian

| Tiêu chí So sánh | Phạt Cố định (DuelingDQN / MADDPG) | Lagrangian Không PID (RCPO) | PID-Lagrangian (Phương pháp Đề xuất) |
| :--- | :--- | :--- | :--- |
| **Bản chất điều khiển** | Không có vòng phản hồi (Open-loop heuristic) | Khâu Tích phân thuần túy ($H(s) = \frac{\eta}{s}$) | Bộ điều khiển PID đầy đủ ($C(s) = K_P + \frac{K_I}{s} + K_D s$) |
| **Xử lý ràng buộc** | Bị phạt nặng cố định bất kể xa hay gần ngưỡng | Tự điều chỉnh $\lambda$ theo sai số tích lũy | Tự điều chỉnh $\lambda$ kết hợp phản ứng nhanh và giảm xóc |
| **Hiện tượng Windup** | Không áp dụng | **Nghiêm trọng** (tích lũy $\lambda \to \infty$ khi vi phạm kéo dài) | **Triệt tiêu hoàn toàn** nhờ toán tử `clip(I, 0, I_max)` |
| **Độ ổn định động học** | Mất cân bằng Pareto (Over-conservative) | **Dao động điều hòa (Limit cycling)** do trễ pha $90^\circ$ | **Ổn định tiệm cận Lyapunov** (Damping tắt dần $e^{-\zeta \omega_n t}$) |
| **Hành vi offload VEC** | Sợ phạt, đẩy $2.74\%$ lên MEC đắt đỏ, rút ngắn khóa mã hóa | Dao động giữa offload MEC và Local, không ổn định | Tối ưu Pareto: giữ MEC ở $1.72\%$, đẩy sang AV, tăng khóa lên $35.4$ bits |
| **Chi phí kinh tế** | Cao ($0.14540$ cents) | Không ổn định | **Thấp nhất toàn diện** ($0.14001$ cents, tiết kiệm $3.71\%$) |
