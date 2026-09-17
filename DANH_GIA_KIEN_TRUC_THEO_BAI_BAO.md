# Đánh giá kiến trúc dự án so với hai bài báo gốc

> **17/09/2026:** Các đề xuất đã được thực hiện, xem `BAO_CAO_SUA_KIEN_TRUC.md`. Đính chính §3.2: phương án (a), tức cấu trúc của [B], **không** sửa được thiên lệch, vì trong [B] local cũng nhận −ϖ₂T ≤ 0. Dự án đã dùng phương án (b).

**Ngày:** 17/09/2026 · Đọc toàn văn:
- **[A]** Xue, Yao, Wang, *Ad Hoc Networks* 161 (2024) 103545
- **[B]** Xiao et al., *IEEE TWC* 22(12), 2023

Mọi số liệu trong tài liệu này đo bằng `scripts/calibrate_energy.py`: thử mọi action trên 200–300 state.

---

## 1. Tóm tắt kiến trúc hai bài

| | [A] CSO: kinh tế + Homomorphic Encryption | [B]: độ mạnh bảo mật + tài nguyên |
|---|---|---|
| Thực thể | M user, N xe phụ (AUX), J MEC. Mô phỏng: 1 BS, 10 user, 3 AUX (dùng 1–4 user cho DRL) | I xe trong vùng phủ 1 edge server; attacker là mật độ ρ trên các xe |
| Offload | **Từng phần** α = (α_l, α_j, α_n), hoặc MEC hoặc AUX (23b–g) | **Nhị phân** x_i ∈ {0,1}, chỉ tới edge |
| Chọn đích | **Heuristic Algorithm 1**: trong vùng phủ thì MEC, ngoài vùng thì AUX rẻ nhất (không học) | Không có AUX |
| Mã hóa | Paillier HE, 800 cycles/bit, chống **cả** MEC tò mò lẫn kẻ nghe lén (§3.3) | Mã hóa khối độ dài N_i, 90 cycles/bit, chống kẻ nghe lén; S = (N_i/N_f)^{nᵉ} |
| Mục tiêu | min Σ C_m (tiền: năng lượng quy đổi + giá dịch vụ động), Eq(22) | max-min O_i = (1−x)ϖ₁U_i − xϖ₂T_i^c, Eq(14–15) |
| Ràng buộc | Delay (23e), Σβ ≤ 1, f ≤ f_max | Delay ≤ thời gian lưu trú τ_i (C2), **E ≤ 0.4 J (C3)**, 0 ≤ f^c, f^en ≤ F^loc (C4, **DVFS**), Σf^e ≤ F^e |
| Lời giải | **DDRP = Dueling DQN tập trung**: state = [s_1..s_M; t^w_1..t^w_M], action = [α_1..α_M; β_1..β_M], reward = −Σ C, có penalty khi vi phạm (chỉ nhánh MEC). Nhánh AUX: **CMSO** (lồi, CVX) với f* = α_l·s·c/t (Eq 34) | **GBD** (không học): bài toán primal lồi (KKT) + master (x, N_i); N_i* dạng đóng Eq(39) |
| Tham số chính | xe 0.5 GHz, MEC 1 GHz, c = 1000, c_en = 800, s = 100–1000 kb, v = 15–25 m/s, L = 300 m | F^loc = 2 GHz (tối đa), F^e = 3 GHz, q^c = 600, **q^e = 100**, q^en = 90, D = 100–900 KB, κ = 1e-27, E = 0.4 J, v ~ TruncGauss(13, 5) trên [2, 24], L = 100 m, R = α√p |

## 2. Những gì dự án làm đúng / có cơ sở
- Công thức [A] Eq(1–2, 6–13, 22) và [B] Eq(6–9) được cài đúng. Riêng `baselines/gbd.py` khớp chính xác Eq(39). (Báo cáo Phase 2 ghi sai công thức là `floor(1/√nᵉ)`, nhưng code thì đúng.)
- Giá AUX: dùng ε_n = tải + 1, tránh trường hợp g_n = 0. Đây là mở rộng hợp lý.
- DVFS vừa thêm khớp với **cả hai bài**: [B] có ràng buộc C4, còn [A] Eq(34) cũng chọn f* = α_l·s·c/t (tần số nhỏ nhất kịp deadline).
- Semi-MDP, action masking, deadline động theo mobility, Lagrangian: đều là mở rộng hợp lý so với penalty cố định của [A].

## 3. Điểm lệch / vấn đề kiến trúc (xếp theo mức nghiêm trọng)

### 3.1 [Nghiêm trọng] Hai lớp mã hóa bị thừa về mặt mô hình an ninh
[A] §3.3 nói rõ: Paillier HE bảo vệ **cả trước kẻ nghe lén** ("even if the data is eavesdropped on during transmission, it is impossible to access the plaintext").

Dự án (T0.3) chồng thêm mã hóa khối của [B], rồi tính S = (N_i/N_f)^{nᵉ} như thể việc bẻ khóa lớp ngoài làm lộ dữ liệu. Nhưng dữ liệu bên trong vẫn là ciphertext Paillier. Phản biện sẽ hỏi: **"vậy U và N_i còn đo được điều gì?"**

Ngoài ra, `c_en(N_i) = 890·(1 + 0.1·log₂(N_i/8))` nhân cả phần Paillier (800) theo N_i. Chi phí Paillier không phụ thuộc độ dài khối của cipher kia. Hơn nữa [B] không có hệ số này: q^en = 90 là hằng số, và đánh đổi của N_i trong [B] nằm sẵn trong U (log₂(8N)/(8N) giảm khi N tăng, còn S tăng).

**Đề xuất** (chọn một):
- **(a) Phân vai theo đích (khuyến nghị).** Nhánh MEC dùng Paillier HE, vì server là bên không tin cậy ([A]). Nhánh AUX/V2V dùng mã hóa khối N_i với mô hình attacker của [B], vì đó là kênh V2V đa bước qua xe khác. Lựa chọn đích lúc đó mang đánh đổi thật: MEC an toàn tuyệt đối nhưng đắt (800 cycles/bit, ciphertext phồng 2×), AUX rẻ nhưng rủi ro theo mật độ attacker. Hai bài báo ghép vào nhau có ý nghĩa.
- (b) Giữ cả hai lớp, nhưng định nghĩa rò rỉ = bẻ được lớp ngoài **và** lộ khóa HE. Khi đó S chỉ là metadata/integrity. Khó thuyết phục hơn.
- Trong mọi trường hợp, bỏ việc nhân chi phí Paillier theo N_i.

### 3.2 [Nghiêm trọng] Hàm mục tiêu thưởng offload bằng ϖ₁·U, không thưởng gì cho local
Reward = −(C − ϖ₁U). U > 0 chỉ khi có truyền dữ liệu, nên xử lý local (không lộ gì, an toàn tuyệt đối) lại nhận "an toàn = 0".

Trong [B], hai nhánh cân bằng nhau: local được −ϖ₂T^c. Dự án bỏ ϖ₂T (đưa delay thành ràng buộc) mà không thay thế.

Số đo (cents/task): full-offload tốn **0.18** so với local **0.02**, nhưng được thưởng ϖ₁U = **0.055**. Khi offload khả dụng, 53% trường hợp action tốt nhất là offload, chủ yếu nhờ khoản thưởng này. Offload **không bao giờ tiết kiệm năng lượng cho xe**: mã hóa 890 cycles/bit gần bằng tính local 1000 cycles/bit.

**Đề xuất:**
- (a) Khôi phục cấu trúc của [B]: O = (1−α_l)·ϖ₁U − α_l·ϖ₂T_local, rồi cộng vào −C.
- (b) Hoặc thay phần thưởng bằng **chi phí rủi ro**: phạt ϖ·P(lộ)·dữ liệu offload, với P(lộ) = 1 − S. Local có P(lộ) = 0 nên công bằng.
- (c) Hoặc đưa bảo mật thành ràng buộc S ≥ S_min, dùng thêm một multiplier Lagrangian (hợp với phương pháp đề xuất).

### 3.3 [Nghiêm trọng] Trộn tham số làm edge chậm hơn xe
Hiện tại: xe **2 GHz** (F^loc, là mức *tối đa* của [B]), MEC **1 GHz** ([A]), AUX **0.5 GHz** ([A]), c = 1000 cho cả local lẫn edge.
- Trong [A], edge nhanh gấp 2 lần xe.
- Trong [B], edge **hiệu quả gấp ~18 lần**: F^e = 3 GHz và q^e = 100 so với q^c = 600.

Trong dự án, offload không nhanh hơn, cũng không tiết kiệm hơn. Lý do offload chỉ còn là khoản thưởng ở §3.2.

| Cấu hình | Khả thi | Khi offload khả dụng thì best = offload |
|---|---|---|
| Hiện tại (xe 2, MEC 1, AUX 0.5 GHz) | 99% | 53% |
| xe 1, MEC 5, AUX 2 GHz | 95% | **74%** |
| Theo [A] đúng nghĩa (xe 0.5 GHz, τ = 1 s) | **46%** | – |

Với τ_app = 1 s, task 1 Mbit ở 1000 cycles/bit không thể xong đúng hạn ở đâu cả (MEC 1 GHz cần ≥ 1 s chỉ để tính). [A] không công bố giá trị t_m; τ_app = 1 s là lựa chọn của dự án.

**Đề xuất:** chọn một chế độ nhất quán và ghi nguồn cho từng tham số. Ví dụ: xe F^loc = 1 GHz (DVFS), MEC = 3–5 GHz ([B] F^e = 3), AUX = 1–2 GHz, τ_app theo task (hoặc theo thời gian lưu trú như [B] Eq(2)). Sau đó chạy lại `calibrate_energy.py` để kiểm tra tính khả thi và tỷ lệ offload.

### 3.4 [Trung bình] Giá dịch vụ: ω vượt miền của [A]
[A] định nghĩa ω_m ∈ (0,1). Dự án dùng ω = min(1/τ, 3), nên khi τ < 1 s thì exp(ω) lên tới e³ ≈ 20, trong khi [A] tối đa e ≈ 2.7.

**Đề xuất:** ω = τ_min/τ ∈ (0, 1], hoặc chuẩn hóa về (0, 1). Khi đó bỏ được cơ chế chặn `urgency_cap`.

### 3.5 [Trung bình] Năng lượng nhánh AUX tính sai hệ số so với [A] Eq(22)
[A]: λⁿ nhân với **năng lượng của user** khi offload tới AUX (Eⁿ_m = mã hóa + upload + chờ, Eq 20).

Dự án: năng lượng user ở mọi nhánh đều nhân λˡ, còn λⁿ nhân với **năng lượng tính toán của chính AUX** (κ·c·s·f_AV²), một đại lượng [A] không có. Nếu giữ thì phải ghi là mở rộng, kèm lý do (bù năng lượng cho AUX).

### 3.6 [Trung bình] Mô hình thời gian: mã hóa và local
[A] Eq(21): t = max{t_l, t_en + t_up + t_exe}. Mã hóa chỉ cộng vào nhánh remote; local chạy song song, dù hai việc dùng chung một CPU f_m. Đây là thiếu sót vật lý của [A].

Dự án (sau khi sửa DVFS): t = t_en + max(t_local, T_remote). Mã hóa và local chạy tuần tự trên một CPU, chặt chẽ hơn. Cần ghi rõ là khác [A].

### 3.7 [Trung bình] Thuật toán đề xuất không học phần bảo mật
- N_i chưa có trong action space (cờ `use_4d_action` không được dùng).
- Công suất phát p cố định 0.2 W, trong khi [B] tối ưu p.
- Tần số CPU không phải biến học (hiện do DVFS dạng đóng quyết định). Cách này ổn, nhưng nên nói rõ "DRL quyết định (α, đích, β, N_i); tần số giải bằng KKT", tức kiến trúc lai DRL + tối ưu lồi. Kiến trúc lai này chính là tinh thần của [A] (DDRP + CMSO) và [B] (GBD).

### 3.8 [Nhỏ] So sánh với DDRP của [A]
- DDRP điều khiển **tập trung mọi user** (action gồm α và β của cả M user, M ≤ 4).
- Dự án: DQN chỉ điều khiển user 0; 9 user còn lại chạy Algorithm 1.
- MADDPG: phân tán cả 10 user.

Nếu gọi DuelingDQN là "baseline [A]" thì cần một biến thể trung thành hơn: DDRP tập trung với M nhỏ (2–4 user), hoặc ghi rõ đây là "DDRP-style single-agent".

[B] tối ưu max-min giữa các xe; dự án có `reward_mode="max_min"` trong env đa agent nhưng chưa dùng trong thí nghiệm chính.

### 3.9 [Nhỏ] Kênh truyền và mobility
- **Tốc độ truyền:** [B] dùng mô hình TCP R = α√p, dự án dùng Shannon của [A]. Hợp lý; chỉ cần ghi rõ U tính trên tốc độ Shannon.
- **Vận tốc synthetic:** [A] U(15, 25); [B] TruncGauss(13, 5) trên [2, 24] m/s. Dữ liệu Rio thật có trung bình 4.4 m/s, gần [B] hơn. Nên dùng phân phối của [B] cho synthetic để khoảng cách sim-to-real công bằng hơn.
- **Hình học:** offload chỉ khả dụng ở **34%** state (user phân bố đều trong 1 km², vùng phủ MEC 300 m, tầm V2V 200 m). Với dữ liệu Rio, MEC đặt ở hotspot nên tỷ lệ này sẽ cao hơn.

## 4. Thứ tự đề xuất trước khi train lại
1. **Quyết định mô hình an ninh** (§3.1) và **hàm mục tiêu** (§3.2). Hai việc này thay đổi reward nên phải chốt trước.
2. **Chốt bộ tham số nhất quán** (§3.3, §3.4, §3.5), ghi nguồn trong `params_master.yaml`, rồi chạy lại `calibrate_energy.py`. Mục tiêu: khả thi ≥ 90%, và khi offload khả dụng thì offload tốt nhất ở phần đáng kể các state.
3. **Bật N_i trong action space** (§3.7).
4. Sau đó mới train lại (xem `BAO_CAO_HUAN_LUYEN_LAI.md`).

## 5. Ghi chú khi chạy trên Colab
- Colab miễn phí chỉ có **2 vCPU**. GPU không giúp (mạng nhỏ; nút thắt nằm ở vòng lặp env Python). Dùng runtime CPU và `--jobs 2`.
- Ước lượng cả lưới 18 run với `--jobs 2`: 12.7 h; thêm `--maddpg-train-freq 4`: 6.5 h; thêm `--episodes 200`: **2.6 h**. Colab ngắt phiên sau vài giờ, nên:
  - đặt `--out-root` trên Google Drive (`/content/drive/MyDrive/...`). Launcher bỏ qua run đã có `agent_model.pth`, nên chạy lại lệnh là tiếp tục được;
  - hoặc chia lưới ra nhiều notebook/phiên chạy song song, mỗi phiên một `--agents` hoặc `--seeds`.
- Cần upload `data/processed/riobus/riobus_{train,test}.parquet` (68 MB, đã nằm trong `.gitignore`), hoặc chạy lại `preprocess_riobus.py` từ dữ liệu thô.
