# Hướng dẫn huấn luyện trên Google Colab

Notebook mẫu (chạy lần lượt từng ô): `notebooks/colab_train.ipynb`. Tài liệu này giải thích từng bước.

## 0. Chuẩn bị trên máy lab (làm một lần)

1. **Push code lên GitHub.** Colab lấy code bằng `git clone`, mà các thay đổi hiện chưa commit.
   ```bash
   git add -A && git commit -m "Role-based security, DVFS, calibrated model, MADDPG, Colab pipeline"
   git push origin main
   ```
2. **Upload dữ liệu Rio lên Google Drive.** Các file parquet nằm trong `.gitignore`. Gói dữ liệu đã được tạo sẵn:
   `outputs/colab/riobus_processed.tgz` (55 MB, gồm `riobus_train.parquet`, `riobus_test.parquet`, `riobus_stats.json`).
   Upload vào `MyDrive/vec_paper1/`.
   Nếu cần tạo lại gói: `tar czf riobus_processed.tgz data/processed/riobus/riobus_{train,test}.parquet data/processed/riobus/riobus_stats.json`.

## 1. Runtime

**Runtime → Change runtime type → CPU.** GPU không giúp:
- mạng MLP rất nhỏ;
- thời gian chủ yếu nằm ở vòng lặp môi trường Python.

Colab miễn phí có **2 vCPU**, nên dùng `--jobs 2`. Kiểm tra bằng `!nproc`. Colab Pro có thể nhiều core hơn; khi đó tăng `--jobs` bằng số core.

## 2. Các ô lệnh

```python
# (1) Mount Drive — mọi kết quả ghi lên Drive để không mất khi phiên bị ngắt
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/vec_paper1'
!mkdir -p {DRIVE}
```

```python
# (2) Code + thư viện + dữ liệu
%cd /content
!test -d LabEdgeComputing_Paper1 || git clone https://github.com/Linh126-minari/LabEdgeComputing_Paper1.git
%cd /content/LabEdgeComputing_Paper1
!git pull
!pip install -q -r requirements.txt
!tar xzf {DRIVE}/riobus_processed.tgz
!ls -la data/processed/riobus/
```

```python
# (3) Kiểm tra nhanh (~20 s) — phải thấy "101 passed"
!python -m pytest tests/ -q 2>&1 | tail -3
```

```python
# (4) Chạy thử 1 seed để kiểm tra hội tụ (xem ước lượng trước bằng --dry-run)
RUNS = f'{DRIVE}/runs'
!python scripts/run_experiments.py --dry-run --agents lagrangian_dqn dueling_dqn --seeds 0 --episodes 150 --jobs 2 --speed-factor 1.8 --out-root {RUNS}
!nohup python scripts/run_experiments.py --agents lagrangian_dqn dueling_dqn --seeds 0 --episodes 150 --jobs 2 --out-root {RUNS} > {DRIVE}/launcher_seed0.log 2>&1 &
```

```python
# Theo dõi tiến độ (chạy lại ô này bất cứ lúc nào)
!tail -n 20 {DRIVE}/launcher_seed0.log
!ls {RUNS}/*/*/seed0/agent_model.pth 2>/dev/null | wc -l     # 4 = xong (2 thuật toán x 2 miền)
!tail -n 3 {RUNS}/lagrangian_dqn/real/seed0/train.log
```

```python
# (5) Đủ 3 seed — seed 0 đã xong sẽ được bỏ qua (giữ nguyên --episodes và --out-root)
!nohup python scripts/run_experiments.py --agents lagrangian_dqn dueling_dqn --seeds 0 1 2 --episodes 150 --jobs 2 --out-root {RUNS} > {DRIVE}/launcher_all.log 2>&1 &
```

```python
# (6) Đánh giá chung (synthetic và Rio test) từ model đã lưu — không train lại
EVAL = f'{DRIVE}/eval'
for a in ['lagrangian_dqn', 'dueling_dqn']:
    !python scripts/sim2real_study.py --runs-dir {RUNS} --agent {a} --seeds 0 1 2 --eval-episodes 10 --output_dir {EVAL} 2>&1 | grep -A8 "Transfer matrix"
```

```python
# (7) Xuất toàn bộ biểu đồ so sánh chuẩn bài báo khoa học (DPI 300)
PLOTS = f'{DRIVE}/plots'
!python scripts/plot_comparison.py --runs-dir {RUNS} --eval-dir {EVAL} --out-dir {PLOTS}
```

Kết quả biểu đồ nằm trong `{DRIVE}/plots/`:
- `fig1_convergence_comparison.png`: Tốc độ hội tụ Objective & Vi phạm ràng buộc (Synthetic vs Rio).
- `fig2_pid_multiplier_dynamics.png`: Động thái nhân tử PID $\lambda_T, \lambda_E$ (chứng minh dập tắt dao động và chống vọt lố).
- `fig3_sim2real_transfer.png`: Biểu đồ cột ma trận chuyển giao Sim-to-Real giữa 2 thuật toán.

## 3. Thời gian dự kiến (Đã lược bỏ MADDPG)

| Cấu hình (2 tiến trình Colab) | Ước lượng |
|---|---|
| 1 seed × 2 thuật toán × 2 miền, 150 episode | **~45 phút** |
| 3 seed, 150 episode (toàn bộ lưới 12 runs) | **~2.2 giờ** |
| Đánh giá Sim-to-Real (2 thuật toán) | ~5 phút |
| Vẽ toàn bộ biểu đồ | ~30 giây |

## 4. Mẹo với Colab

- **Phiên bị ngắt**: Chạy lại ô mount Drive và clone code, sau đó chạy lại lệnh launcher y hệt. Run đã xong (`agent_model.pth`) sẽ được tự động bỏ qua; run dở dang sẽ tiếp tục.
- **Chia phiên nếu muốn nhanh hơn nữa**:
  - Phiên 1: `--agents lagrangian_dqn` (~1.1 giờ)
  - Phiên 2: `--agents dueling_dqn` (~1.1 giờ)
  - Phiên C: `--agents maddpg`
- **Không đặt `--jobs` lớn hơn số core** (`!nproc`): các tiến trình sẽ tranh CPU và chậm hơn.
- RAM: mỗi run trên dữ liệu Rio dùng ~0.6 GB; 2 tiến trình vừa sức 12 GB của Colab.
- Muốn thử nhanh trước khi chạy dài: thêm `--episodes 5 --eval-interval 5 --eval-episodes 1 --out-root {DRIVE}/smoke`.

## 5. Tham số thường chỉnh (thêm qua `--extra`)

| Mục đích | Override |
|---|---|
| Ràng buộc năng lượng chặt hơn | `--extra env.constraints.energy_budget_max=0.2` |
| Tắt N_i trong action (ablation T3.1) | `--extra agent.hyperparameters.use_4d_action=false` |
| Attacker kỳ vọng thay vì attacker thật | `--extra env.attack.model=expected` |
| Mã khối cho cả MEC (so sánh HE vs AES) | `--extra security.encryption.scheme=aes` |
| Fairness max-min cho MADDPG ([B]) | `--extra agent.hyperparameters.reward_mode=max_min` |

Khi dùng `--extra` cho một ablation, **đặt `--out-root` riêng**, để không lẫn với run chính.
