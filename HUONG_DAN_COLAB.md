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
# (4) Chạy thử 1 seed để chọn số episode (xem ước lượng trước bằng --dry-run)
RUNS = f'{DRIVE}/runs'
!python scripts/run_experiments.py --dry-run --seeds 0 --episodes 200 --maddpg-train-freq 4 --jobs 2 --speed-factor 1.8 --out-root {RUNS}
!nohup python scripts/run_experiments.py --seeds 0 --episodes 200 --maddpg-train-freq 4 --jobs 2 --out-root {RUNS} > {DRIVE}/launcher_seed0.log 2>&1 &
```

```python
# Theo dõi tiến độ (chạy lại ô này bất cứ lúc nào)
!tail -n 20 {DRIVE}/launcher_seed0.log
!ls {RUNS}/*/*/seed0/agent_model.pth 2>/dev/null | wc -l     # 6 = xong
!tail -n 3 {RUNS}/lagrangian_dqn/real/seed0/train.log
```

```python
# (5) Đường hội tụ → chọn số episode
!python scripts/plot_convergence.py --runs-dir {RUNS} --out {DRIVE}/convergence_seed0.png
from IPython.display import Image; Image(f'{DRIVE}/convergence_seed0.png')
```

Nếu `objective` và `viol_T` đã phẳng từ khoảng episode 120–150 thì giữ 200 episode. Nếu vẫn còn giảm ở episode 200:
- chạy lại với `--episodes 300` (hoặc hơn);
- **đổi `--out-root`** sang thư mục mới, vì launcher bỏ qua run đã có model mà không kiểm tra số episode.

```python
# (6) Đủ 3 seed — seed 0 đã xong sẽ được bỏ qua (giữ nguyên --episodes và --out-root)
!nohup python scripts/run_experiments.py --seeds 0 1 2 --episodes 200 --maddpg-train-freq 4 --jobs 2 --out-root {RUNS} > {DRIVE}/launcher_all.log 2>&1 &
```

```python
# (7) Đánh giá chung trên synthetic và Rio-test (ngày khác tập train) từ model đã lưu — không train lại
for a in ['lagrangian_dqn', 'dueling_dqn', 'maddpg']:
    !python scripts/sim2real_study.py --runs-dir {RUNS} --agent {a} --seeds 0 1 2 --eval-episodes 10 --output_dir {DRIVE}/eval 2>&1 | grep -A8 "Transfer matrix"
```

Kết quả nằm trong `{DRIVE}/eval/`:
- `transfer_matrix_<agent>.csv`: trung bình ± CI95 theo seed, gồm objective (cents/task), viol_T, viol_E, offload_ratio, aux_share, mec_share, block_length_aux, cost_security;
- `transfer_raw_<agent>.csv`: số liệu thô từng episode;
- `distribution_gap.csv`: KS vận tốc và thời gian lưu trú.

**So sánh các thuật toán bằng `objective_cents` + `viol_T` + `viol_E`, không dùng `total_reward` trong log train.** Reward huấn luyện của LagrangianDQN không trừ penalty, còn baseline thì có.

## 3. Thời gian dự kiến

Đo trên máy lab rồi nhân 1.8 cho CPU Colab. Đây là ước lượng; con số thật in ra ở `--dry-run`.

| Cấu hình (2 tiến trình) | Ước lượng |
|---|---|
| 1 seed × 3 thuật toán × 2 miền, 200 episode, MADDPG `train_freq=4` | **~2 h** |
| 3 seed, 200 episode, `train_freq=4` | ~6 h |
| 3 seed, chỉ miền Rio (`--domains real`) | ~3.8 h |
| 3 seed, 500 episode, `train_freq=4` | ~15 h |
| 3 seed, 500 episode, MADDPG `train_freq=1` | ~26 h |

## 4. Mẹo với Colab

- **Phiên bị ngắt** (tối đa ~12 h, hoặc ~90 phút nếu không tương tác): chạy lại ô (2), sau đó chạy lại lệnh launcher y hệt. Run đã xong sẽ được bỏ qua; run đang dở sẽ train lại từ đầu, vì chưa có checkpoint giữa chừng.
- **Chạy song song nhiều phiên** (nhiều tài khoản hoặc nhiều notebook): chia theo thuật toán, dùng chung `--out-root` trên cùng Drive.
  - Phiên A: `--agents lagrangian_dqn`
  - Phiên B: `--agents dueling_dqn`
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
