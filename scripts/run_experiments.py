"""
Chạy lưới huấn luyện (thuật toán × miền dữ liệu × seed) SONG SONG trên nhiều core CPU.

Mỗi run là một tiến trình `train.py` độc lập, 1 luồng torch (OMP_NUM_THREADS=1) — với mạng
MLP nhỏ, chạy N tiến trình 1 luồng nhanh hơn nhiều so với 1 tiến trình N luồng.
Run đã có agent_model.pth sẽ được bỏ qua → chạy lại lệnh là tiếp tục lưới dang dở.

  python scripts/run_experiments.py --dry-run                     # xem kế hoạch + ước lượng thời gian
  python scripts/run_experiments.py --jobs 10                     # chạy toàn bộ
  python scripts/run_experiments.py --agents lagrangian_dqn --domains real --seeds 0 --episodes 200
Sau đó: python scripts/sim2real_study.py --runs-dir outputs/runs   (đánh giá, không train lại)
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {
    "synthetic": ["env=scenario_default"],
    "real": ["env=scenario_riobus", "env.mobility.trace_file=data/processed/riobus/riobus_train.parquet"],
}
# (giây train, giây eval) / episode 1000 bước, 1 luồng CPU — đo 17/09/2026 trên máy lab (x86, 12 core).
# CPU Colab thường chậm hơn ~1.5-2x → dùng --speed-factor 1.8 khi ước lượng cho Colab.
SEC_PER_EPISODE = {("dqn", "synthetic"): (5.1, 1.6), ("dqn", "real"): (7.3, 3.5),
                   ("maddpg", "synthetic"): (22.0, 1.0), ("maddpg", "real"): (22.2, 1.3)}


def run_dir(root: Path, agent: str, domain: str, seed: int) -> Path:
    return root / agent / domain / f"seed{seed}"


def estimate(agent, domain, episodes, eval_interval, eval_episodes, train_freq, speed_factor=1.0):
    kind = "maddpg" if agent == "maddpg" else "dqn"
    train_s, eval_s = SEC_PER_EPISODE[(kind, domain)]
    if kind == "maddpg" and train_freq > 1:
        train_s *= 0.1 + 0.9 / train_freq        # đo: train_freq=4 → 6.6 s thay vì 22 s
    evals = episodes // max(eval_interval, 1) * eval_episodes
    return speed_factor * (episodes * train_s + evals * eval_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["lagrangian_dqn", "dueling_dqn"],
                    help="Danh sách thuật toán huấn luyện (mặc định: lagrangian_dqn và dueling_dqn)")
    ap.add_argument("--domains", nargs="+", default=["synthetic", "real"], choices=list(DOMAINS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--eval-interval", type=int, default=25)
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--maddpg-train-freq", type=int, default=4)
    # mỗi run trace ~0.6 GB RAM riêng (+ thư viện torch dùng chung) → mặc định tối đa 8 tiến trình
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--out-root", default="outputs/runs")
    ap.add_argument("--extra", nargs="*", default=[], help="override Hydra thêm, vd env.episode_length=500")
    ap.add_argument("--speed-factor", type=float, default=1.0, help="hệ số tốc độ CPU cho ước lượng (Colab ≈ 1.8)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = ROOT / args.out_root
    jobs = []
    for agent, domain, seed in itertools.product(args.agents, args.domains, args.seeds):
        d = run_dir(root, agent, domain, seed)
        est = estimate(agent, domain, args.episodes, args.eval_interval, args.eval_episodes, args.maddpg_train_freq,
                       args.speed_factor)
        cmd = [sys.executable, "train.py", f"agent={agent}", *DOMAINS[domain],
               f"general.random_seed={seed}", f"general.num_episodes={args.episodes}", "general.device=cpu",
               f"general.eval_interval={args.eval_interval}", f"general.eval_episodes={args.eval_episodes}",
               f"general.project_name={d.relative_to(ROOT / 'outputs')}", f"hydra.run.dir={d}/hydra", *args.extra]
        if agent == "maddpg":
            cmd.append(f"agent.hyperparameters.train_freq={args.maddpg_train_freq}")
        done = (d / "agent_model.pth").exists()
        jobs.append({"agent": agent, "domain": domain, "seed": seed, "dir": d, "cmd": cmd, "est": est, "done": done})

    todo = sorted([j for j in jobs if not j["done"]], key=lambda j: -j["est"])   # dài trước → xếp lịch tốt hơn
    # ước lượng wall-clock: xếp tham lam vào `jobs` làn
    lanes = [0.0] * args.jobs
    for j in todo:
        k = lanes.index(min(lanes)); lanes[k] += j["est"]
    total = sum(j["est"] for j in todo)
    print(f"{len(jobs)} run, {len(jobs) - len(todo)} đã xong, {len(todo)} cần chạy | {args.jobs} tiến trình song song")
    print(f"Ước lượng: tuần tự {total / 3600:.1f} h → song song {max(lanes) / 3600:.1f} h")
    for j in todo:
        print(f"  {j['agent']:15s} {j['domain']:9s} seed{j['seed']}  ~{j['est'] / 60:5.0f} phút")
    if args.dry_run or not todo:
        return

    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run(j):
        j["dir"].mkdir(parents=True, exist_ok=True)
        t = time.time()
        with open(j["dir"] / "train.log", "w") as log:
            rc = subprocess.run(j["cmd"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        return j, rc, time.time() - t

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for fut in as_completed([pool.submit(run, j) for j in todo]):
            j, rc, dt = fut.result()
            status = "OK " if rc == 0 else f"LỖI rc={rc}"
            print(f"[{time.strftime('%H:%M:%S')}] {status} {j['agent']} {j['domain']} seed{j['seed']} "
                  f"({dt / 60:.0f} phút) → {j['dir'] / 'train.log'}", flush=True)


if __name__ == "__main__":
    main()
