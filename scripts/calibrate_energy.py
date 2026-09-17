"""
Hiệu chỉnh mô hình (năng lượng, năng lực tính toán, giá rủi ro rò rỉ) — kiểm tra bài toán
KHÔNG suy biến trước khi huấn luyện.

Với mỗi cấu hình, lấy mẫu state quyết định thật, thử MỌI action bằng env.evaluate_action
(dry-run) và đo:
  feasible_any      : tỉ lệ state có ≥1 action thỏa cả deadline lẫn E_max
  feasible_T_any    : tỉ lệ state có ≥1 action kịp deadline (bỏ qua năng lượng)
  remote_available  : tỉ lệ state có thể offload (trong vùng phủ MEC / tầm AUX)
  best_offload|avail: trong các state offload được, action khả thi rẻ nhất có offload không
  best_target_aux   : ... trong đó chọn AUX (so với MEC)
  mec_viol_E        : mọi action offload MEC đều vi phạm E_max (ràng buộc năng lượng "cắn")
  best_N_i          : phân bố N_i của action tốt nhất khi đi AUX (đánh đổi an ninh có nghĩa?)

  python scripts/calibrate_energy.py --states 300
"""
import sys
import argparse
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig


def evaluate_config(n_states: int = 200, seed: int = 0, **cfg_kwargs) -> pd.DataFrame:
    cfg = EnvironmentConfig(**cfg_kwargs)
    env = VehicularEdgeEnvironment(cfg)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    nvec = tuple(int(n) for n in env.action_space.nvec)
    L = nvec[0]
    rows = []
    for _ in range(n_states):
        mask = env.action_mask()
        M = mask.reshape(nvec)
        feasible_idx = np.where(mask)[0]
        # resource levels: lấy thưa để nhanh
        res_keep = {0, nvec[2] // 3, 2 * nvec[2] // 3, nvec[2] - 1}
        results = []
        for idx in feasible_idx:
            a = np.unravel_index(idx, nvec)
            if a[2] not in res_keep and a[0] != L - 1:
                continue
            m = env.evaluate_action(np.array(a))
            results.append((a, m))
        ok = [(a, m) for a, m in results if not m["viol_T"] and m["viol_E"] <= 0]
        okT = [(a, m) for a, m in results if not m["viol_T"]]
        remote = bool(M[:-1].any())
        best = min(ok, key=lambda r: r[1]["cost_total"]) if ok else None
        local = [m for a, m in results if a[0] == L - 1][0]
        mec = [m for a, m in results if a[0] < L - 1 and a[1] == 0]
        row = {
            "feasible_any": bool(ok), "feasible_T_any": bool(okT), "remote_available": remote,
            "local_viol_T": local["viol_T"], "local_viol_E": local["viol_E"] > 0,
            "mec_viol_E": all(m["viol_E"] > 0 for m in mec) if mec else np.nan,
            "best_offload": (best[0][0] < L - 1) if (best and remote) else np.nan,
            "best_target_aux": (best[0][1] == 1) if (best and remote and best[0][0] < L - 1) else np.nan,
            "best_N_i": (cfg.block_length_options[best[0][3]] if (best and len(nvec) == 4 and best[0][0] < L - 1
                                                                 and best[0][1] == 1) else np.nan),
            "cost_best": best[1]["cost_total"] if best else np.nan,
            "cost_security_best": best[1]["cost_security"] if best else np.nan,
        }
        rows.append(row)
        idx = rng.choice(feasible_idx)
        env.step(np.array(np.unravel_index(idx, nvec)))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    out = {k: round(float(df[k].astype(float).mean()), 3) for k in
           ["feasible_any", "feasible_T_any", "remote_available", "local_viol_T", "mec_viol_E",
            "best_offload", "best_target_aux"]}
    out["best_N_i"] = dict(sorted(Counter(df["best_N_i"].dropna().astype(int)).items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=200)
    ap.add_argument("--out", default="outputs/calibrate_model.csv")
    args = ap.parse_args()
    GHz = 1e9
    grid = [
        ("mặc định (params_master)", {}),
        ("cũ: xe 2 / MEC 1 / AUX 0.5 GHz, κ=1e-28", dict(user_cpu_frequency=2 * GHz, mec_cpu_frequency=1 * GHz,
                                                        aux_cpu_frequency=0.5 * GHz, kappa=1e-28)),
        ("κ = 1e-27 ([B])", dict(kappa=1e-27)),
        ("attacker kỳ vọng (nᵉ hằng)", dict(attacker_model="expected")),
        ("leakage_price = 0", dict(leakage_price=0.0)),
        ("leakage_price = 1e-6", dict(leakage_price=1e-6)),
    ]
    rows = []
    for name, g in grid:
        row = {"config": name, **summarize(evaluate_config(args.states, **g))}
        print(row, flush=True)
        rows.append(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
