"""
Homomorphic Encryption (HE / Paillier) vs Symmetric Block Cipher (AES) Full Comparison (Phase 3 - Task T3.4).

Replicates and extends Paper [A] Fig. 9:
1. Compares Paillier (HE) vs AES vs No-Encryption across task sizes (100 KB to 1 MB).
2. Evaluates the impact of Paillier ciphertext expansion (2.0x payload) on transmission delay & energy.
3. Identifies the task size / network bandwidth threshold where HE advantage reverses.
"""
import sys
import argparse
import csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig
from src.baselines.heuristic import GreedyPolicy, AllMECPolicy
from src.utils.logger import Logger


def run_crypto_benchmark(crypto_scheme: str, ciphertext_expansion: float, task_sizes: list, num_episodes: int = 5) -> list:
    results = []
    policy = AllMECPolicy(resource_level=4)

    for s_val in task_sizes:
        cfg = EnvironmentConfig(
            task_size_min=s_val,
            task_size_max=s_val,
            episode_length=30,
            encryption_enabled=(crypto_scheme != "none")
        )
        # Dynamic monkey-patch for Phase 3 crypto scheme evaluation
        cfg.crypto_scheme = crypto_scheme
        cfg.ciphertext_expansion = ciphertext_expansion

        env = VehicularEdgeEnvironment(cfg)
        ep_delays, ep_energies, ep_costs = [], [], []

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=42 + ep)
            for _ in range(30):
                action = policy.select_action(obs, env=env)
                next_obs, r, term, trunc, info = env.step(action)
                if "task_delay" in info:
                    ep_delays.append(info["task_delay"])
                if "task_energy" in info:
                    ep_energies.append(info["task_energy"])
                if "task_cost" in info:
                    ep_costs.append(info["task_cost"])
                obs = next_obs
                if term or trunc:
                    break

        results.append({
            "crypto_scheme": crypto_scheme,
            "ciphertext_expansion": ciphertext_expansion,
            "task_size_bits": s_val,
            "task_size_kb": s_val / 8000.0,
            "avg_delay": float(np.mean(ep_delays)) if ep_delays else 0.0,
            "avg_energy": float(np.mean(ep_energies)) if ep_energies else 0.0,
            "avg_cost": float(np.mean(ep_costs)) if ep_costs else 0.0,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="HE vs AES Cryptographic Scheme Benchmark")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    logger = Logger("CryptoComparison")
    logger.info("Starting HE (Paillier) vs AES Cryptographic Benchmark...")

    task_sizes = [100000, 300000, 500000, 800000, 1000000]

    # Benchmark 1: Paillier without expansion (Paper [A] baseline assumption, expansion = 1.0)
    res_paillier_ideal = run_crypto_benchmark("paillier", 1.0, task_sizes)
    # Benchmark 2: Paillier realistic with 2.0x ciphertext expansion (mod h -> mod h^2)
    res_paillier_real = run_crypto_benchmark("paillier", 2.0, task_sizes)
    # Benchmark 3: AES (q_en = 90 cycles/bit, expansion = 1.0, dec_cycles = 90)
    res_aes = run_crypto_benchmark("aes", 1.0, task_sizes)
    # Benchmark 4: No encryption baseline
    res_none = run_crypto_benchmark("none", 1.0, task_sizes)

    all_results = res_paillier_ideal + res_paillier_real + res_aes + res_none

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_file = output_path / "crypto_comparison.csv"

    fieldnames = list(all_results[0].keys())
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    logger.info(f"Results saved to {csv_file}")
    logger.info("=" * 80)
    logger.info("Crypto Comparison Completed Successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
