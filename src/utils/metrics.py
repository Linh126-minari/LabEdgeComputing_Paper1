"""
Evaluation Metrics Collector (Phase 2 - T2.6).

Collects and aggregates episode metrics:
- Economic cost (cents)
- Secure information capacity U
- Execution delay (s) and p95 delay
- Energy consumption (J)
- Deadline violation rate & energy violation rate
- Link-loss failure rate
- Jain's fairness index
"""
from typing import Dict, List, Any
import numpy as np


class MetricsCollector:
    """Collects per-step and per-episode metrics for evaluation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.episode_rewards: List[float] = []
        self.episode_delays: List[float] = []
        self.episode_energies: List[float] = []
        self.episode_costs: List[float] = []
        self.episode_sec_caps: List[float] = []
        self.deadline_violations: List[bool] = []
        self.energy_violations: List[bool] = []
        self.link_loss_failures: List[bool] = []
        self.decisions_count = 0

    def record_step(self, reward: float, info: Dict[str, Any]):
        """Record metrics from env step info dictionary."""
        self.episode_rewards.append(reward)
        self.decisions_count += 1

        delay = info.get("task_delay", info.get("delay"))
        if delay is not None:
            self.episode_delays.append(float(delay))

        energy = info.get("task_energy", info.get("total_energy"))
        if energy is not None:
            self.episode_energies.append(float(energy))

        cost = info.get("task_cost", info.get("cost"))
        if cost is not None:
            self.episode_costs.append(float(cost))

        sec_cap = info.get("task_secure_capacity", info.get("secure_capacity"))
        if sec_cap is not None:
            self.episode_sec_caps.append(float(sec_cap))

        if "deadline_violated" in info:
            self.deadline_violations.append(bool(info["deadline_violated"]))
        if "energy_violation" in info:
            self.energy_violations.append(bool(info["energy_violation"]))
        if "link_limited" in info or "failure_cause" in info:
            link_fail = info.get("link_limited", False) or (info.get("failure_cause") == "link_loss")
            self.link_loss_failures.append(bool(link_fail))

    def compute_summary(self) -> Dict[str, float]:
        """Compute aggregated episode metrics."""
        return {
            "total_reward": float(np.sum(self.episode_rewards)) if self.episode_rewards else 0.0,
            "avg_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "avg_delay": float(np.mean(self.episode_delays)) if self.episode_delays else 0.0,
            "p95_delay": float(np.percentile(self.episode_delays, 95)) if self.episode_delays else 0.0,
            "avg_energy": float(np.mean(self.episode_energies)) if self.episode_energies else 0.0,
            "avg_cost": float(np.mean(self.episode_costs)) if self.episode_costs else 0.0,
            "avg_sec_capacity": float(np.mean(self.episode_sec_caps)) if self.episode_sec_caps else 0.0,
            "deadline_violation_rate": float(np.mean(self.deadline_violations)) if self.deadline_violations else 0.0,
            "energy_violation_rate": float(np.mean(self.energy_violations)) if self.energy_violations else 0.0,
            "link_loss_failure_rate": float(np.mean(self.link_loss_failures)) if self.link_loss_failures else 0.0,
            "decisions_count": self.decisions_count,
        }

    @staticmethod
    def jains_fairness_index(values: List[float]) -> float:
        """Compute Jain's Fairness Index: (sum(x)^2) / (n * sum(x^2))."""
        arr = np.array(values, dtype=np.float64)
        if len(arr) == 0 or np.sum(arr ** 2) == 0:
            return 1.0
        return float((np.sum(arr) ** 2) / (len(arr) * np.sum(arr ** 2)))
