"""
MADDPG cho VEC đa user — Lowe et al., "Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments", NeurIPS 2017.

Thiết kế (ghi rõ trong bài):
- Mỗi user là một agent (MultiAgentVehicularEdgeEnv, event-driven). Huấn luyện tập trung,
  thực thi phân tán: actor π_i(o_i) chỉ thấy quan sát cục bộ; critic Q_i(o_1..o_N, a_1..a_N).
- Action rời rạc MultiDiscrete (α_l, target, resource[, N_i]) → actor xuất logits theo từng
  chiều, dùng Gumbel-Softmax (straight-through, hard) để đạo hàm qua action rời rạc
  (cách MADDPG gốc xử lý kênh truyền thông rời rạc).
- Chia sẻ tham số giữa các agent đồng nhất (actor & critic nhận thêm one-hot id agent).
- Action masking (T1.6) phân rã theo chiều: target không khả thi bị chặn; không có target nào
  khả thi → chỉ cho α_l = 1 (local).
- Semi-MDP (T1.4): chiết khấu γ^(Δt/dt).
"""
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import Logger


def _mlp(inp: int, out: int, hidden: int) -> nn.Module:
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, out))


def head_masks_from_joint(mask: np.ndarray, nvec: np.ndarray) -> np.ndarray:
    """Mask joint (C-order trên nvec) → mask nối các chiều (Σ nvec,), bool."""
    parts = [np.ones(n, dtype=bool) for n in nvec]
    if mask is None:
        return np.concatenate(parts)
    M = np.asarray(mask, dtype=bool).reshape(tuple(nvec))
    target_ok = M[:-1, :].reshape(nvec[0] - 1, nvec[1], -1).any(axis=(0, 2))
    if target_ok.any():
        parts[1] = target_ok
    else:
        parts[0] = np.zeros(nvec[0], dtype=bool)
        parts[0][-1] = True
    return np.concatenate(parts)


class _Buffer:
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, act_dim: int):
        self.cap, self.ptr, self.size = capacity, 0, 0
        shp = lambda *d: (capacity, *d)
        self.o = np.zeros(shp(n_agents, obs_dim), np.float32)
        self.o2 = np.zeros(shp(n_agents, obs_dim), np.float32)
        self.a = np.zeros(shp(n_agents, act_dim), np.float32)
        self.m = np.zeros(shp(n_agents, act_dim), bool)
        self.m2 = np.zeros(shp(n_agents, act_dim), bool)
        self.r = np.zeros(shp(n_agents), np.float32)
        self.acted = np.zeros(shp(n_agents), np.float32)
        self.has2 = np.zeros(shp(n_agents), np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.el = np.zeros(capacity, np.float32)

    def add(self, **kw):
        for k, v in kw.items():
            getattr(self, k)[self.ptr] = v
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n: int, device):
        idx = np.random.randint(0, self.size, size=n)
        names = ["o", "o2", "a", "m", "m2", "r", "acted", "has2", "done", "el"]
        return {k: torch.as_tensor(getattr(self, k)[idx], device=device) for k in names}


class MADDPGAgent:
    def __init__(self, obs_dim: int, action_dim, n_agents: int,
                 actor_lr: float = 1e-4, critic_lr: float = 1e-3, gamma: float = 0.95, tau: float = 0.01,
                 batch_size: int = 64, buffer_size: int = 100000, learning_starts: int = 2000,
                 hidden_size: int = 256, gumbel_temperature: float = 1.0,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, exploration_fraction: float = 0.3,
                 total_training_steps: int = 500000, base_dt: float = 0.1, grad_clip: float = 1.0, train_freq: int = 1,
                 device: str = "cpu", **_ignored):
        self.nvec = np.array(action_dim, dtype=np.int64)
        self.splits = [int(n) for n in self.nvec]
        self.act_dim = int(self.nvec.sum())
        self.N, self.obs_dim = int(n_agents), int(obs_dim)
        self.gamma, self.tau, self.batch_size = gamma, tau, batch_size
        self.learning_starts = max(int(learning_starts), batch_size)
        self.temp, self.grad_clip, self.base_dt = gumbel_temperature, grad_clip, float(base_dt)
        self.train_freq = max(1, int(train_freq))   # cập nhật mạng mỗi train_freq bước (giảm thời gian train)
        self.epsilon, self.epsilon_start, self.epsilon_end = epsilon_start, epsilon_start, epsilon_end
        self.decay_steps = max(1, int(exploration_fraction * total_training_steps))
        self.device = torch.device("cpu" if str(device).startswith("cuda") and not torch.cuda.is_available() else device)
        self.logger = Logger("MADDPGAgent")

        actor_in = self.obs_dim + self.N
        critic_in = self.N * self.obs_dim + self.N * self.act_dim + self.N
        self.actor = _mlp(actor_in, self.act_dim, hidden_size).to(self.device)
        self.critic = _mlp(critic_in, 1, hidden_size).to(self.device)
        self.actor_t = _mlp(actor_in, self.act_dim, hidden_size).to(self.device)
        self.critic_t = _mlp(critic_in, 1, hidden_size).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = _Buffer(buffer_size, self.N, self.obs_dim, self.act_dim)
        self.ids = torch.eye(self.N, device=self.device)
        self.step_count = 0
        self.episode_count = 0
        self.logger.info(f"MADDPG: {self.N} agents, obs={obs_dim}, action heads={self.splits}")

    # ------------------------------------------------------------------ acting
    def _logits(self, net, obs: torch.Tensor) -> torch.Tensor:
        ids = self.ids.expand(*obs.shape[:-2], self.N, self.N)
        return net(torch.cat([obs, ids], dim=-1))

    def _masked_onehot(self, logits: torch.Tensor, mask: torch.Tensor, gumbel: bool) -> torch.Tensor:
        logits = logits.masked_fill(~mask, -1e9)
        outs = []
        for lg in torch.split(logits, self.splits, dim=-1):
            if gumbel:
                outs.append(F.gumbel_softmax(lg, tau=self.temp, hard=True, dim=-1))
            else:
                outs.append(F.one_hot(lg.argmax(-1), lg.shape[-1]).float())
        return torch.cat(outs, dim=-1)

    def _onehot_to_action(self, onehot: np.ndarray) -> np.ndarray:
        return np.array([int(p.argmax()) for p in np.split(onehot, np.cumsum(self.splits)[:-1])], dtype=np.int64)

    def _action_to_onehot(self, action: np.ndarray) -> np.ndarray:
        return np.concatenate([np.eye(n, dtype=np.float32)[int(a)] for n, a in zip(self.splits, action)])

    def select_actions(self, obs: np.ndarray, head_masks: np.ndarray, training: bool = True) -> np.ndarray:
        """obs [N,d], head_masks [N,A] → one-hot [N,A]."""
        if training and np.random.rand() < self.epsilon:
            out = np.zeros((self.N, self.act_dim), np.float32)
            for i in range(self.N):
                parts = np.split(head_masks[i], np.cumsum(self.splits)[:-1])
                a = [np.random.choice(np.where(p)[0]) for p in parts]
                out[i] = self._action_to_onehot(np.array(a))
            return out
        with torch.no_grad():
            o = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            m = torch.as_tensor(head_masks, device=self.device).unsqueeze(0)
            return self._masked_onehot(self._logits(self.actor, o), m, gumbel=training)[0].cpu().numpy()

    # ------------------------------------------------------------------ learning
    def _critic_q(self, net, obs: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        """obs [B,N,d], acts [B,K,N,A] (K bộ action, thường K=N: thay action agent i) → Q [B,K]."""
        B, K = acts.shape[0], acts.shape[1]
        g = obs.reshape(B, 1, -1).expand(B, K, -1)
        ids = self.ids[:K].expand(B, K, self.N)
        return net(torch.cat([g, acts.reshape(B, K, -1), ids], dim=-1)).squeeze(-1)

    def train_step(self) -> float:
        if self.buffer.size < self.learning_starts:
            return 0.0
        b = self.buffer.sample(self.batch_size, self.device)
        acted_sum = b["acted"].sum().clamp(min=1.0)
        with torch.no_grad():
            a2 = self._masked_onehot(self._logits(self.actor_t, b["o2"]), b["m2"], gumbel=False)
            a2 = a2 * b["has2"].unsqueeze(-1)
            q2 = self._critic_q(self.critic_t, b["o2"], a2.unsqueeze(1).expand(-1, self.N, -1, -1))
            disc = self.gamma ** (b["el"] / self.base_dt)
            y = b["r"] + (disc * (1.0 - b["done"])).unsqueeze(-1) * q2
        q = self._critic_q(self.critic, b["o"], b["a"].unsqueeze(1).expand(-1, self.N, -1, -1))
        critic_loss = (b["acted"] * F.smooth_l1_loss(q, y, reduction="none")).sum() / acted_sum
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()

        logits = self._logits(self.actor, b["o"])
        a_pi = self._masked_onehot(logits, b["m"], gumbel=True)
        eye = self.ids.view(1, self.N, self.N, 1)
        joint = b["a"].unsqueeze(1) * (1 - eye) + a_pi.unsqueeze(1) * eye   # hàng i: agent i dùng π
        q_pi = self._critic_q(self.critic, b["o"], joint)
        reg = (logits.masked_fill(~b["m"], 0.0) ** 2).mean()
        actor_loss = -(b["acted"] * q_pi).sum() / acted_sum + 1e-3 * reg
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        for net, tgt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
            for p, pt in zip(net.parameters(), tgt.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return float(critic_loss.item())

    # ------------------------------------------------------------------ episodes
    def _obs_masks(self, ma_env, obs_dict, info_dict):
        obs = np.stack([obs_dict[a] for a in ma_env.agents]).astype(np.float32)
        masks = np.stack([head_masks_from_joint(info_dict[a]["action_mask"], self.nvec) for a in ma_env.agents])
        return obs, masks

    def _run_episode(self, ma_env, max_steps: int, training: bool) -> dict:
        obs_dict, info_dict = ma_env.reset()
        obs, masks = self._obs_masks(ma_env, obs_dict, info_dict)
        has = np.array([ma_env.has_task(i) for i in range(self.N)], np.float32)
        ep_reward, losses, step = 0.0, [], 0
        for step in range(max_steps):
            onehot = self.select_actions(obs, masks, training=training)
            actions = {a: self._onehot_to_action(onehot[i]) for i, a in enumerate(ma_env.agents) if has[i]}
            obs2_dict, rew, _, trunc, info2 = ma_env.step(actions)
            obs2, masks2 = self._obs_masks(ma_env, obs2_dict, info2)
            acted = np.array([info2[a]["acted"] for a in ma_env.agents], np.float32)
            has2 = np.array([info2[a]["has_task"] for a in ma_env.agents], np.float32)
            r = np.array([rew[a] for a in ma_env.agents], np.float32)
            done = bool(trunc["__all__"])
            ep_reward += float(np.sum([info2[a]["raw_reward"] for a in ma_env.agents]))
            if training:
                self.buffer.add(o=obs, o2=obs2, a=onehot * acted[:, None], m=masks, m2=masks2, r=r,
                                acted=acted, has2=has2, done=float(done), el=info2["__common__"]["elapsed_time"])
                if self.step_count % self.train_freq == 0:
                    losses.append(self.train_step())
                self.step_count += 1
                frac = min(1.0, self.step_count / self.decay_steps)
                self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
            obs, masks, has = obs2, masks2, has2
            if done:
                break
        m = ma_env.env.get_metrics()
        return {"total_reward": ep_reward, "episode_reward": ep_reward, "avg_loss": float(np.mean(losses)) if losses else 0.0,
                "epsilon": self.epsilon, "steps": step + 1,
                "viol_rate_T": m.get("deadline_violation_rate", 0.0), **m}

    def train_episode(self, ma_env, max_steps: int = 1000) -> dict:
        self.episode_count += 1
        return {"episode": self.episode_count, **self._run_episode(ma_env, max_steps, training=True)}

    def evaluate_episode(self, ma_env, max_steps: int = 1000) -> dict:
        return self._run_episode(ma_env, max_steps, training=False)

    def save_model(self, filepath: str):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        ck = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
