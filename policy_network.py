import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
THROTTLE_RESIDUAL_LIMIT = 0.03
STEER_RESIDUAL_LIMIT = 0.12


def build_mlp(input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    """
    SAC actor network.

    Input:
        - observation: current state + reference state + error + context
    Output:
        - throttle residual in [-0.06, 0.06]
        - steering residual in [-0.25, 0.25]
    """

    def __init__(self, obs_dim: int, action_dim: int = 2, hidden_dims: Tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        self.backbone = build_mlp(obs_dim, hidden_dims, hidden_dims[-1])
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        self.obs_dim = obs_dim

    def _apply_action_constraints(self, action: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        del observation
        throttle = action[..., :1]
        steering = action[..., 1:]
        constrained_throttle = THROTTLE_RESIDUAL_LIMIT * throttle
        constrained_steering = STEER_RESIDUAL_LIMIT * steering
        return torch.cat([constrained_throttle, constrained_steering], dim=-1)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(observation)
        mean = self.mean_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(observation)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        squashed_action = torch.tanh(raw_action)
        constrained_action = self._apply_action_constraints(squashed_action, observation)

        log_prob = normal.log_prob(raw_action)
        log_prob -= torch.log(1.0 - squashed_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        deterministic_action = self._apply_action_constraints(torch.tanh(mean), observation)
        return constrained_action, log_prob, deterministic_action

    @torch.no_grad()
    def act(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        action, _, deterministic_action = self.sample(observation)
        chosen = deterministic_action if deterministic else action
        return chosen.squeeze(0)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 2, hidden_dims: Tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim + action_dim, hidden_dims, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([observation, action], dim=-1))


class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 2, hidden_dims: Tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dims)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(observation, action), self.q2(observation, action)


@dataclass
class SACConfig:
    obs_dim: int
    action_dim: int = 2
    hidden_dims: Tuple[int, ...] = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 1e-3
    target_entropy: float = -2.0


class SACAgent:
    def __init__(self, config: SACConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

        self.actor = PolicyNetwork(config.obs_dim, config.action_dim, config.hidden_dims).to(device)
        self.critic = DoubleQCritic(config.obs_dim, config.action_dim, config.hidden_dims).to(device)
        self.critic_target = DoubleQCritic(config.obs_dim, config.action_dim, config.hidden_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.log_alpha = torch.tensor(math.log(0.1), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, observation, deterministic: bool = False):
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        return self.actor.act(obs_tensor, deterministic=deterministic).cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target_value = rewards + (1.0 - dones) * self.config.gamma * target_q

        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        sampled_actions, log_prob, _ = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.config.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.mul_(1.0 - self.config.tau).add_(self.config.tau * param.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "q1_mean": float(current_q1.mean().item()),
            "q2_mean": float(current_q2.mean().item()),
            "log_prob": float(log_prob.mean().item()),
        }

    def behavior_clone_loss(self, observations: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        observations = observations.to(self.device)
        target_actions = target_actions.to(self.device)
        mean, _ = self.actor(observations)
        pred_actions = self.actor._apply_action_constraints(torch.tanh(mean), observations)
        return F.mse_loss(pred_actions, target_actions)

    def actor_state_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "config": self.config.__dict__,
        }

    def load_actor_state_dict(self, payload: Dict[str, object]) -> None:
        actor_state = payload["actor"] if "actor" in payload else payload
        self.actor.load_state_dict(actor_state, strict=True)

    def state_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "config": self.config.__dict__,
        }

    def load_state_dict(self, payload: Dict[str, object]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
