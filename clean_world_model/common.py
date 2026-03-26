import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


STATE_NAMES = [
    "pos_x",
    "pos_y",
    "pos_z",
    "yaw_sin",
    "yaw_cos",
    "roll_deg",
    "pitch_deg",
]
ACTION_NAMES = ["throttle", "steering"]
FEATURE_NAMES = STATE_NAMES + ACTION_NAMES


class CleanWorldModel(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 192, num_layers: int = 2, output_dim: int = 7):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


@dataclass
class Normalizer:
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    seq_length: int
    state_names: Sequence[str]
    action_names: Sequence[str]

    @classmethod
    def from_file(cls, path: str) -> "Normalizer":
        data = torch.load(path, map_location="cpu")
        return cls(
            x_mean=data["x_mean"].float(),
            x_std=data["x_std"].float().clamp_min(1e-6),
            y_mean=data["y_mean"].float(),
            y_std=data["y_std"].float().clamp_min(1e-6),
            seq_length=int(data["seq_length"]),
            state_names=list(data["state_names"]),
            action_names=list(data["action_names"]),
        )

    def norm_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)

    def denorm_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std.to(y.device) + self.y_mean.to(y.device)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def package_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def project_root() -> str:
    return os.path.dirname(package_dir())


def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def shortest_angle_delta_deg(target_deg: float, current_deg: float) -> float:
    return wrap_angle_deg(target_deg - current_deg)


def euler_deg_to_yaw_sin_cos(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    yaw_rad = deg2rad(yaw_deg)
    return np.array([math.sin(yaw_rad), math.cos(yaw_rad), roll_deg, pitch_deg], dtype=np.float32)


def quat_wxyz_to_euler_deg(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float]:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-8:
        return 0.0, 0.0, 0.0

    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return rad2deg(roll), rad2deg(pitch), rad2deg(yaw)


def state_from_row_dict(row: Dict[str, float]) -> np.ndarray:
    roll_deg, pitch_deg, yaw_deg = quat_wxyz_to_euler_deg(
        float(row["rot_0"]),
        float(row["rot_1"]),
        float(row["rot_2"]),
        float(row["rot_3"]),
    )
    yaw_sin, yaw_cos, roll_deg, pitch_deg = euler_deg_to_yaw_sin_cos(roll_deg, pitch_deg, yaw_deg)
    return np.array(
        [
            float(row["pos_x"]),
            float(row["pos_y"]),
            float(row["pos_z"]),
            float(yaw_sin),
            float(yaw_cos),
            float(roll_deg),
            float(pitch_deg),
        ],
        dtype=np.float32,
    )


def build_windows(
    trajectories: Iterable[Dict[str, torch.Tensor]],
    seq_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    meta_list: List[torch.Tensor] = []

    for episode_id, traj in enumerate(trajectories):
        states = traj["states"]
        actions = traj["actions"]
        if states.shape[0] <= seq_length:
            continue

        for start in range(states.shape[0] - seq_length):
            end = start + seq_length
            x_list.append(torch.cat([states[start:end], actions[start:end]], dim=1))
            y_list.append(states[end])
            meta_list.append(torch.tensor([episode_id, start], dtype=torch.float32))

    if not x_list:
        raise RuntimeError("No valid training windows were created.")
    return torch.stack(x_list), torch.stack(y_list), torch.stack(meta_list)


def split_by_episode(
    trajectories: Sequence[Dict[str, torch.Tensor]],
    train_ratio: float,
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    total = len(trajectories)
    split_idx = max(1, min(total - 1, int(total * train_ratio)))
    return list(trajectories[:split_idx]), list(trajectories[split_idx:])
