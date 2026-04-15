import argparse
import ctypes
import csv
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np
try:
    import pygame
except ImportError:
    pygame = None
import torch
import torch.nn as nn

try:
    import carla
except ImportError:
    carla = None


class QCarWorldModel(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 256, num_layers: int = 3, output_dim: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def raw_quaternion_to_xyzw(quat_raw: np.ndarray) -> np.ndarray:
    quat_raw = quat_raw.astype(np.float32, copy=False)
    return np.array([quat_raw[0], quat_raw[2], quat_raw[1], quat_raw[3]], dtype=np.float32)


def xyzw_to_raw_quaternion(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = quat_xyzw.astype(np.float32, copy=False)
    return np.array([quat_xyzw[0], quat_xyzw[2], quat_xyzw[1], quat_xyzw[3]], dtype=np.float32)


def canonicalize_position_history(history_tx9: np.ndarray) -> np.ndarray:
    out = history_tx9.astype(np.float32, copy=True)
    anchor = out[-1, :3].copy()
    out[:, :3] = out[:, :3] - anchor
    anchor_quat = raw_quaternion_to_xyzw(out[-1, 3:7])
    anchor_inv = quat_xyzw_inverse(anchor_quat)
    for step_idx in range(out.shape[0]):
        step_quat = raw_quaternion_to_xyzw(out[step_idx, 3:7])
        rel_quat = normalize_quaternion_xyzw(quat_xyzw_multiply(step_quat, anchor_inv))
        out[step_idx, 3:7] = xyzw_to_raw_quaternion(rel_quat)
    return out


@dataclass
class Normalizer:
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    seq_length: int

    @classmethod
    def from_file(cls, path: str) -> "Normalizer":
        data = torch.load(path, map_location="cpu")
        required = {"x_mean", "x_std", "y_mean", "y_std", "seq_length"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Normalization file is missing fields: {sorted(missing)}")

        return cls(
            x_mean=data["x_mean"].float(),
            x_std=data["x_std"].float().clamp_min(1e-6),
            y_mean=data["y_mean"].float(),
            y_std=data["y_std"].float().clamp_min(1e-6),
            seq_length=int(data["seq_length"]),
        )

    def norm_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)

    def denorm_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std.to(y.device) + self.y_mean.to(y.device)


@dataclass
class ModelBundle:
    name: str
    model: QCarWorldModel
    normalizer: Normalizer


@dataclass
class ControllerGoal:
    name: str
    throttle_sign: int
    steer_bias: float
    target_speed: float


@dataclass
class ActionEvaluation:
    action: np.ndarray
    bundle_name: str
    predicted_state: np.ndarray
    predicted_delta: np.ndarray
    score: float
    progress: float
    lateral: float
    yaw_delta: float


@dataclass
class PredictionInfo:
    bundle_name: str
    action: np.ndarray
    delta_xy_body: np.ndarray
    delta_yaw: float
    predicted_delta: np.ndarray
    yaw_delta_raw: float
    yaw_jump_suppressed: int


@dataclass
class CameraState:
    x: float
    y: float
    z: float
    yaw: float


def ensure_carla_available() -> None:
    if carla is None:
        raise RuntimeError("Failed to import carla. Install the CARLA Python API in the current environment first.")


def ensure_pygame_available() -> None:
    if pygame is None:
        raise RuntimeError("Failed to import pygame. Install pygame in the current environment first.")


def find_project_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for root in (script_dir, os.path.dirname(script_dir)):
        if os.path.exists(os.path.join(root, "PDHModel")):
            return root
    raise FileNotFoundError("Could not find the project root containing PDHModel")


def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def speed_mps(v: "carla.Vector3D") -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def euler_deg_to_quat_xyzw(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = deg2rad(roll_deg)
    pitch = deg2rad(pitch_deg)
    yaw = deg2rad(yaw_deg)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w], dtype=np.float32)


def quat_xyzw_to_euler_deg(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
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
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return rad2deg(roll), rad2deg(pitch), rad2deg(yaw)


def quat_xyzw_to_rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = normalize_quaternion_xyzw(quat_xyzw)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def quat_xyzw_inverse(quat_xyzw: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = normalize_quaternion_xyzw(quat_xyzw)
    return np.array([-qx, -qy, -qz, qw], dtype=np.float32)


def quat_xyzw_multiply(lhs_xyzw: np.ndarray, rhs_xyzw: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = lhs_xyzw
    rx, ry, rz, rw = rhs_xyzw
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float32,
    )


def rotation_matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = matrix.astype(np.float64, copy=False)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return normalize_quaternion_xyzw(np.array([qx, qy, qz, qw], dtype=np.float32))


MODEL_TO_CARLA_BASIS = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
CARLA_TO_MODEL_BASIS = MODEL_TO_CARLA_BASIS.T


def model_position_to_carla_xyz(position_xyz: np.ndarray) -> np.ndarray:
    return MODEL_TO_CARLA_BASIS @ position_xyz.astype(np.float32, copy=False)


def carla_xyz_to_model_position(position_xyz: np.ndarray) -> np.ndarray:
    return CARLA_TO_MODEL_BASIS @ position_xyz.astype(np.float32, copy=False)


def model_quat_raw_to_carla_quat_xyzw(quat_raw: np.ndarray) -> np.ndarray:
    rot_model = quat_xyzw_to_rotation_matrix(raw_quaternion_to_xyzw(quat_raw))
    rot_carla = MODEL_TO_CARLA_BASIS @ rot_model @ CARLA_TO_MODEL_BASIS
    return rotation_matrix_to_quat_xyzw(rot_carla)


def carla_quat_xyzw_to_model_quat_raw(quat_xyzw: np.ndarray) -> np.ndarray:
    rot_carla = quat_xyzw_to_rotation_matrix(quat_xyzw)
    rot_model = CARLA_TO_MODEL_BASIS @ rot_carla @ MODEL_TO_CARLA_BASIS
    return xyzw_to_raw_quaternion(rotation_matrix_to_quat_xyzw(rot_model))


def model_state_to_carla_transform(state: np.ndarray) -> "carla.Transform":
    carla_position = model_position_to_carla_xyz(state[:3])
    carla_quat = model_quat_raw_to_carla_quat_xyzw(state[3:7])
    roll_deg, pitch_deg, yaw_deg = quat_xyzw_to_euler_deg(*carla_quat.tolist())
    return carla.Transform(
        carla.Location(x=float(carla_position[0]), y=float(carla_position[1]), z=float(carla_position[2])),
        carla.Rotation(roll=roll_deg, pitch=pitch_deg, yaw=yaw_deg),
    )


def carla_transform_to_model_state(transform: "carla.Transform") -> np.ndarray:
    carla_quat = euler_deg_to_quat_xyzw(
        roll_deg=transform.rotation.roll,
        pitch_deg=transform.rotation.pitch,
        yaw_deg=transform.rotation.yaw,
    )
    model_quat = carla_quat_xyzw_to_model_quat_raw(carla_quat)
    model_position = carla_xyz_to_model_position(
        np.array([transform.location.x, transform.location.y, transform.location.z], dtype=np.float32)
    )
    return np.concatenate([model_position, model_quat], axis=0).astype(np.float32)


def model_quat_to_carla_yaw_deg(quat_raw: np.ndarray) -> float:
    carla_quat = model_quat_raw_to_carla_quat_xyzw(quat_raw)
    _, _, yaw_deg = quat_xyzw_to_euler_deg(*carla_quat.tolist())
    return yaw_deg


def apply_carla_yaw_delta_to_model_quat(quat_raw: np.ndarray, yaw_delta_deg: float) -> np.ndarray:
    carla_quat = model_quat_raw_to_carla_quat_xyzw(quat_raw)
    roll_deg, pitch_deg, yaw_deg = quat_xyzw_to_euler_deg(*carla_quat.tolist())
    updated_carla_quat = euler_deg_to_quat_xyzw(roll_deg, pitch_deg, yaw_deg + yaw_delta_deg)
    return carla_quat_xyzw_to_model_quat_raw(updated_carla_quat)


def follow_vehicle_with_spectator(
    world: "carla.World",
    vehicle: "carla.Vehicle",
    camera_state: Optional[CameraState] = None,
    smoothing: float = 0.18,
) -> CameraState:
    spectator = world.get_spectator()
    tf = vehicle.get_transform()
    yaw_rad = deg2rad(tf.rotation.yaw)

    back_dist = 8.0
    up_dist = 3.0

    cam_x = tf.location.x - back_dist * math.cos(yaw_rad)
    cam_y = tf.location.y - back_dist * math.sin(yaw_rad)
    cam_z = tf.location.z + up_dist

    if camera_state is None:
        camera_state = CameraState(
            x=cam_x,
            y=cam_y,
            z=cam_z,
            yaw=tf.rotation.yaw,
        )
    else:
        alpha = float(np.clip(smoothing, 0.0, 1.0))
        yaw_error = wrap_angle_deg(tf.rotation.yaw - camera_state.yaw)
        camera_state = CameraState(
            x=(1.0 - alpha) * camera_state.x + alpha * cam_x,
            y=(1.0 - alpha) * camera_state.y + alpha * cam_y,
            z=(1.0 - alpha) * camera_state.z + alpha * cam_z,
            yaw=wrap_angle_deg(camera_state.yaw + alpha * yaw_error),
        )

    spectator.set_transform(
        carla.Transform(
            carla.Location(x=camera_state.x, y=camera_state.y, z=camera_state.z),
            carla.Rotation(pitch=-15.0, yaw=camera_state.yaw, roll=0.0),
        )
    )
    return camera_state


def extract_state_vector_from_vehicle(vehicle: "carla.Vehicle") -> np.ndarray:
    return carla_transform_to_model_state(vehicle.get_transform())


def normalize_quaternion_in_state(state: np.ndarray) -> np.ndarray:
    out = state.copy()
    out[3:7] = xyzw_to_raw_quaternion(normalize_quaternion_xyzw(raw_quaternion_to_xyzw(out[3:7])))
    return out


def normalize_quaternion_xyzw(quat: np.ndarray) -> np.ndarray:
    out = quat.astype(np.float32, copy=True)
    q_norm = float(np.linalg.norm(out))
    if q_norm > 1e-8:
        out /= q_norm
    return out


def align_quaternion_xyzw(reference_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    ref = normalize_quaternion_xyzw(reference_quat)
    tgt = normalize_quaternion_xyzw(target_quat)
    if float(np.dot(ref, tgt)) < 0.0:
        tgt = -tgt
    return tgt


def align_quaternion_raw(reference_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    aligned = align_quaternion_xyzw(raw_quaternion_to_xyzw(reference_quat), raw_quaternion_to_xyzw(target_quat))
    return xyzw_to_raw_quaternion(aligned)


def blend_quaternion_xyzw(current_quat: np.ndarray, target_quat: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    current_xyzw = raw_quaternion_to_xyzw(current_quat)
    aligned_target = raw_quaternion_to_xyzw(align_quaternion_raw(current_quat, target_quat))
    blended = (1.0 - alpha) * normalize_quaternion_xyzw(current_xyzw) + alpha * aligned_target
    return xyzw_to_raw_quaternion(normalize_quaternion_xyzw(blended))


def choose_spawn_transform(world: "carla.World", spawn_index: int) -> "carla.Transform":
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available in the current CARLA map")
    spawn_index = max(0, min(spawn_index, len(spawn_points) - 1))
    return spawn_points[spawn_index]


def spawn_vehicle(world: "carla.World", vehicle_filter: str, spawn_transform: "carla.Transform") -> "carla.Vehicle":
    blueprints = world.get_blueprint_library().filter(vehicle_filter)
    if not blueprints:
        raise RuntimeError(f"Vehicle blueprint not found: {vehicle_filter}")

    bp = blueprints[0]
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "hero")
    if bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if colors:
            bp.set_attribute("color", colors[0])

    vehicle = world.try_spawn_actor(bp, spawn_transform)
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle. Try a different spawn index.")
    return vehicle


def predict_delta_state(
    history_tx9: np.ndarray,
    bundle: ModelBundle,
    device: torch.device,
) -> np.ndarray:
    history_tx9 = canonicalize_position_history(history_tx9)
    x = torch.from_numpy(history_tx9).float().unsqueeze(0).to(device)
    x = bundle.normalizer.norm_x(x)
    with torch.no_grad():
        pred_norm = bundle.model(x)
        delta_state = bundle.normalizer.denorm_y(pred_norm)
    return delta_state.squeeze(0).cpu().numpy().astype(np.float32)


def body_delta_to_world_state_delta(current_state: np.ndarray, predicted_delta: np.ndarray) -> np.ndarray:
    out = np.zeros(3, dtype=np.float32)
    yaw_deg = model_quat_to_carla_yaw_deg(current_state[3:7])
    yaw_rad = deg2rad(yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    dx_body = float(predicted_delta[0])
    dy_body = float(predicted_delta[1])
    dz_body = float(predicted_delta[2])

    carla_dx = dx_body * cos_y - dy_body * sin_y
    carla_dy = dx_body * sin_y + dy_body * cos_y
    model_delta = carla_xyz_to_model_position(np.array([carla_dx, carla_dy, dz_body], dtype=np.float32))

    out[0] = float(model_delta[0])
    out[1] = float(model_delta[1])
    out[2] = float(model_delta[2])
    return out


def predicted_output_to_next_state(current_state: np.ndarray, predicted_output: np.ndarray) -> np.ndarray:
    next_state = current_state.copy().astype(np.float32)
    world_delta = body_delta_to_world_state_delta(current_state, predicted_output[:3])
    next_state[:3] = current_state[:3] + world_delta[:3]
    yaw_delta_deg = float(predicted_output[3])
    next_state[3:7] = apply_carla_yaw_delta_to_model_quat(current_state[3:7], yaw_delta_deg)
    return normalize_quaternion_in_state(next_state)


def format_vector(name: str, vec: np.ndarray) -> str:
    values = ", ".join(f"{float(v):+.5f}" for v in vec.tolist())
    return f"{name}=[{values}]"


def bootstrap_model_history(
    history: Deque[np.ndarray],
    init_model_state: np.ndarray,
    initial_action: np.ndarray,
    seq_length: int,
) -> None:
    history.clear()
    feat = np.concatenate([init_model_state, initial_action], axis=0).astype(np.float32)
    for _ in range(seq_length):
        history.append(feat.copy())


def build_candidate_actions(goal: ControllerGoal) -> List[np.ndarray]:
    if goal.throttle_sign >= 0:
        throttle_levels = [0.04, 0.06, 0.08]
    else:
        throttle_levels = [-0.04, -0.06, -0.08]

    steer_center = goal.steer_bias
    steer_values = [steer_center - 0.18, steer_center, steer_center + 0.18]

    candidates: List[np.ndarray] = [np.array([0.0, 0.0], dtype=np.float32)]
    for throttle in throttle_levels:
        for steer in steer_values:
            candidates.append(
                np.array(
                    [float(throttle), float(np.clip(steer, -0.45, 0.45))],
                    dtype=np.float32,
                )
            )
    return candidates


def choose_bundle_for_action(
    action: np.ndarray,
    forward_bundle: ModelBundle,
    backward_bundle: ModelBundle,
) -> ModelBundle:
    return forward_bundle if float(action[0]) >= 0.0 else backward_bundle


def evaluate_action(
    history: Sequence[np.ndarray],
    model_state: np.ndarray,
    action: np.ndarray,
    goal: ControllerGoal,
    speed_now: float,
    forward_bundle: ModelBundle,
    backward_bundle: ModelBundle,
    device: torch.device,
) -> ActionEvaluation:
    bundle = choose_bundle_for_action(action, forward_bundle, backward_bundle)
    history_copy = deque(history, maxlen=bundle.normalizer.seq_length)
    feature = np.concatenate([model_state, action], axis=0).astype(np.float32)
    if len(history_copy) == 0:
        for _ in range(bundle.normalizer.seq_length):
            history_copy.append(feature.copy())
    else:
        history_copy.append(feature.copy())
        while len(history_copy) < bundle.normalizer.seq_length:
            history_copy.appendleft(history_copy[0].copy())

    hist_np = np.stack(list(history_copy), axis=0).astype(np.float32)
    predicted_output = predict_delta_state(hist_np, bundle, device)
    predicted_state = predicted_output_to_next_state(model_state, predicted_output)

    progress = float(predicted_output[0])
    lateral = float(predicted_output[1])

    yaw0 = model_quat_to_carla_yaw_deg(model_state[3:7])
    yaw1 = model_quat_to_carla_yaw_deg(predicted_state[3:7])
    yaw_delta = wrap_angle_deg(yaw1 - yaw0)

    desired_progress_sign = 1.0 if goal.throttle_sign >= 0 else -1.0
    speed_gain = progress / 0.05 if goal.target_speed >= 0.1 else 0.0
    predicted_speed = max(0.0, speed_now + desired_progress_sign * speed_gain)
    speed_error = abs(predicted_speed - goal.target_speed)

    score = 0.0
    score += desired_progress_sign * progress * 12.0
    score -= abs(lateral) * 2.5
    score -= abs(float(action[1]) - goal.steer_bias) * 1.5
    score -= speed_error * 0.8
    score -= abs(yaw_delta - goal.steer_bias * 12.0) * 0.15
    if abs(float(action[0])) < 1e-6 and goal.target_speed > 0.1:
        score -= 2.0

    return ActionEvaluation(
        action=action,
        bundle_name=bundle.name,
        predicted_state=predicted_state,
        predicted_delta=predicted_output,
        score=score,
        progress=progress,
        lateral=lateral,
        yaw_delta=yaw_delta,
    )


# 给carla车辆应用预测的数据
def apply_predicted_transition_to_vehicle(
    vehicle: "carla.Vehicle",
    curr_model_state: np.ndarray,
    next_model_state: np.ndarray,
    fixed_delta: float,
) -> None:
    tf = vehicle.get_transform()

    scale_gain = 10.0
    curr_carla_pos = model_position_to_carla_xyz(curr_model_state[:3])
    next_carla_pos = model_position_to_carla_xyz(next_model_state[:3])
    delta_xy_world = (next_carla_pos[:2] - curr_carla_pos[:2]) * scale_gain

    yaw0 = model_quat_to_carla_yaw_deg(curr_model_state[3:7])
    yaw1 = model_quat_to_carla_yaw_deg(next_model_state[3:7])
    delta_yaw = wrap_angle_deg(yaw1 - yaw0) * scale_gain

    max_step_dist = 0.25
    max_step_yaw = 8.0

    step_dist = float(math.hypot(delta_xy_world[0], delta_xy_world[1]))
    if step_dist > max_step_dist and step_dist > 1e-6:
        delta_xy_world *= max_step_dist / step_dist

    delta_yaw = float(np.clip(delta_yaw, -max_step_yaw, max_step_yaw))
    dx_world = float(delta_xy_world[0])
    dy_world = float(delta_xy_world[1])

    new_tf = carla.Transform(
        carla.Location(
            x=tf.location.x + dx_world,
            y=tf.location.y + dy_world,
            z=tf.location.z,
        ),
        carla.Rotation(
            roll=tf.rotation.roll,
            pitch=tf.rotation.pitch,
            yaw=tf.rotation.yaw + delta_yaw,
        ),
    )
    vehicle.set_transform(new_tf)

    dt = max(fixed_delta, 1e-6)
    vehicle.set_target_velocity(carla.Vector3D(x=dx_world / dt, y=dy_world / dt, z=0.0))


def save_run_log(rows: List[dict], output_csv: str) -> None:
    if not rows:
        return
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def control_to_action(control: "carla.VehicleControl") -> np.ndarray:
    throttle = float(control.throttle)
    if bool(control.reverse):
        throttle = -throttle
    throttle *= max(0.0, 1.0 - float(control.brake))
    steer = float(control.steer)
    return np.array([throttle, steer], dtype=np.float32)


class QCarVehicle:
    def __init__(
        self,
        actor: "carla.Vehicle",
        forward_bundle: ModelBundle,
        backward_bundle: ModelBundle,
        fixed_delta: float,
        interpolation_alpha: float = 0.35,
    ) -> None:
        self.actor = actor
        self.forward_bundle = forward_bundle
        self.backward_bundle = backward_bundle
        self.fixed_delta = fixed_delta
        self.interpolation_alpha = float(np.clip(interpolation_alpha, 0.0, 1.0))
        self.seq_length = forward_bundle.normalizer.seq_length
        self.history: Deque[np.ndarray] = deque(maxlen=self.seq_length)
        self.model_state = extract_state_vector_from_vehicle(actor)
        self.last_prediction: Optional[PredictionInfo] = None
        self.last_yaw_delta_cmd = 0.0
        bootstrap_model_history(
            self.history,
            self.model_state,
            np.array([0.0, 0.0], dtype=np.float32),
            self.seq_length,
        )

    def get_transform(self) -> "carla.Transform":
        return self.actor.get_transform()

    def get_velocity(self) -> "carla.Vector3D":
        return self.actor.get_velocity()

    def reset(self, transform: "carla.Transform") -> None:
        self.actor.set_transform(transform)
        self.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self.model_state = extract_state_vector_from_vehicle(self.actor)
        self.last_prediction = None
        self.last_yaw_delta_cmd = 0.0
        bootstrap_model_history(
            self.history,
            self.model_state,
            np.array([0.0, 0.0], dtype=np.float32),
            self.seq_length,
        )

    def _stabilize_predicted_output(self, bundle_name: str, action: np.ndarray, predicted_output: np.ndarray) -> np.ndarray:
        stabilized = predicted_output.astype(np.float32, copy=True)
        yaw_delta_raw = float(stabilized[3])

        if bundle_name == "backward" and abs(float(action[1])) >= 0.15:
            prev = float(self.last_yaw_delta_cmd)
            curr = yaw_delta_raw
            large_prev = abs(prev) >= 0.6
            large_curr = abs(curr) >= 0.6
            sign_flip = (prev * curr) < 0.0

            if large_prev and large_curr and sign_flip:
                replacement = prev
                stabilized[3] = float(np.clip(replacement, -1.5, 1.5))
            else:
                stabilized[3] = float(np.clip(curr, -1.5, 1.5))
        else:
            stabilized[3] = float(np.clip(yaw_delta_raw, -2.0, 2.0))

        self.last_yaw_delta_cmd = float(stabilized[3])
        return stabilized

    def apply_control(self, control: "carla.VehicleControl", device: torch.device) -> None:
        action = control_to_action(control)
        bundle = choose_bundle_for_action(action, self.forward_bundle, self.backward_bundle)

        current_state = self.model_state.copy()
        self.model_state = current_state

        history_copy = deque(self.history, maxlen=bundle.normalizer.seq_length)
        feature = np.concatenate([current_state, action], axis=0).astype(np.float32)
        history_copy.append(feature.copy())
        while len(history_copy) < bundle.normalizer.seq_length:
            history_copy.appendleft(feature.copy())

        hist_np = np.stack(list(history_copy), axis=0).astype(np.float32)
        predicted_output_raw = predict_delta_state(hist_np, bundle, device)
        predicted_output = self._stabilize_predicted_output(bundle.name, action, predicted_output_raw)
        next_state = predicted_output_to_next_state(current_state, predicted_output)

        self._apply_model_state_transition(current_state, next_state, bundle.name, action, predicted_output, predicted_output_raw)
        self.model_state = next_state.copy()
        self.history.append(np.concatenate([self.model_state, action], axis=0).astype(np.float32))

    def _apply_model_state_transition(
        self,
        curr_model_state: np.ndarray,
        next_model_state: np.ndarray,
        bundle_name: str,
        action: np.ndarray,
        predicted_delta: np.ndarray,
        predicted_delta_raw: np.ndarray,
    ) -> None:
        tf = self.actor.get_transform()
        target_tf = model_state_to_carla_transform(next_model_state)
        alpha = self.interpolation_alpha

        dx_world = float((target_tf.location.x - tf.location.x) * alpha)
        dy_world = float((target_tf.location.y - tf.location.y) * alpha)
        dz_world = float((target_tf.location.z - tf.location.z) * alpha)
        delta_yaw = wrap_angle_deg(target_tf.rotation.yaw - tf.rotation.yaw) * alpha

        new_tf = carla.Transform(
            carla.Location(
                x=tf.location.x + dx_world,
                y=tf.location.y + dy_world,
                z=tf.location.z + dz_world,
            ),
            carla.Rotation(
                roll=tf.rotation.roll + (target_tf.rotation.roll - tf.rotation.roll) * alpha,
                pitch=tf.rotation.pitch + (target_tf.rotation.pitch - tf.rotation.pitch) * alpha,
                yaw=tf.rotation.yaw + delta_yaw,
            ),
        )
        self.actor.set_transform(new_tf)

        dt = max(self.fixed_delta, 1e-6)
        self.actor.set_target_velocity(carla.Vector3D(x=dx_world / dt, y=dy_world / dt, z=dz_world / dt))
        self.last_prediction = PredictionInfo(
            bundle_name=bundle_name,
            action=action.copy(),
            delta_xy_body=np.array([dx_world, dy_world], dtype=np.float32),
            delta_yaw=delta_yaw,
            predicted_delta=predicted_delta.copy(),
            yaw_delta_raw=float(predicted_delta_raw[3]),
            yaw_jump_suppressed=int(abs(float(predicted_delta[3]) - float(predicted_delta_raw[3])) > 1e-6),
        )


VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44
VK_SPACE = 0x20
VK_R = 0x52
VK_ESC = 0x1B


def is_key_down(vk_code: int) -> bool:
    return (ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000) != 0


class EdgeKeyReader:
    def __init__(self) -> None:
        self.prev = {
            VK_W: False,
            VK_A: False,
            VK_S: False,
            VK_D: False,
            VK_SPACE: False,
            VK_R: False,
            VK_ESC: False,
        }

    def just_pressed(self, vk_code: int) -> bool:
        now = is_key_down(vk_code)
        was = self.prev[vk_code]
        self.prev[vk_code] = now
        return now and (not was)


def load_bundle(
    name: str,
    model_path: str,
    norm_path: str,
    device: torch.device,
) -> ModelBundle:
    normalizer = Normalizer.from_file(norm_path)
    model = QCarWorldModel()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return ModelBundle(name=name, model=model, normalizer=normalizer)


def build_vehicle_control(throttle_cmd: float, steer_cmd: float) -> "carla.VehicleControl":
    if throttle_cmd > 0.0:
        return carla.VehicleControl(throttle=throttle_cmd, steer=steer_cmd, brake=0.0, reverse=False)
    if throttle_cmd < 0.0:
        return carla.VehicleControl(throttle=abs(throttle_cmd), steer=steer_cmd, brake=0.0, reverse=True)
    return carla.VehicleControl(throttle=0.0, steer=steer_cmd, brake=1.0, reverse=False)


def build_mode_name(throttle_cmd: float, steer_cmd: float) -> str:
    throttle_name = "IDLE"
    if throttle_cmd > 0.0:
        throttle_name = "FORWARD"
    elif throttle_cmd < 0.0:
        throttle_name = "REVERSE"

    steer_name = "STRAIGHT"
    if steer_cmd < -1e-6:
        steer_name = "LEFT"
    elif steer_cmd > 1e-6:
        steer_name = "RIGHT"

    if throttle_name == "IDLE":
        return throttle_name if steer_name == "STRAIGHT" else f"{throttle_name}_{steer_name}"
    return throttle_name if steer_name == "STRAIGHT" else f"{throttle_name}_{steer_name}"


def main() -> None:
    ensure_carla_available()
    ensure_pygame_available()
    project_root = find_project_root()

    parser = argparse.ArgumentParser(
        description="CARLA controller driven by forward/backward QCar world models from PDHModel"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--fixed-delta", type=float, default=0.05)
    parser.add_argument(
        "--forward-model-path",
        default=os.path.join(project_root, "PDHModel", "forward_world_model.pth"),
    )
    parser.add_argument(
        "--forward-norm-path",
        default=os.path.join(project_root, "PDHModel", "forward_normalization.pt"),
    )
    parser.add_argument(
        "--backward-model-path",
        default=os.path.join(project_root, "PDHModel", "backward_world_model.pth"),
    )
    parser.add_argument(
        "--backward-norm-path",
        default=os.path.join(project_root, "PDHModel", "backward_normalization.pt"),
    )
    parser.add_argument("--interpolation-alpha", type=float, default=0.35)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forward_bundle = load_bundle("forward", args.forward_model_path, args.forward_norm_path, device)
    backward_bundle = load_bundle("backward", args.backward_model_path, args.backward_norm_path, device)

    if forward_bundle.normalizer.seq_length != backward_bundle.normalizer.seq_length:
        raise ValueError("Forward and backward models use different seq_length values")

    pygame.init()
    pygame.display.set_caption("CARLA Controller PDH")
    pygame.display.set_mode((980, 260))
    font = pygame.font.SysFont(None, 24)
    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()

    original_settings = world.get_settings()
    vehicle: Optional["carla.Vehicle"] = None
    qcar_vehicle: Optional[QCarVehicle] = None
    log_rows: List[dict] = []
    current_mode = "IDLE"
    throttle_cmd = 0.0
    steer_cmd = 0.0
    current_control = build_vehicle_control(throttle_cmd, steer_cmd)
    camera_state: Optional[CameraState] = None
    control_active = False

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta
        world.apply_settings(settings)

        spawn_transform = choose_spawn_transform(world, args.spawn_index)
        vehicle = spawn_vehicle(world, args.vehicle_filter, spawn_transform)
        vehicle.set_autopilot(False)
        qcar_vehicle = QCarVehicle(
            actor=vehicle,
            forward_bundle=forward_bundle,
            backward_bundle=backward_bundle,
            fixed_delta=args.fixed_delta,
            interpolation_alpha=args.interpolation_alpha,
        )

        world.tick()
        world.tick()

        reset_transform = qcar_vehicle.get_transform()
        camera_state = follow_vehicle_with_spectator(world, qcar_vehicle.actor, camera_state)

        frame_count = 0
        key_reader = EdgeKeyReader()

        print("=" * 80)
        print("carla_controller_PDH started")
        print(f"Forward model: {args.forward_model_path}")
        print(f"Backward model: {args.backward_model_path}")
        print(f"Interpolation alpha: {args.interpolation_alpha:.2f}")
        print("Quaternion control: model-owned pose updates with interpolation only")
        print("QCarVehicle.apply_control() now applies the model target pose directly.")
        print("W: forward | S: reverse | Space: idle | A: toggle left steer | D: toggle right steer")
        print("R: reset | ESC: quit")
        print("=" * 80)

        while True:
            pygame.event.pump()

            if key_reader.just_pressed(VK_ESC):
                break

            if key_reader.just_pressed(VK_W):
                throttle_cmd = 0.08
                current_control = build_vehicle_control(throttle_cmd, steer_cmd)
                current_mode = build_mode_name(throttle_cmd, steer_cmd)
                control_active = True
                print(f"[control] {current_mode}")
            elif key_reader.just_pressed(VK_A):
                steer_cmd = 0.0 if steer_cmd < 0.0 else -0.20
                current_control = build_vehicle_control(throttle_cmd, steer_cmd)
                current_mode = build_mode_name(throttle_cmd, steer_cmd)
                control_active = abs(throttle_cmd) > 0.0
                print(f"[control] {current_mode}")
            elif key_reader.just_pressed(VK_D):
                steer_cmd = 0.0 if steer_cmd > 0.0 else 0.20
                current_control = build_vehicle_control(throttle_cmd, steer_cmd)
                current_mode = build_mode_name(throttle_cmd, steer_cmd)
                control_active = abs(throttle_cmd) > 0.0
                print(f"[control] {current_mode}")
            elif key_reader.just_pressed(VK_S):
                throttle_cmd = -0.08
                current_control = build_vehicle_control(throttle_cmd, steer_cmd)
                current_mode = build_mode_name(throttle_cmd, steer_cmd)
                control_active = True
                print(f"[control] {current_mode}")
            elif key_reader.just_pressed(VK_SPACE):
                throttle_cmd = 0.0
                current_control = build_vehicle_control(throttle_cmd, steer_cmd)
                current_mode = build_mode_name(throttle_cmd, steer_cmd)
                control_active = False
                qcar_vehicle.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                print(f"[control] {current_mode}")
            elif key_reader.just_pressed(VK_R):
                throttle_cmd = 0.0
                steer_cmd = 0.0
                current_control = build_vehicle_control(throttle_cmd, steer_cmd)
                current_mode = build_mode_name(throttle_cmd, steer_cmd)
                control_active = False
                qcar_vehicle.reset(reset_transform)
                world.tick()
                camera_state = follow_vehicle_with_spectator(world, qcar_vehicle.actor, camera_state=None)
                print("[control] RESET")

            if control_active:
                qcar_vehicle.apply_control(current_control, device)
            else:
                qcar_vehicle.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                qcar_vehicle.model_state = extract_state_vector_from_vehicle(qcar_vehicle.actor)

            world.tick()
            frame_count += 1
            camera_state = follow_vehicle_with_spectator(world, qcar_vehicle.actor, camera_state)

            tf = qcar_vehicle.get_transform()
            vel = qcar_vehicle.get_velocity()
            speed_now = speed_mps(vel)
            carla_state = extract_state_vector_from_vehicle(qcar_vehicle.actor)
            model_state = qcar_vehicle.model_state
            pred = qcar_vehicle.last_prediction
            action = control_to_action(current_control)

            log_rows.append(
                {
                    "frame": frame_count,
                    "time": frame_count * args.fixed_delta,
                    "mode": current_mode,
                    "selected_model": "" if pred is None else pred.bundle_name,
                    "action_throttle": float(action[0]),
                    "action_steer": float(action[1]),
                    "carla_pos_x": tf.location.x,
                    "carla_pos_y": tf.location.y,
                    "carla_pos_z": tf.location.z,
                    "carla_yaw_deg": tf.rotation.yaw,
                    "carla_rot_0": float(carla_state[3]),
                    "carla_rot_1": float(carla_state[4]),
                    "carla_rot_2": float(carla_state[5]),
                    "carla_rot_3": float(carla_state[6]),
                    "speed_mps": speed_now,
                    "model_pos_x": float(model_state[0]),
                    "model_pos_y": float(model_state[1]),
                    "model_pos_z": float(model_state[2]),
                    "model_rot_0": float(model_state[3]),
                    "model_rot_1": float(model_state[4]),
                    "model_rot_2": float(model_state[5]),
                    "model_rot_3": float(model_state[6]),
                    "pred_delta_0": 0.0 if pred is None else float(pred.predicted_delta[0]),
                    "pred_delta_1": 0.0 if pred is None else float(pred.predicted_delta[1]),
                    "pred_delta_2": 0.0 if pred is None else float(pred.predicted_delta[2]),
                    "pred_delta_3": 0.0 if pred is None else float(pred.predicted_delta[3]),
                    "pred_delta_3_raw": 0.0 if pred is None else float(pred.yaw_delta_raw),
                    "yaw_jump_suppressed": 0 if pred is None else int(pred.yaw_jump_suppressed),
                }
            )

            lines = [
                f"Mode: {current_mode}",
                f"apply_control input: throttle={action[0]:.3f}, steer={action[1]:.3f}",
                "World model: " + ("-" if pred is None else pred.bundle_name),
                f"Vehicle pos: x={tf.location.x:.3f}, y={tf.location.y:.3f}, z={tf.location.z:.3f}",
                f"Vehicle yaw={tf.rotation.yaw:.3f} deg | speed={speed_now:.3f} m/s",
                format_vector("CARLA state7", carla_state),
                f"Model pos: x={model_state[0]:.3f}, y={model_state[1]:.3f}, z={model_state[2]:.3f}",
                format_vector("Model state7", model_state),
                "W forward | S reverse | A toggle left | D toggle right | Space idle | R reset | ESC quit",
            ]

            if pred is not None:
                lines.insert(
                    3,
                    f"Applied world dx={pred.delta_xy_body[0]:.4f}, dy={pred.delta_xy_body[1]:.4f}, yaw_delta={pred.delta_yaw:.3f}, raw_yaw_delta={pred.yaw_delta_raw:.3f}, suppressed={pred.yaw_jump_suppressed}",
                )
                lines.insert(7, format_vector("Pred delta7", pred.predicted_delta))

            screen.fill((20, 20, 20))
            y = 12
            for line in lines:
                img = font.render(line, True, (235, 235, 235))
                screen.blit(img, (12, y))
                y += 30
            pygame.display.flip()
            clock.tick(60)

            # 只在非 IDLE 状态且每10帧打印一次详细调试信息
            if current_mode != "IDLE" and frame_count % 10 == 0:
                print(
                    f"[frame={frame_count:05d}] mode={current_mode} model={'' if pred is None else pred.bundle_name} "
                    f"action=({action[0]:+.3f},{action[1]:+.3f}) speed={speed_now:.3f}\n"
                    f"  {format_vector('carla_state7', carla_state)}\n"
                    f"  {format_vector('model_state7', model_state)}\n"
                    f"  {format_vector('pred_delta7', np.zeros(7, dtype=np.float32) if pred is None else pred.predicted_delta)}",
                    flush=True,
                )

    finally:
        print("Cleaning up CARLA resources...")
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        if vehicle is not None:
            try:
                vehicle.destroy()
            except Exception:
                pass

        pygame.quit()

        try:
            output_csv = os.path.join(project_root, "carla_controller_PDH_run.csv")
            save_run_log(log_rows, output_csv)
            if log_rows:
                print(f"[log] Saved run log to {output_csv}")
        except Exception as e:
            print(f"[log] Failed to save run log: {e}")


if __name__ == "__main__":
    main()
