import argparse
import ctypes
import csv
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np
import pygame
import torch
import torch.nn as nn

try:
    import carla
except ImportError as e:
    raise SystemExit(
        "Failed to import carla. Install the CARLA Python API in the current environment first."
    ) from e


class QCarWorldModel(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 256, num_layers: int = 3, output_dim: int = 7):
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


@dataclass
class CameraState:
    x: float
    y: float
    z: float
    yaw: float


def find_project_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for root in (script_dir, os.path.dirname(script_dir)):
        if os.path.exists(os.path.join(root, "models_saved")):
            return root
    raise FileNotFoundError("Could not find the project root containing models_saved")


def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def speed_mps(v: "carla.Vector3D") -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def euler_deg_to_quat_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
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
    return np.array([w, x, y, z], dtype=np.float32)


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
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return rad2deg(roll), rad2deg(pitch), rad2deg(yaw)


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
    tf = vehicle.get_transform()
    quat = euler_deg_to_quat_wxyz(
        roll_deg=tf.rotation.roll,
        pitch_deg=tf.rotation.pitch,
        yaw_deg=tf.rotation.yaw,
    )
    return np.array(
        [tf.location.x, tf.location.y, tf.location.z, quat[0], quat[1], quat[2], quat[3]],
        dtype=np.float32,
    )


def normalize_quaternion_in_state(state: np.ndarray) -> np.ndarray:
    out = state.copy()
    q = out[3:7]
    q_norm = float(np.linalg.norm(q))
    if q_norm > 1e-8:
        out[3:7] = q / q_norm
    return out


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
    x = torch.from_numpy(history_tx9).float().unsqueeze(0).to(device)
    x = bundle.normalizer.norm_x(x)
    with torch.no_grad():
        pred_norm = bundle.model(x)
        delta_state = bundle.normalizer.denorm_y(pred_norm)
    return delta_state.squeeze(0).cpu().numpy().astype(np.float32)


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
    predicted_delta = predict_delta_state(hist_np, bundle, device)
    predicted_state = normalize_quaternion_in_state(model_state + predicted_delta)

    delta_xy = predicted_state[0:2] - model_state[0:2]
    progress = float(delta_xy[0])
    lateral = float(delta_xy[1])

    _, _, yaw0 = quat_wxyz_to_euler_deg(*model_state[3:7].tolist())
    _, _, yaw1 = quat_wxyz_to_euler_deg(*predicted_state[3:7].tolist())
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
        predicted_delta=predicted_delta,
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
    delta_xy_body = (next_model_state[0:2] - curr_model_state[0:2]) * scale_gain

    _, _, yaw0 = quat_wxyz_to_euler_deg(*curr_model_state[3:7].tolist())
    _, _, yaw1 = quat_wxyz_to_euler_deg(*next_model_state[3:7].tolist())
    delta_yaw = wrap_angle_deg(yaw1 - yaw0) * scale_gain

    max_step_dist = 0.25
    max_step_yaw = 8.0

    step_dist = float(math.hypot(delta_xy_body[0], delta_xy_body[1]))
    if step_dist > max_step_dist and step_dist > 1e-6:
        delta_xy_body *= max_step_dist / step_dist

    delta_yaw = float(np.clip(delta_yaw, -max_step_yaw, max_step_yaw))

    yaw_world_rad = deg2rad(tf.rotation.yaw)
    cos_y = math.cos(yaw_world_rad)
    sin_y = math.sin(yaw_world_rad)
    dx_body = float(delta_xy_body[0])
    dy_body = float(delta_xy_body[1])
    dx_world = dx_body * cos_y - dy_body * sin_y
    dy_world = dx_body * sin_y + dy_body * cos_y

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
        scale_gain: float = 10.0,
    ) -> None:
        self.actor = actor
        self.forward_bundle = forward_bundle
        self.backward_bundle = backward_bundle
        self.fixed_delta = fixed_delta
        self.scale_gain = scale_gain
        self.seq_length = forward_bundle.normalizer.seq_length
        self.history: Deque[np.ndarray] = deque(maxlen=self.seq_length)
        self.model_state = extract_state_vector_from_vehicle(actor)
        self.last_prediction: Optional[PredictionInfo] = None
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
        bootstrap_model_history(
            self.history,
            self.model_state,
            np.array([0.0, 0.0], dtype=np.float32),
            self.seq_length,
        )

    def apply_control(self, control: "carla.VehicleControl", device: torch.device) -> None:
        action = control_to_action(control)
        bundle = choose_bundle_for_action(action, self.forward_bundle, self.backward_bundle)

        current_state = extract_state_vector_from_vehicle(self.actor)
        self.model_state = current_state

        history_copy = deque(self.history, maxlen=bundle.normalizer.seq_length)
        feature = np.concatenate([current_state, action], axis=0).astype(np.float32)
        history_copy.append(feature.copy())
        while len(history_copy) < bundle.normalizer.seq_length:
            history_copy.appendleft(feature.copy())

        hist_np = np.stack(list(history_copy), axis=0).astype(np.float32)
        predicted_delta = predict_delta_state(hist_np, bundle, device)
        next_state = normalize_quaternion_in_state(current_state + predicted_delta)

        self._apply_model_state_transition(current_state, next_state, bundle.name, action)
        self.model_state = next_state
        self.history.append(np.concatenate([self.model_state, action], axis=0).astype(np.float32))

    def _apply_model_state_transition(
        self,
        curr_model_state: np.ndarray,
        next_model_state: np.ndarray,
        bundle_name: str,
        action: np.ndarray,
    ) -> None:
        tf = self.actor.get_transform()

        delta_xy_body = (next_model_state[0:2] - curr_model_state[0:2]) * self.scale_gain

        _, _, yaw0 = quat_wxyz_to_euler_deg(*curr_model_state[3:7].tolist())
        _, _, yaw1 = quat_wxyz_to_euler_deg(*next_model_state[3:7].tolist())
        delta_yaw = wrap_angle_deg(yaw1 - yaw0) * self.scale_gain

        max_step_dist = 0.25
        max_step_yaw = 8.0

        step_dist = float(math.hypot(delta_xy_body[0], delta_xy_body[1]))
        if step_dist > max_step_dist and step_dist > 1e-6:
            delta_xy_body *= max_step_dist / step_dist

        delta_yaw = float(np.clip(delta_yaw, -max_step_yaw, max_step_yaw))

        yaw_world_rad = deg2rad(tf.rotation.yaw)
        cos_y = math.cos(yaw_world_rad)
        sin_y = math.sin(yaw_world_rad)
        dx_body = float(delta_xy_body[0])
        dy_body = float(delta_xy_body[1])
        dx_world = dx_body * cos_y - dy_body * sin_y
        dy_world = dx_body * sin_y + dy_body * cos_y

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
        self.actor.set_transform(new_tf)

        dt = max(self.fixed_delta, 1e-6)
        self.actor.set_target_velocity(carla.Vector3D(x=dx_world / dt, y=dy_world / dt, z=0.0))
        self.last_prediction = PredictionInfo(
            bundle_name=bundle_name,
            action=action.copy(),
            delta_xy_body=np.array([dx_body, dy_body], dtype=np.float32),
            delta_yaw=delta_yaw,
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


def main() -> None:
    project_root = find_project_root()

    parser = argparse.ArgumentParser(
        description="CARLA controller driven by forward/backward QCar world models"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--fixed-delta", type=float, default=0.05)
    parser.add_argument(
        "--forward-model-path",
        default=os.path.join(project_root, "models_saved", "forward_world_model.pth"),
    )
    parser.add_argument(
        "--forward-norm-path",
        default=os.path.join(project_root, "models_saved", "forward_normalization.pt"),
    )
    parser.add_argument(
        "--backward-model-path",
        default=os.path.join(project_root, "models_saved", "backward_world_model.pth"),
    )
    parser.add_argument(
        "--backward-norm-path",
        default=os.path.join(project_root, "models_saved", "backward_normalization.pt"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forward_bundle = load_bundle("forward", args.forward_model_path, args.forward_norm_path, device)
    backward_bundle = load_bundle("backward", args.backward_model_path, args.backward_norm_path, device)

    if forward_bundle.normalizer.seq_length != backward_bundle.normalizer.seq_length:
        raise ValueError("Forward and backward models use different seq_length values")

    pygame.init()
    pygame.display.set_caption("CARLA Controller v1")
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
    current_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False)
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
            scale_gain=10.0,
        )

        world.tick()
        world.tick()

        reset_transform = qcar_vehicle.get_transform()
        camera_state = follow_vehicle_with_spectator(world, qcar_vehicle.actor, camera_state)

        frame_count = 0
        key_reader = EdgeKeyReader()

        print("=" * 80)
        print("carla_controller_v1 started")
        print("QCarVehicle.apply_control() now interprets VehicleControl with the world model.")
        print("W: forward | A: left-forward | D: right-forward | S: reverse | Space: idle")
        print("R: reset | ESC: quit")
        print("=" * 80)

        while True:
            pygame.event.pump()

            if key_reader.just_pressed(VK_ESC):
                break

            if key_reader.just_pressed(VK_W):
                current_mode = "FORWARD"
                current_control = carla.VehicleControl(throttle=0.08, steer=0.0, brake=0.0, reverse=False)
                control_active = True
                print("[control] FORWARD")
            elif key_reader.just_pressed(VK_A):
                current_mode = "LEFT_FORWARD"
                current_control = carla.VehicleControl(throttle=0.08, steer=-0.18, brake=0.0, reverse=False)
                control_active = True
                print("[control] LEFT_FORWARD")
            elif key_reader.just_pressed(VK_D):
                current_mode = "RIGHT_FORWARD"
                current_control = carla.VehicleControl(throttle=0.08, steer=0.18, brake=0.0, reverse=False)
                control_active = True
                print("[control] RIGHT_FORWARD")
            elif key_reader.just_pressed(VK_S):
                current_mode = "REVERSE"
                current_control = carla.VehicleControl(throttle=0.08, steer=0.0, brake=0.0, reverse=True)
                control_active = True
                print("[control] REVERSE")
            elif key_reader.just_pressed(VK_SPACE):
                current_mode = "IDLE"
                current_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False)
                control_active = False
                qcar_vehicle.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                print("[control] IDLE")
            elif key_reader.just_pressed(VK_R):
                current_mode = "IDLE"
                current_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False)
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
                    "speed_mps": speed_now,
                    "model_pos_x": float(model_state[0]),
                    "model_pos_y": float(model_state[1]),
                    "model_pos_z": float(model_state[2]),
                    "model_rot_0": float(model_state[3]),
                    "model_rot_1": float(model_state[4]),
                    "model_rot_2": float(model_state[5]),
                    "model_rot_3": float(model_state[6]),
                }
            )

            lines = [
                f"Mode: {current_mode}",
                f"apply_control input: throttle={action[0]:.3f}, steer={action[1]:.3f}",
                "World model: " + ("-" if pred is None else pred.bundle_name),
                f"Vehicle pos: x={tf.location.x:.3f}, y={tf.location.y:.3f}, z={tf.location.z:.3f}",
                f"Vehicle yaw={tf.rotation.yaw:.3f} deg | speed={speed_now:.3f} m/s",
                f"Model pos: x={model_state[0]:.3f}, y={model_state[1]:.3f}, z={model_state[2]:.3f}",
                "W cruise | A left bias | D right bias | S reverse | Space idle | R reset | ESC quit",
            ]

            if pred is not None:
                lines.insert(
                    3,
                    f"Predicted body dx={pred.delta_xy_body[0]:.4f}, dy={pred.delta_xy_body[1]:.4f}, yaw_delta={pred.delta_yaw:.3f}",
                )

            screen.fill((20, 20, 20))
            y = 12
            for line in lines:
                img = font.render(line, True, (235, 235, 235))
                screen.blit(img, (12, y))
                y += 30
            pygame.display.flip()
            clock.tick(60)

            if frame_count % 40 == 0:
                print(
                    f"[frame={frame_count}] mode={current_mode} model={'' if pred is None else pred.bundle_name} "
                    f"action=({action[0]:.3f},{action[1]:.3f}) speed={speed_now:.3f}"
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
            output_csv = os.path.join(project_root, "carla_controller_v1_run.csv")
            save_run_log(log_rows, output_csv)
            if log_rows:
                print(f"[log] Saved run log to {output_csv}")
        except Exception as e:
            print(f"[log] Failed to save run log: {e}")


if __name__ == "__main__":
    main()
