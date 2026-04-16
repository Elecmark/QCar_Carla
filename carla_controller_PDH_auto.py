import argparse
import ctypes
import csv
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None
import torch

try:
    import carla
except ImportError:
    carla = None

from carla_controller_PDH import (
    ModelBundle,
    bootstrap_model_history,
    build_vehicle_control,
    choose_bundle_for_action,
    extract_state_vector_from_vehicle,
    find_project_root,
    follow_vehicle_with_spectator,
    load_bundle,
    predict_delta_state,
    spawn_vehicle,
    choose_spawn_transform,
    wrap_angle_deg,
)
from policy_network import SACAgent, SACConfig
from reference_generator import list_reference_csvs, quaternion_yaw_deg, yaw_deg_to_raw_quaternion

KEY_MAP = (
    {
        pygame.K_1: 0,
        pygame.K_2: 1,
        pygame.K_3: 2,
        pygame.K_4: 3,
        pygame.K_5: 4,
        pygame.K_6: 5,
    }
    if pygame is not None
    else {}
)

STANDARD_REFERENCE_FILES = [
    "circle_radius5_dt0.004.csv",
    "figure8_size10_dt0.004.csv",
    "mixed_trajectory_001.csv",
    "s_curve_length20_dt0.004.csv",
    "sine_amplitude2_dt0.004.csv",
    "straight_length100_dt0.004.csv",
]

VK_SPACE = 0x20
VK_R = 0x52
VK_Q = 0x51
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_ESC = 0x1B
USE_RL_CORRECTION_DEFAULT = True


@dataclass
class ReferenceFrame:
    time: float
    action: np.ndarray
    state: np.ndarray
    linear_speed: float
    source_row: Dict[str, float]


def load_policy_agent(policy_path: Path, device: torch.device) -> SACAgent:
    payload = torch.load(policy_path, map_location=device)
    config = SACConfig(**payload["config"])
    agent = SACAgent(config, device)
    if "critic" in payload:
        agent.load_state_dict(payload)
    else:
        agent.load_actor_state_dict(payload)
    return agent


def target_yaw_step_deg(current_ref: ReferenceFrame, next_ref: ReferenceFrame) -> float:
    yaw0 = float(quaternion_yaw_deg(current_ref.state[3:7]))
    yaw1 = float(quaternion_yaw_deg(next_ref.state[3:7]))
    return float(wrap_angle_deg(yaw1 - yaw0))


def ensure_carla_available() -> None:
    if carla is None:
        raise RuntimeError("Failed to import carla. Install the CARLA Python API in the current environment first.")


def ensure_pygame_available() -> None:
    if pygame is None:
        raise RuntimeError("Failed to import pygame. Install pygame in the current environment first.")


class EdgeKeyReader:
    def __init__(self) -> None:
        self._last: Dict[int, bool] = {}

    def just_pressed(self, vk_code: int) -> bool:
        pressed = bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)
        prev = self._last.get(vk_code, False)
        self._last[vk_code] = pressed
        return pressed and not prev


def load_reference_frames(path: str) -> List[ReferenceFrame]:
    if not os.path.isabs(path):
        path = os.path.join(find_project_root(), path)
    frames: List[ReferenceFrame] = []
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"time", "pos_x", "pos_y", "pos_z", "rot_0", "rot_1", "rot_2", "rot_3"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Reference CSV is missing columns: {sorted(missing)}")

        for raw in reader:
            world_state = np.array(
                [
                    float(raw["pos_x"]),
                    float(raw["pos_y"]),
                    float(raw["pos_z"]),
                    float(raw["rot_0"]),
                    float(raw["rot_1"]),
                    float(raw["rot_2"]),
                    float(raw["rot_3"]),
                ],
                dtype=np.float32,
            )
            frames.append(
                ReferenceFrame(
                    time=float(raw["time"]),
                    action=np.array(
                        [
                            float(raw["throttle"]) if "throttle" in raw and raw["throttle"] not in {"", None} else 0.0,
                            float(raw["steering"]) if "steering" in raw and raw["steering"] not in {"", None} else 0.0,
                        ],
                        dtype=np.float32,
                    ),
                    state=world_state,
                    linear_speed=float(raw.get("linear_speed", 0.0) or 0.0),
                    source_row={key: float(value) for key, value in raw.items() if value not in {"", None}},
                )
            )
    if not frames:
        raise RuntimeError(f"No frames found in {path}")
    if all(abs(frame.linear_speed) < 1e-9 for frame in frames) and len(frames) > 1:
        for idx in range(len(frames) - 1):
            dt = max(frames[idx + 1].time - frames[idx].time, 1e-6)
            delta = frames[idx + 1].state[:3] - frames[idx].state[:3]
            frames[idx].linear_speed = float(np.linalg.norm(delta) / dt)
        frames[-1].linear_speed = frames[-2].linear_speed
    return frames


def resample_reference_frames(frames: List[ReferenceFrame], target_dt: float) -> List[ReferenceFrame]:
    if len(frames) < 2:
        return frames
    raw_dt = max(float(frames[1].time - frames[0].time), 1e-6)
    if raw_dt >= target_dt * 0.95:
        return frames

    raw_times = np.asarray([frame.time for frame in frames], dtype=np.float64)
    start_time = float(raw_times[0])
    end_time = float(raw_times[-1])
    target_times = np.arange(start_time, end_time + 1e-9, target_dt, dtype=np.float64)
    indices = np.searchsorted(raw_times, target_times, side="left")
    indices = np.clip(indices, 0, len(frames) - 1)

    time_scale = float(target_dt / raw_dt)
    resampled: List[ReferenceFrame] = []
    for idx, source_idx in enumerate(indices):
        src = frames[int(source_idx)]
        action = src.action.astype(np.float32, copy=True) * time_scale
        if action[0] >= 0.0:
            action[0] = np.clip(action[0], 0.0, 0.20)
        else:
            action[0] = np.clip(action[0], -0.20, 0.0)
        action[1] = np.clip(action[1], -0.45, 0.45)
        resampled.append(
            ReferenceFrame(
                time=float(target_times[idx]),
                action=action,
                state=src.state.astype(np.float32, copy=True),
                linear_speed=float(src.linear_speed),
                source_row=dict(src.source_row),
            )
        )

    if len(resampled) > 1:
        for idx in range(len(resampled) - 1):
            dt = max(resampled[idx + 1].time - resampled[idx].time, 1e-6)
            delta = resampled[idx + 1].state[:3] - resampled[idx].state[:3]
            resampled[idx].linear_speed = float(np.linalg.norm(delta) / dt)
        resampled[-1].linear_speed = resampled[-2].linear_speed
    return resampled


def estimate_forward_nominal_speed(forward_bundle: ModelBundle, state: np.ndarray, control_dt: float) -> float:
    device = next(forward_bundle.model.parameters()).device
    throttle_nominal = float(forward_bundle.normalizer.x_mean[7] + 2.0 * forward_bundle.normalizer.x_std[7])
    throttle_nominal = float(np.clip(throttle_nominal, 0.04, 0.12))
    history: Deque[np.ndarray] = deque(maxlen=forward_bundle.normalizer.seq_length)
    bootstrap_model_history(
        history,
        state.astype(np.float32, copy=True),
        np.array([throttle_nominal, 0.0], dtype=np.float32),
        forward_bundle.normalizer.seq_length,
    )
    hist_np = np.stack(list(history), axis=0).astype(np.float32)
    predicted_delta = predict_delta_state(hist_np, forward_bundle, device)
    planar_step = float(np.linalg.norm(predicted_delta[:2]))
    return max(planar_step / max(control_dt, 1e-6), 1e-3)


class CSVModelReplayController:
    def __init__(
        self,
        forward_bundle: ModelBundle,
        backward_bundle: ModelBundle,
        fixed_delta: float,
        actor: Optional["carla.Vehicle"] = None,
        data_root: str = "QCarDataSet",
        interpolation_alpha: float = 0.35,
        spawn_transform: Optional["carla.Transform"] = None,
        policy_agent: Optional[SACAgent] = None,
        policy_deterministic: bool = True,
        use_rl_correction: bool = USE_RL_CORRECTION_DEFAULT,
        debug_draw_enabled: bool = True,
    ) -> None:
        self.forward_bundle = forward_bundle
        self.backward_bundle = backward_bundle
        self.fixed_delta = fixed_delta
        self.actor = actor
        self.data_root = data_root
        self.interpolation_alpha = float(np.clip(interpolation_alpha, 0.0, 1.0))
        self.spawn_transform = spawn_transform
        self.policy_agent = policy_agent
        self.policy_deterministic = bool(policy_deterministic)
        self.use_rl_correction = bool(use_rl_correction)
        self.debug_draw_enabled = bool(debug_draw_enabled)
        self.seq_length = forward_bundle.normalizer.seq_length
        self.logs_dir = Path(find_project_root()) / "PDH_auto_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.current_folder_idx = -1
        self.current_file_idx = 0
        self.current_csv_files: List[str] = []
        self.reference_frames: List[ReferenceFrame] = []
        self.controller: Optional["_ReplayInstance"] = None
        self.loop_current_csv = False
        self.current_source_label = "IDLE"

    def load_csv_sequence(self, csv_files: List[str], label: str = "reference_trajectories") -> bool:
        if not csv_files:
            print(f"[error] No CSV files available for {label}")
            return False
        self.current_folder_idx = -1
        self.current_file_idx = 0
        self.current_csv_files = list(csv_files)
        self.loop_current_csv = False
        self.current_source_label = label
        return self.load_current_file()

    def load_standard_reference(self, reference_dir: Path, reference_idx: int) -> bool:
        if reference_idx < 0 or reference_idx >= len(STANDARD_REFERENCE_FILES):
            print(f"[error] Invalid standard reference index: {reference_idx}")
            return False
        csv_path = reference_dir / STANDARD_REFERENCE_FILES[reference_idx]
        if not csv_path.exists():
            print(f"[error] Missing standard reference CSV: {csv_path}")
            return False
        self.current_folder_idx = -1
        self.current_file_idx = reference_idx
        self.current_csv_files = [str(reference_dir / name) for name in STANDARD_REFERENCE_FILES if (reference_dir / name).exists()]
        if str(csv_path) not in self.current_csv_files:
            self.current_csv_files.insert(reference_idx, str(csv_path))
        self.current_file_idx = self.current_csv_files.index(str(csv_path))
        self.loop_current_csv = False
        self.current_source_label = "standard_references"
        return self.load_current_file()

    def load_current_file(self) -> bool:
        if self.current_file_idx >= len(self.current_csv_files):
            return False
        if self.actor is None:
            raise RuntimeError("CARLA actor is required for auto replay")

        csv_path = self.current_csv_files[self.current_file_idx]
        folder_name = self.current_source_label
        loop_marker = " [LOOP]" if self.loop_current_csv else ""
        print(
            f"\n[load] Model replay: {folder_name}/{os.path.basename(csv_path)} "
            f"({self.current_file_idx + 1}/{len(self.current_csv_files)}){loop_marker}"
        )

        try:
            raw_frames = load_reference_frames(csv_path)
            current_dt_state = extract_state_vector_from_vehicle(self.actor)
            nominal_speed = estimate_forward_nominal_speed(self.forward_bundle, current_dt_state, self.fixed_delta)
            raw_speeds = np.asarray([frame.linear_speed for frame in raw_frames if frame.linear_speed > 1e-6], dtype=np.float32)
            reference_speed = float(raw_speeds.mean()) if raw_speeds.size > 0 else nominal_speed
            raw_dt = max(float(raw_frames[1].time - raw_frames[0].time), 1e-6)
            route_dt = self.fixed_delta * nominal_speed / max(reference_speed, 1e-6)
            route_dt = float(np.clip(route_dt, raw_dt, self.fixed_delta))
            self.reference_frames = resample_reference_frames(raw_frames, route_dt)
            if len(self.reference_frames) < 2:
                print(f"[error] {csv_path} has insufficient frames ({len(self.reference_frames)})")
                return False

            self.controller = _ReplayInstance(
                forward_bundle=self.forward_bundle,
                backward_bundle=self.backward_bundle,
                reference_frames=self.reference_frames,
                fixed_delta=self.fixed_delta,
                actor=self.actor,
                seq_length=self.seq_length,
                interpolation_alpha=self.interpolation_alpha,
                csv_path=csv_path,
                policy_agent=self.policy_agent,
                policy_deterministic=self.policy_deterministic,
                use_rl_correction=self.use_rl_correction,
                debug_draw_enabled=self.debug_draw_enabled,
            )
            print(
                f"[info] Loaded {len(self.reference_frames)} action frames "
                f"(resampled from {len(raw_frames)}, raw_dt={raw_dt:.4f}, route_dt={route_dt:.4f}, "
                f"ref_speed={reference_speed:.3f}m/s, dt_nominal={nominal_speed:.3f}m/s), "
                f"ready to replay from current position"
            )
            return True
        except Exception as e:
            print(f"[error] Failed to load {csv_path}: {e}")
            return False

    def reset_vehicle(self) -> bool:
        if self.actor is None or self.spawn_transform is None:
            return False
        self.current_folder_idx = -1
        self.current_file_idx = 0
        self.current_csv_files = []
        self.reference_frames = []
        self.controller = None
        self.loop_current_csv = False
        self.current_source_label = "IDLE"
        self.actor.set_transform(self.spawn_transform)
        self.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self.actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        print("[control] RESET")
        return True

    def reset_current_file(self) -> bool:
        if self.current_folder_idx == -1:
            print("[error] No file loaded to reset")
            return False
        return self.load_current_file()

    def step_once(self, world: Optional["carla.World"] = None) -> bool:
        if self.controller is None:
            return False

        finished = not self.controller.step_once(world=world)
        if finished:
            if self.loop_current_csv:
                print("[loop] Repeating current file")
                return self.load_current_file()

            self.current_file_idx += 1
            if self.current_file_idx < len(self.current_csv_files):
                self.controller.save_records(self.logs_dir)
                if not self.load_current_file():
                    print("[error] Failed to load next file")
                    self.enter_idle()
                    return False
                return True

            self.controller.save_records(self.logs_dir)
            print(f"\n[complete] All files in {self.current_source_label} completed")
            self.enter_idle()
            return False

        return True

    def next_file(self) -> bool:
        if not self.current_csv_files:
            print("[error] No file loaded")
            return False
        if self.current_file_idx + 1 < len(self.current_csv_files):
            self.current_file_idx += 1
            self.loop_current_csv = False
            return self.load_current_file()
        print("[info] Already at last file")
        return False

    def prev_file(self) -> bool:
        if not self.current_csv_files:
            print("[error] No file loaded")
            return False
        if self.current_file_idx > 0:
            self.current_file_idx -= 1
            self.loop_current_csv = False
            return self.load_current_file()
        print("[info] Already at first file")
        return False

    def toggle_loop(self) -> None:
        self.loop_current_csv = not self.loop_current_csv
        print(f"[loop] Loop mode: {'ON' if self.loop_current_csv else 'OFF'}")

    def enter_idle(self) -> None:
        self.current_folder_idx = -1
        self.controller = None
        self.reference_frames = []
        self.loop_current_csv = False
        self.current_source_label = "IDLE"
        if self.actor is not None:
            self.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        print("[idle] Entering IDLE state")

    def is_idle(self) -> bool:
        return self.controller is None

    def get_current_info(self) -> str:
        if self.controller is None:
            return "IDLE"
        folder_name = self.current_source_label
        csv_name = os.path.basename(self.current_csv_files[self.current_file_idx])
        step_info = f"step={self.controller.step_idx}/{self.controller.total_steps()}"
        loop_marker = " [LOOP]" if self.loop_current_csv else ""
        return f"{folder_name}/{csv_name} {step_info}{loop_marker}"

    def draw_ui(self, screen: Optional["pygame.Surface"], font: Optional["pygame.font.Font"], paused: bool) -> None:
        if pygame is None or screen is None or font is None:
            return

        lines = [
            "CSV Model Replay (REFERENCE + CONTROLLER CORRECTION)",
            f"Status: {'PAUSED' if paused else 'RUNNING'}",
            f"Loop Mode: {'ON' if self.loop_current_csv else 'OFF'}",
            f"Current: {self.get_current_info()}",
            "",
            "Key mappings:",
            "1-6   : Select standard reference CSV",
            "SPACE : Pause/Resume",
            "R     : Reset vehicle to spawn point",
            "Q     : Toggle loop mode (repeat current CSV)",
            "LEFT  : Previous CSV file",
            "RIGHT : Next CSV file",
            "ESC   : Quit",
            "",
            (
                f"Spawn point: x={self.spawn_transform.location.x:.3f}, y={self.spawn_transform.location.y:.3f}"
                if self.spawn_transform is not None
                else "Spawn point: unavailable"
            ),
            "",
            "Standard references:",
        ]

        for idx, file_name in enumerate(STANDARD_REFERENCE_FILES):
            key_name = f"{idx + 1}"
            active = self.current_file_idx < len(self.current_csv_files) and os.path.basename(self.current_csv_files[self.current_file_idx]) == file_name
            highlight = "->" if active and self.current_source_label == "standard_references" else "  "
            lines.append(f"{highlight}{key_name}: {file_name}")

        screen.fill((20, 20, 20))
        y = 12
        for line in lines:
            img = font.render(line, True, (235, 235, 235))
            screen.blit(img, (12, y))
            y += 30
        pygame.display.flip()


class _ReplayInstance:
    def __init__(
        self,
        forward_bundle: ModelBundle,
        backward_bundle: ModelBundle,
        reference_frames: List[ReferenceFrame],
        fixed_delta: float,
        actor: "carla.Vehicle",
        seq_length: int,
        interpolation_alpha: float,
        csv_path: str,
        policy_agent: Optional[SACAgent],
        policy_deterministic: bool,
        use_rl_correction: bool,
        debug_draw_enabled: bool,
    ) -> None:
        self.forward_bundle = forward_bundle
        self.backward_bundle = backward_bundle
        self.reference_frames = reference_frames
        self.fixed_delta = fixed_delta
        self.actor = actor
        self.seq_length = seq_length
        self.interpolation_alpha = float(np.clip(interpolation_alpha, 0.0, 1.0))
        self.csv_path = csv_path
        self.policy_agent = policy_agent
        self.policy_deterministic = bool(policy_deterministic)
        self.use_rl_correction = bool(use_rl_correction)
        self.debug_draw_enabled = bool(debug_draw_enabled)
        self.step_idx = 0
        self.history: Deque[np.ndarray] = deque(maxlen=seq_length)
        self.model_state = extract_state_vector_from_vehicle(actor)
        self.prev_action: Optional[np.ndarray] = None
        self.records: List[Dict[str, float]] = []
        self.cumulative_state_error_sum = np.zeros(7, dtype=np.float32)
        self.last_speed_mps = 0.0
        self.stuck_steps = 0
        actor_tf = actor.get_transform()
        self.anchor_location = np.array([actor_tf.location.x, actor_tf.location.y, actor_tf.location.z], dtype=np.float32)
        self.anchor_yaw_deg = float(actor_tf.rotation.yaw)
        self.debug_ground_z = float(actor_tf.location.z + 0.08)
        self.debug_draw_period = 15
        self.debug_line_stride = 6
        self.reference_origin_xy = reference_frames[0].state[:2].astype(np.float32, copy=True)
        self.reference_origin_z = float(reference_frames[0].state[2])
        self.reference_origin_yaw_deg = float(quaternion_yaw_deg(reference_frames[0].state[3:7]))
        self.forward_action_low = np.array([max(0.0, float(self.forward_bundle.normalizer.x_mean[7] - 3.0 * self.forward_bundle.normalizer.x_std[7])), -0.45], dtype=np.float32)
        self.forward_action_high = np.array([min(0.20, float(self.forward_bundle.normalizer.x_mean[7] + 3.0 * self.forward_bundle.normalizer.x_std[7])), 0.45], dtype=np.float32)
        self.backward_action_low = np.array([max(-0.20, float(self.backward_bundle.normalizer.x_mean[7] - 3.0 * self.backward_bundle.normalizer.x_std[7])), -0.45], dtype=np.float32)
        self.backward_action_high = np.array([min(0.0, float(self.backward_bundle.normalizer.x_mean[7] + 3.0 * self.backward_bundle.normalizer.x_std[7])), 0.45], dtype=np.float32)
        self.reference_world_points = [self._flatten_debug_point(self._reference_target_pose(frame)[0]) for frame in reference_frames]
        self.state_scale = np.array([20.0, 20.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.error_scale = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.model_state = self._current_actor_route_state()
        bootstrap_model_history(
            self.history,
            self.model_state,
            self.reference_frames[0].action,
            self.seq_length,
        )
        actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))

    def total_steps(self) -> int:
        return len(self.reference_frames) - 1

    def _route_to_world_xy(self, route_xy: np.ndarray) -> np.ndarray:
        relative_xy = route_xy.astype(np.float32) - self.reference_origin_xy
        yaw_offset_deg = self.anchor_yaw_deg - self.reference_origin_yaw_deg
        yaw_offset_rad = np.radians(yaw_offset_deg)
        cos_y = float(np.cos(yaw_offset_rad))
        sin_y = float(np.sin(yaw_offset_rad))
        rotated_relative = np.array(
            [
                float(relative_xy[0] * cos_y - relative_xy[1] * sin_y),
                float(relative_xy[0] * sin_y + relative_xy[1] * cos_y),
            ],
            dtype=np.float32,
        )
        return self.anchor_location[:2] + rotated_relative

    def _world_to_route_xy(self, world_xy: np.ndarray) -> np.ndarray:
        delta_xy = world_xy.astype(np.float32) - self.anchor_location[:2]
        yaw_offset_deg = self.anchor_yaw_deg - self.reference_origin_yaw_deg
        yaw_offset_rad = np.radians(-yaw_offset_deg)
        cos_y = float(np.cos(yaw_offset_rad))
        sin_y = float(np.sin(yaw_offset_rad))
        local_delta = np.array(
            [
                float(delta_xy[0] * cos_y - delta_xy[1] * sin_y),
                float(delta_xy[0] * sin_y + delta_xy[1] * cos_y),
            ],
            dtype=np.float32,
        )
        return self.reference_origin_xy + local_delta

    def _planar_state_to_world_pose(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        target_xy = self._route_to_world_xy(state[:2])
        target_z = float(self.anchor_location[2] + (float(state[2]) - self.reference_origin_z))
        ref_yaw = float(quaternion_yaw_deg(state[3:7]))
        yaw_offset_deg = self.anchor_yaw_deg - self.reference_origin_yaw_deg
        target_yaw = wrap_angle_deg(ref_yaw + yaw_offset_deg)
        return np.array([float(target_xy[0]), float(target_xy[1]), target_z], dtype=np.float32), target_yaw

    def _reference_target_pose(self, frame: ReferenceFrame) -> tuple[np.ndarray, float]:
        return self._planar_state_to_world_pose(frame.state)

    def _current_actor_route_state(self) -> np.ndarray:
        actor_tf = self.actor.get_transform()
        route_xy = self._world_to_route_xy(np.array([actor_tf.location.x, actor_tf.location.y], dtype=np.float32))
        yaw_offset_deg = self.anchor_yaw_deg - self.reference_origin_yaw_deg
        route_yaw_deg = wrap_angle_deg(actor_tf.rotation.yaw - yaw_offset_deg)
        route_z = self.reference_origin_z + float(actor_tf.location.z - self.anchor_location[2])
        route_quat = yaw_deg_to_raw_quaternion(route_yaw_deg)
        return np.array([float(route_xy[0]), float(route_xy[1]), route_z, *route_quat.tolist()], dtype=np.float32)

    def _current_actor_speed_mps(self) -> float:
        velocity = self.actor.get_velocity()
        return float(np.linalg.norm([velocity.x, velocity.y, velocity.z]))

    def save_records(self, logs_dir: Path) -> None:
        if not self.records:
            return
        out_path = logs_dir / f"{Path(self.csv_path).stem}_auto_error.csv"
        fieldnames = list(self.records[0].keys())
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[log] Saved auto replay error log to {out_path}")

    def _normalize_vector(self, values: np.ndarray, scales: np.ndarray) -> np.ndarray:
        normalized = np.asarray(values, dtype=np.float32) / np.asarray(scales, dtype=np.float32)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        return np.array(
            [
                np.clip((action[0] / 0.12) * 2.0 - 1.0, -1.0, 1.0),
                np.clip(action[1] / 0.45, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _build_policy_observation(self, current_state: np.ndarray, current_ref: ReferenceFrame, next_ref: ReferenceFrame) -> np.ndarray:
        del next_ref
        tracking_error = current_state - current_ref.state
        prev_action = self.prev_action if self.prev_action is not None else np.zeros(2, dtype=np.float32)
        current_loss = float(np.mean(np.square(tracking_error)))
        history_steps = max(1, self.step_idx)
        cumulative_mean_loss = float(np.mean(self.cumulative_state_error_sum / float(history_steps))) if self.step_idx > 0 else current_loss
        yaw_error_deg = float(
            wrap_angle_deg(
                quaternion_yaw_deg(current_state[3:7]) - quaternion_yaw_deg(current_ref.state[3:7])
            )
        )
        return np.concatenate(
            [
                self._normalize_vector(current_state, self.state_scale),
                self._normalize_vector(current_ref.state, self.state_scale),
                self._normalize_vector(tracking_error, self.error_scale),
                self._normalize_action(prev_action),
                np.array(
                    [
                        np.clip(cumulative_mean_loss / 4.0, -1.0, 1.0),
                        np.clip(current_loss / 4.0, -1.0, 1.0),
                        np.clip(yaw_error_deg / 30.0, -1.0, 1.0),
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        ).astype(np.float32)

    def _choose_corrected_action(
        self,
        current_state: np.ndarray,
        current_ref: ReferenceFrame,
        next_ref: ReferenceFrame,
        observation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        baseline_action = self._build_baseline_action(current_state, current_ref, next_ref)
        if self.use_rl_correction and self.policy_agent is not None:
            residual_action = self.policy_agent.select_action(observation, deterministic=self.policy_deterministic).astype(np.float32)
            residual_action = np.array(
                [
                    float(np.clip(residual_action[0], -0.06, 0.06)),
                    float(np.clip(residual_action[1], -0.25, 0.25)),
                ],
                dtype=np.float32,
            )
            applied_action = self._clip_action_for_dt(baseline_action + residual_action)
            return residual_action, applied_action

        baseline_action = self._clip_action_for_dt(baseline_action)
        return baseline_action.copy(), baseline_action.copy()

    def _build_baseline_action(
        self,
        current_state: np.ndarray,
        current_ref: ReferenceFrame,
        next_ref: ReferenceFrame,
    ) -> np.ndarray:
        dt = max(float(next_ref.time - current_ref.time), 1e-6)
        yaw_deg = float(quaternion_yaw_deg(current_state[3:7]))
        yaw_rad = math.radians(yaw_deg)
        cos_y = float(math.cos(yaw_rad))
        sin_y = float(math.sin(yaw_rad))

        target_vec = next_ref.state[:2] - current_state[:2]
        forward_err = float(target_vec[0] * cos_y + target_vec[1] * sin_y)
        lateral_err = float(-target_vec[0] * sin_y + target_vec[1] * cos_y)
        yaw_target_deg = float(quaternion_yaw_deg(next_ref.state[3:7]))
        yaw_err_deg = float(wrap_angle_deg(yaw_target_deg - yaw_deg))
        target_step = float(np.linalg.norm(next_ref.state[:2] - current_ref.state[:2]))
        target_speed = target_step / dt

        throttle = 0.04 + 0.60 * max(0.0, forward_err) + 0.03 * target_speed
        if forward_err < -0.05:
            throttle = 0.035
        throttle = float(np.clip(throttle, 0.035, 0.12))

        steer = 0.85 * lateral_err + 0.020 * yaw_err_deg
        steer = float(np.clip(steer, -0.45, 0.45))
        return np.array([throttle, steer], dtype=np.float32)

    def _apply_launch_assist(
        self,
        current_state: np.ndarray,
        next_ref: ReferenceFrame,
        action: np.ndarray,
    ) -> np.ndarray:
        adjusted = np.asarray(action, dtype=np.float32).copy()
        speed_mps = self._current_actor_speed_mps()
        yaw_deg = float(quaternion_yaw_deg(current_state[3:7]))
        yaw_rad = math.radians(yaw_deg)
        cos_y = float(math.cos(yaw_rad))
        sin_y = float(math.sin(yaw_rad))
        target_vec = next_ref.state[:2] - current_state[:2]
        forward_err = float(target_vec[0] * cos_y + target_vec[1] * sin_y)
        target_distance = float(np.linalg.norm(target_vec))
        should_drive = target_distance > 0.03
        if should_drive and speed_mps < 0.12:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        if should_drive and (speed_mps < 0.05 or self.stuck_steps >= 3 or forward_err > 0.02):
            adjusted[0] = max(float(adjusted[0]), 0.12)
            adjusted[1] = float(np.clip(adjusted[1], -0.08, 0.08))
        return self._clip_action_for_dt(adjusted)

    def _clip_action_for_dt(self, action: np.ndarray) -> np.ndarray:
        clipped = np.asarray(action, dtype=np.float32).copy()
        if clipped[0] >= 0.0:
            clipped = np.clip(clipped, self.forward_action_low, self.forward_action_high)
        else:
            clipped = np.clip(clipped, self.backward_action_low, self.backward_action_high)
        return clipped.astype(np.float32)

    def _history_array(self, current_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        feature = np.concatenate([current_state, action], axis=0).astype(np.float32)
        temp_history = deque(self.history, maxlen=self.seq_length)
        if not temp_history:
            for _ in range(self.seq_length):
                temp_history.append(feature.copy())
        else:
            temp_history.append(feature.copy())
            while len(temp_history) < self.seq_length:
                temp_history.appendleft(temp_history[0].copy())
        return np.stack(list(temp_history), axis=0).astype(np.float32)

    def _flatten_debug_point(self, point_xyz: np.ndarray) -> np.ndarray:
        flat = np.asarray(point_xyz, dtype=np.float32).copy()
        flat[2] = self.debug_ground_z
        return flat

    def _predict_next_route_state(self, current_state: np.ndarray, predicted_delta: np.ndarray) -> np.ndarray:
        next_state = current_state.copy().astype(np.float32)
        yaw_deg = float(quaternion_yaw_deg(current_state[3:7]))
        yaw_rad = np.radians(yaw_deg)
        cos_y = float(np.cos(yaw_rad))
        sin_y = float(np.sin(yaw_rad))
        dx_body = float(predicted_delta[0])
        dy_body = float(predicted_delta[1])
        dz_body = float(predicted_delta[2])
        next_state[0] = float(current_state[0] + dx_body * cos_y - dy_body * sin_y)
        next_state[1] = float(current_state[1] + dx_body * sin_y + dy_body * cos_y)
        next_state[2] = float(current_state[2] + dz_body)
        next_yaw_deg = wrap_angle_deg(yaw_deg + float(predicted_delta[3]))
        next_state[3:7] = yaw_deg_to_raw_quaternion(next_yaw_deg)
        return next_state.astype(np.float32)

    def _draw_debug_reference(self, world: Optional["carla.World"], ref_target_pos: np.ndarray) -> None:
        if (not self.debug_draw_enabled) or world is None or carla is None:
            return
        if self.step_idx % self.debug_draw_period != 0:
            return
        debug = world.debug
        life_time = 0.75
        target_flat = self._flatten_debug_point(ref_target_pos)

        if len(self.reference_world_points) >= 2:
            for idx in range(0, len(self.reference_world_points) - 1, self.debug_line_stride):
                p0 = self.reference_world_points[idx]
                p1 = self.reference_world_points[min(idx + self.debug_line_stride, len(self.reference_world_points) - 1)]
                debug.draw_line(
                    carla.Location(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                    carla.Location(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
                    thickness=0.02,
                    color=carla.Color(r=180, g=30, b=30),
                    life_time=life_time,
                )

        actor_tf = self.actor.get_transform()
        actor_loc = actor_tf.location
        debug.draw_point(
            carla.Location(x=float(target_flat[0]), y=float(target_flat[1]), z=float(target_flat[2] + 0.03)),
            size=0.06,
            color=carla.Color(r=40, g=200, b=40),
            life_time=life_time,
        )
        debug.draw_line(
            carla.Location(x=actor_loc.x, y=actor_loc.y, z=self.debug_ground_z + 0.03),
            carla.Location(x=float(target_flat[0]), y=float(target_flat[1]), z=float(target_flat[2] + 0.03)),
            thickness=0.025,
            color=carla.Color(r=220, g=180, b=0),
            life_time=life_time,
        )

    def step_once(self, world: Optional["carla.World"] = None) -> bool:
        if self.step_idx >= self.total_steps():
            return False

        current_ref = self.reference_frames[self.step_idx]
        next_ref = self.reference_frames[self.step_idx + 1]
        current_route_state = self._current_actor_route_state()
        self.model_state = current_route_state.copy()
        observation = self._build_policy_observation(current_route_state, current_ref, next_ref)
        ref_target_pos, ref_target_yaw = self._reference_target_pose(next_ref)
        target_yaw_step = target_yaw_step_deg(current_ref, next_ref)
        policy_action, action = self._choose_corrected_action(current_route_state, current_ref, next_ref, observation)
        action = self._apply_launch_assist(current_route_state, next_ref, action)
        bundle = choose_bundle_for_action(action, self.forward_bundle, self.backward_bundle)
        device = next(bundle.model.parameters()).device
        hist_np = self._history_array(current_route_state, action)
        predicted_delta = predict_delta_state(hist_np, bundle, device)
        predicted_route_state = self._predict_next_route_state(current_route_state, predicted_delta)

        control = build_vehicle_control(float(max(0.0, action[0])), float(action[1]))
        self.actor.apply_control(control)

        if world is not None:
            world.tick()
            follow_vehicle_with_spectator(world, self.actor)

        actual_route_state = self._current_actor_route_state()
        self.model_state = actual_route_state.copy()
        self.last_speed_mps = self._current_actor_speed_mps()
        bundle_name = f"{bundle.name}_control"
        self._draw_debug_reference(world, ref_target_pos)
        actor_tf = self.actor.get_transform()
        pos_error = np.array(
            [
                float(actor_tf.location.x - ref_target_pos[0]),
                float(actor_tf.location.y - ref_target_pos[1]),
                float(actor_tf.location.z - ref_target_pos[2]),
            ],
            dtype=np.float32,
        )
        yaw_error = wrap_angle_deg(actor_tf.rotation.yaw - ref_target_yaw)
        state_sq_error = np.square(actual_route_state - next_ref.state).astype(np.float32)
        self.cumulative_state_error_sum += state_sq_error
        current_loss = float(np.mean(state_sq_error))
        cumulative_mean_loss = float(np.mean(self.cumulative_state_error_sum / float(max(1, self.step_idx + 1))))
        throttle_excess = max(0.0, float(action[0]) - 0.10)
        steer_excess = max(0.0, abs(float(action[1])) - 0.40)
        saturation_penalty = 2.0 * (throttle_excess * throttle_excess + steer_excess * steer_excess)
        smooth_penalty = 0.0 if self.prev_action is None else 0.25 * float(np.mean(np.square(action - self.prev_action)))
        yaw_penalty = 0.35 * (abs(float(yaw_error)) / 30.0) ** 2
        total_loss = 0.5 * cumulative_mean_loss + 0.5 * current_loss + yaw_penalty + saturation_penalty + smooth_penalty
        self.records.append(
            {
                "step": int(self.step_idx),
                "time": float(current_ref.time),
                "reference_dx": float(next_ref.state[0] - current_ref.state[0]),
                "reference_dy": float(next_ref.state[1] - current_ref.state[1]),
                "policy_throttle": float(policy_action[0]),
                "policy_steer": float(policy_action[1]),
                "action_throttle": float(action[0]),
                "action_steer": float(action[1]),
                "control_throttle": float(control.throttle),
                "control_steer": float(control.steer),
                "control_brake": float(control.brake),
                "control_reverse": int(bool(control.reverse)),
                "control_hand_brake": int(bool(control.hand_brake)),
                "predicted_delta_x": float(predicted_delta[0]),
                "predicted_delta_y": float(predicted_delta[1]),
                "predicted_delta_z": float(predicted_delta[2]),
                "predicted_delta_yaw": float(predicted_delta[3]),
                "target_yaw_step_deg": float(target_yaw_step),
                "ref_x": float(ref_target_pos[0]),
                "ref_y": float(ref_target_pos[1]),
                "ref_z": float(ref_target_pos[2]),
                "ref_yaw_deg": float(ref_target_yaw),
                "pred_x": float(actor_tf.location.x),
                "pred_y": float(actor_tf.location.y),
                "pred_z": float(actor_tf.location.z),
                "pred_yaw_deg": float(actor_tf.rotation.yaw),
                "speed_mps": float(self.last_speed_mps),
                "stuck_steps": int(self.stuck_steps),
                "error_x": float(pos_error[0]),
                "error_y": float(pos_error[1]),
                "error_z": float(pos_error[2]),
                "error_pos_l2": float(np.linalg.norm(pos_error)),
                "error_yaw_deg": float(yaw_error),
                "current_loss": current_loss,
                "cumulative_mean_loss": cumulative_mean_loss,
                "total_loss": total_loss,
            }
        )

        self.history.append(np.concatenate([current_route_state, action], axis=0).astype(np.float32))
        self.prev_action = action.copy()
        self.step_idx += 1

        if self.step_idx == 1 or self.step_idx % 50 == 0:
            print(
                f"[step={self.step_idx:05d}/{self.total_steps()}] model={bundle_name} "
                f"ref=({next_ref.state[0] - current_ref.state[0]:+.3f},{next_ref.state[1] - current_ref.state[1]:+.3f}) "
                f"policy=({policy_action[0]:+.3f},{policy_action[1]:+.3f}) "
                f"applied=({action[0]:+.3f},{action[1]:+.3f}) "
                f"control=(thr={control.throttle:+.3f},ste={control.steer:+.3f},brk={control.brake:+.3f},rev={int(bool(control.reverse))},hb={int(bool(control.hand_brake))}) "
                f"delta=({predicted_delta[0]:+.4f},{predicted_delta[1]:+.4f},{predicted_delta[2]:+.4f},{predicted_delta[3]:+.4f}) "
                f"speed={self.last_speed_mps:.3f} "
                f"stuck={self.stuck_steps} "
                f"yaw_target={target_yaw_step:+.4f} "
                f"err={total_loss:.3f}"
            )

        return True


def parse_args() -> argparse.Namespace:
    project_root = find_project_root()
    parser = argparse.ArgumentParser(description="CSV state tracking controller driven by DT + optional RL")
    parser.add_argument("--forward-model-path", default=os.path.join(project_root, "PDHModel", "forward_world_model.pth"))
    parser.add_argument("--forward-norm-path", default=os.path.join(project_root, "PDHModel", "forward_normalization.pt"))
    parser.add_argument("--backward-model-path", default=os.path.join(project_root, "PDHModel", "backward_world_model.pth"))
    parser.add_argument("--backward-norm-path", default=os.path.join(project_root, "PDHModel", "backward_normalization.pt"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--fixed-delta", type=float, default=0.05)
    parser.add_argument("--data-root", default="QCarDataSet")
    parser.add_argument("--reference-dir", default=os.path.join(project_root, "PDHModel", "reference_trajectories"))
    parser.add_argument("--reference-csv", default=None)
    parser.add_argument("--policy-path", default=os.path.join(project_root, "PDHModel", "spec_rl_resampled_fix1", "policy_controller.pth"))
    parser.add_argument("--use-rl-correction", action=argparse.BooleanOptionalAction, default=USE_RL_CORRECTION_DEFAULT)
    parser.add_argument("--policy-stochastic", action="store_true")
    parser.add_argument("--no-debug-draw", action="store_true")
    parser.add_argument("--no-carla", action="store_true", help="Run pure model replay without CARLA visualization")
    parser.add_argument("--interpolation-alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forward_bundle = load_bundle("forward", args.forward_model_path, args.forward_norm_path, device)
    backward_bundle = load_bundle("backward", args.backward_model_path, args.backward_norm_path, device)
    policy_agent = load_policy_agent(Path(args.policy_path), device) if args.policy_path else None
    reference_dir = Path(args.reference_dir)

    print(f"[debug] forward seq_length={forward_bundle.normalizer.seq_length}")
    print(f"[debug] forward action mean/std={forward_bundle.normalizer.x_mean[7:9].cpu().numpy()} / {forward_bundle.normalizer.x_std[7:9].cpu().numpy()}")
    print(f"[debug] backward seq_length={backward_bundle.normalizer.seq_length}")
    print(f"[debug] backward action mean/std={backward_bundle.normalizer.x_mean[7:9].cpu().numpy()} / {backward_bundle.normalizer.x_std[7:9].cpu().numpy()}")

    ensure_pygame_available()
    pygame.init()
    pygame.display.set_caption("CARLA CSV Model Replay")
    pygame.display.set_mode((1280, 760))
    font = pygame.font.SysFont(None, 24)
    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()

    actor = None
    world = None
    original_settings = None
    key_reader = EdgeKeyReader()
    paused = False
    spawn_transform = None

    try:
        if not args.no_carla:
            ensure_carla_available()
            client = carla.Client(args.host, args.port)
            client.set_timeout(20.0)
            world = client.get_world()
            original_settings = world.get_settings()
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = args.fixed_delta
            world.apply_settings(settings)

            spawn_transform = choose_spawn_transform(world, args.spawn_index)
            actor = spawn_vehicle(world, args.vehicle_filter, spawn_transform)
            actor.set_autopilot(False)
            actor.set_transform(spawn_transform)
            actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            world.tick()
            world.tick()
            follow_vehicle_with_spectator(world, actor, camera_state=None)

        controller = CSVModelReplayController(
            forward_bundle=forward_bundle,
            backward_bundle=backward_bundle,
            fixed_delta=args.fixed_delta,
            actor=actor,
            data_root=args.data_root,
            interpolation_alpha=args.interpolation_alpha,
            spawn_transform=spawn_transform,
            policy_agent=policy_agent,
            policy_deterministic=not args.policy_stochastic,
            use_rl_correction=args.use_rl_correction,
            debug_draw_enabled=not args.no_debug_draw,
        )

        print("=" * 80)
        print("CSV MODEL Replay Started (RL TRACKING ON DT WORLD MODEL)")
        print(f"Interpolation alpha: {args.interpolation_alpha:.2f}")
        print(f"Controller mode: {'RL direct action on DT world model' if args.use_rl_correction else 'Baseline tracker on DT world model'}")
        print("Press 1-6 to select a standard reference CSV")
        print("SPACE: Pause/Resume | R: Reset vehicle to spawn point")
        print("LEFT/RIGHT: Previous/Next CSV file")
        print("ESC: Quit")
        print("=" * 80)

        auto_loaded = False
        if args.reference_csv:
            auto_loaded = controller.load_csv_sequence([args.reference_csv], label="reference_csv")
        if auto_loaded:
            paused = False
        else:
            paused = True
            print("[idle] Waiting for key 1-6 to load a reference trajectory")

        def reset_vehicle_to_spawn() -> None:
            if controller.reset_vehicle():
                if world is not None and actor is not None:
                    world.tick()
                    follow_vehicle_with_spectator(world, actor, camera_state=None)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("[control] PAUSE" if paused else "[control] RESUME")
                    elif event.key == pygame.K_r:
                        reset_vehicle_to_spawn()
                    elif event.key == pygame.K_q:
                        if not controller.is_idle():
                            controller.toggle_loop()
                    elif event.key == pygame.K_LEFT:
                        if controller.prev_file():
                            paused = False
                    elif event.key == pygame.K_RIGHT:
                        if controller.next_file():
                            paused = False
                    elif event.key in KEY_MAP:
                        reference_idx = KEY_MAP[event.key]
                        if reference_idx < len(STANDARD_REFERENCE_FILES):
                            print(f"\n[control] Loading reference {STANDARD_REFERENCE_FILES[reference_idx]}...")
                            if controller.load_standard_reference(reference_dir, reference_idx):
                                paused = False

            if key_reader.just_pressed(VK_ESC):
                running = False
            if key_reader.just_pressed(VK_R):
                reset_vehicle_to_spawn()
            if key_reader.just_pressed(VK_Q):
                if not controller.is_idle():
                    controller.toggle_loop()
            if key_reader.just_pressed(VK_LEFT):
                if controller.prev_file():
                    paused = False
            if key_reader.just_pressed(VK_RIGHT):
                if controller.next_file():
                    paused = False
            if key_reader.just_pressed(VK_SPACE):
                paused = not paused
                print("[control] PAUSE" if paused else "[control] RESUME")

            for vk_code, reference_idx in [(0x31, 0), (0x32, 1), (0x33, 2), (0x34, 3), (0x35, 4), (0x36, 5)]:
                if key_reader.just_pressed(vk_code) and reference_idx < len(STANDARD_REFERENCE_FILES):
                    print(f"\n[control] Loading reference {STANDARD_REFERENCE_FILES[reference_idx]}...")
                    if controller.load_standard_reference(reference_dir, reference_idx):
                        paused = False

            if not paused and not controller.is_idle():
                controller.step_once(world=world)
            elif not paused and world is not None and actor is not None:
                world.tick()
                follow_vehicle_with_spectator(world, actor)

            controller.draw_ui(screen, font, paused)
            clock.tick(60)

    finally:
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass
        if actor is not None:
            try:
                actor.destroy()
            except Exception:
                pass
        if pygame is not None:
            pygame.quit()


if __name__ == "__main__":
    main()
