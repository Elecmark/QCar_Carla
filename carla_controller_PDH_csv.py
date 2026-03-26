import argparse
import ctypes
import csv
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

try:
    import carla
except ImportError:
    carla = None

from carla_controller_PDH import (
    find_project_root,
    follow_vehicle_with_spectator,
    model_position_to_carla_xyz,
    model_quat_to_carla_yaw_deg,
    model_state_to_carla_transform,
    spawn_vehicle,
    choose_spawn_transform,
    wrap_angle_deg,
)

STATE_COLUMNS = ["pos_x", "pos_y", "pos_z", "rot_0", "rot_1", "rot_2", "rot_3"]
ACTION_COLUMNS = ["throttle", "steering"]
INPUT_COLUMNS = STATE_COLUMNS + ACTION_COLUMNS
POSITION_SCALE = 10.0
INTERPOLATION_STEPS = 4
MIN_HEADING_MOVE = 0.05
PLAYBACK_FPS = 60.0
HEADING_BLEND_ALPHA = 0.35
MAX_FRAME_YAW_DEG = 12.0

# 按键映射（数字键 1-0 和 - =）
KEY_MAP = {
    pygame.K_1: 0,
    pygame.K_2: 1,
    pygame.K_3: 2,
    pygame.K_4: 3,
    pygame.K_5: 4,
    pygame.K_6: 5,
    pygame.K_7: 6,
    pygame.K_8: 7,
    pygame.K_9: 8,
    pygame.K_0: 9,
    pygame.K_MINUS: 10,
    pygame.K_EQUALS: 11,
}

# 文件夹列表（按顺序对应按键）
FOLDER_LIST = [
    "linefollow_constant",
    "linefollow_quadratic",
    "linefollow_sin",
    "linefollow_squareroot",
    "linefollow_triangle",
    "manual_clockwise_forward",
    "manual_counter_forward",
    "manual_clockwise_backward",
    "manual_counter_backward",
    "openloop_constant",
    "openloop_quadratic",
    "openloop_sin",
    "openloop_squareroot",
    "openloop_triangle",
]

VK_SPACE = 0x20
VK_R = 0x52
VK_Q = 0x51
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_ESC = 0x1B

@dataclass
class ReferenceFrame:
    time: float
    action: np.ndarray
    state: np.ndarray
    linear_speed: float
    source_row: Dict[str, float]
    body_delta: np.ndarray = None
    delta_yaw: float = 0.0
    carla_position: np.ndarray = None
    carla_yaw_deg: float = 0.0


@dataclass
class PendingMotion:
    world_delta: np.ndarray
    delta_yaw: float
    total_steps: int
    completed_steps: int = 0


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


def get_csv_files_in_folder(folder_name: str, data_root: str = "QCarDataSet") -> List[str]:
    """获取指定文件夹下所有 CSV 文件路径"""
    folder_path = Path(data_root) / folder_name
    if not folder_path.exists():
        return []
    return sorted([str(p) for p in folder_path.glob("*.csv")])


def compute_relative_motion(frames: List[ReferenceFrame]) -> List[ReferenceFrame]:
    """Compute consecutive body-frame deltas from reference trajectories."""
    if len(frames) < 2:
        return frames

    for frame in frames:
        frame.carla_position = model_position_to_carla_xyz(frame.state[:3]).astype(np.float32)
        frame.carla_yaw_deg = float(model_quat_to_carla_yaw_deg(frame.state[3:7]))

    frames[0].body_delta = np.zeros(3, dtype=np.float32)
    frames[0].delta_yaw = 0.0
    for i in range(1, len(frames)):
        prev_frame = frames[i - 1]
        frame = frames[i]
        delta_world = frame.carla_position - prev_frame.carla_position
        prev_yaw_deg = prev_frame.carla_yaw_deg
        yaw_rad = math.radians(prev_yaw_deg)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        frame.body_delta = np.array(
            [
                float(delta_world[0] * cos_y + delta_world[1] * sin_y),
                float(-delta_world[0] * sin_y + delta_world[1] * cos_y),
                float(delta_world[2]),
            ],
            dtype=np.float32,
        )
        frame.delta_yaw = wrap_angle_deg(frame.carla_yaw_deg - prev_yaw_deg)
    return frames


def load_reference_frames(path: str) -> List[ReferenceFrame]:
    """从 CSV 加载参考帧，并计算相对位移"""
    if not os.path.isabs(path):
        path = os.path.join(find_project_root(), path)
    frames: List[ReferenceFrame] = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"time", "linear_speed", *INPUT_COLUMNS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Reference CSV is missing columns: {sorted(missing)}")

        for raw in reader:
            action = np.array([float(raw["throttle"]), float(raw["steering"])], dtype=np.float32)
            state = np.array([float(raw[column]) for column in STATE_COLUMNS], dtype=np.float32)
            # 归一化四元数
            norm = np.linalg.norm(state[3:7])
            if norm > 0:
                state[3:7] = state[3:7] / norm
            frames.append(
                ReferenceFrame(
                    time=float(raw["time"]),
                    action=action,
                    state=state,
                    linear_speed=float(raw["linear_speed"]),
                    source_row={key: float(value) for key, value in raw.items() if value not in {"", None}},
                )
            )
    if not frames:
        raise RuntimeError(f"No frames found in {path}")

    # 计算相对位移
    frames = compute_relative_motion(frames)
    return frames


def apply_relative_state_to_actor(
    actor: "carla.Vehicle",
    frame: ReferenceFrame,
) -> tuple[np.ndarray, float]:
    """Convert one reference step into a world-space delta and desired yaw change."""
    if frame.body_delta is None:
        return np.zeros(3, dtype=np.float32), 0.0

    tf = actor.get_transform()
    yaw_rad = math.radians(tf.rotation.yaw)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    dx_body = float(frame.body_delta[0])
    dy_body = float(frame.body_delta[1])
    dz_body = float(frame.body_delta[2])

    dx_world = (dx_body * cos_y - dy_body * sin_y) * POSITION_SCALE
    dy_world = (dx_body * sin_y + dy_body * cos_y) * POSITION_SCALE
    dz_world = dz_body * POSITION_SCALE

    move_norm = math.hypot(dx_world, dy_world)
    if move_norm > MIN_HEADING_MOVE:
        applied_delta_yaw = wrap_angle_deg(math.degrees(math.atan2(dy_world, dx_world)) - tf.rotation.yaw)
    else:
        applied_delta_yaw = float(frame.delta_yaw)

    return np.array([dx_world, dy_world, dz_world], dtype=np.float32), float(applied_delta_yaw)


class CSVForcePlayController:
    """强制执行 CSV 数据的控制器（基于相对位移）"""

    def __init__(
            self,
            fixed_delta: float,
            actor: Optional["carla.Vehicle"] = None,
            data_root: str = "QCarDataSet",
            spawn_transform: Optional["carla.Transform"] = None,
    ) -> None:
        self.fixed_delta = fixed_delta
        self.actor = actor
        self.data_root = data_root
        self.spawn_transform = spawn_transform

        # 当前播放状态
        self.current_folder_idx = -1
        self.current_file_idx = 0
        self.current_csv_files: List[str] = []
        self.reference_frames: List[ReferenceFrame] = []
        self.current_frame_idx = 0
        self.last_time = 0.0

        # 循环播放标志
        self.loop_current_csv = False
        self.coast_velocity = np.zeros(3, dtype=np.float32)
        self.coast_yaw_rate_deg = 0.0
        self.coast_decay = 0.90
        self.pending_motion: Optional[PendingMotion] = None
        self.anchor_location = np.zeros(3, dtype=np.float32)
        self.anchor_yaw_deg = 0.0
        self.reference_origin = np.zeros(3, dtype=np.float32)
        self.reference_origin_yaw_deg = 0.0

        # 缓存所有文件夹的 CSV 文件列表
        self.folder_csv_map: Dict[int, List[str]] = {}
        for idx, folder_name in enumerate(FOLDER_LIST):
            csv_files = get_csv_files_in_folder(folder_name, data_root)
            if csv_files:
                self.folder_csv_map[idx] = csv_files
                print(f"[init] {folder_name}: {len(csv_files)} CSV files")
            else:
                print(f"[init] {folder_name}: No CSV files found")

    def load_folder(self, folder_idx: int) -> bool:
        """加载指定文件夹的数据（可打断当前播放）"""
        if folder_idx not in self.folder_csv_map:
            print(f"[error] Folder index {folder_idx} not available")
            return False

        csv_files = self.folder_csv_map[folder_idx]
        if not csv_files:
            print(f"[error] No CSV files in folder {FOLDER_LIST[folder_idx]}")
            return False

        # 完全重置状态（打断当前播放）
        self.current_folder_idx = folder_idx
        self.current_file_idx = 0
        self.current_csv_files = csv_files
        self.loop_current_csv = False

        # 加载第一个文件
        return self.load_current_file()

    def load_current_file(self) -> bool:
        """加载当前索引的文件（不移动车辆）"""
        if self.current_file_idx >= len(self.current_csv_files):
            return False

        csv_path = self.current_csv_files[self.current_file_idx]
        folder_name = FOLDER_LIST[self.current_folder_idx]
        loop_marker = " [LOOP]" if self.loop_current_csv else ""
        print(
            f"\n[load] Standard-answer replay: {folder_name}/{os.path.basename(csv_path)} ({self.current_file_idx + 1}/{len(self.current_csv_files)}){loop_marker}")

        try:
            self.reference_frames = load_reference_frames(csv_path)
            if len(self.reference_frames) < 2:
                print(f"[error] {csv_path} has insufficient frames ({len(self.reference_frames)})")
                return False

            # 重置到第一帧（只是索引，不移动车辆）
            self.current_frame_idx = 0
            self.last_time = self.reference_frames[0].time
            self.coast_velocity[:] = 0.0
            self.coast_yaw_rate_deg = 0.0
            self.pending_motion = None
            if self.actor is not None:
                tf = self.actor.get_transform()
                self.anchor_location = np.array([tf.location.x, tf.location.y, tf.location.z], dtype=np.float32)
                self.anchor_yaw_deg = float(tf.rotation.yaw)
            self.reference_origin = self.reference_frames[0].carla_position.copy()
            self.reference_origin_yaw_deg = float(self.reference_frames[0].carla_yaw_deg)

            print(f"[info] Loaded {len(self.reference_frames)} frames, ready to play from current position")
            return True
        except Exception as e:
            print(f"[error] Failed to load {csv_path}: {e}")
            return False

    def reset_vehicle(self) -> bool:
        """Reset actor to spawn and interrupt playback."""
        if self.actor is not None and self.spawn_transform is not None:
            self.current_folder_idx = -1
            self.current_file_idx = 0
            self.current_csv_files = []
            self.reference_frames = []
            self.current_frame_idx = 0
            self.last_time = 0.0
            self.loop_current_csv = False
            self.actor.set_transform(self.spawn_transform)
            self.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.coast_velocity[:] = 0.0
            self.coast_yaw_rate_deg = 0.0
            self.pending_motion = None
            print("[control] RESET")
            return True
        return False

    def reset_current_file(self) -> bool:
        """重置 CSV 播放进度（不移动车辆）"""
        if self.current_folder_idx == -1:
            print("[error] No file loaded to reset")
            return False

        if self.reference_frames:
            self.current_frame_idx = 0
            self.pending_motion = None
            print(f"[control] RESET playback to start (position unchanged)")
            return True

        return False

    def _apply_pending_motion_step(self) -> None:
        if self.actor is None or self.pending_motion is None:
            return
        remaining_steps = max(1, self.pending_motion.total_steps - self.pending_motion.completed_steps)
        step_delta = self.pending_motion.world_delta / remaining_steps
        step_yaw = self.pending_motion.delta_yaw / remaining_steps
        tf = self.actor.get_transform()
        self.actor.set_transform(
            carla.Transform(
                carla.Location(
                    x=tf.location.x + float(step_delta[0]),
                    y=tf.location.y + float(step_delta[1]),
                    z=tf.location.z + float(step_delta[2]),
                ),
                carla.Rotation(
                    roll=tf.rotation.roll,
                    pitch=tf.rotation.pitch,
                    yaw=tf.rotation.yaw + float(step_yaw),
                ),
            )
        )
        self.pending_motion.world_delta = self.pending_motion.world_delta - step_delta
        self.pending_motion.delta_yaw = float(self.pending_motion.delta_yaw - step_yaw)
        self.pending_motion.completed_steps += 1

    def _frame_target_pose(self, frame: ReferenceFrame) -> tuple[np.ndarray, float]:
        relative_position = (frame.carla_position - self.reference_origin) * POSITION_SCALE
        yaw_offset_deg = self.anchor_yaw_deg - self.reference_origin_yaw_deg
        yaw_offset_rad = math.radians(yaw_offset_deg)
        cos_y = math.cos(yaw_offset_rad)
        sin_y = math.sin(yaw_offset_rad)
        rotated_relative = np.array(
            [
                float(relative_position[0] * cos_y - relative_position[1] * sin_y),
                float(relative_position[0] * sin_y + relative_position[1] * cos_y),
                float(relative_position[2]),
            ],
            dtype=np.float32,
        )
        target_position = self.anchor_location + rotated_relative
        target_yaw_deg = wrap_angle_deg(frame.carla_yaw_deg + yaw_offset_deg)
        return target_position, target_yaw_deg

    def _prepare_next_motion(self) -> bool:
        total_frames = len(self.reference_frames)
        if self.current_frame_idx >= total_frames - 1:
            if self.loop_current_csv:
                print(f"[loop] Repeating current file")
                self.current_frame_idx = 0
                self.pending_motion = None
                return True
            self.current_file_idx += 1
            if self.current_file_idx < len(self.current_csv_files):
                if not self.load_current_file():
                    print(f"[error] Failed to load next file")
                    self.enter_idle()
                    return False
                return True
            print(f"\n[complete] All files in {FOLDER_LIST[self.current_folder_idx]} completed")
            self.start_coasting()
            self.enter_idle(stop_immediately=False)
            return False

        self.current_frame_idx += 1
        current_frame = self.reference_frames[self.current_frame_idx]
        previous_frame = self.reference_frames[self.current_frame_idx - 1]

        if self.actor is not None:
            previous_target_position, previous_target_yaw = self._frame_target_pose(previous_frame)
            current_target_position, current_target_yaw = self._frame_target_pose(current_frame)
            world_delta = current_target_position - previous_target_position
            move_norm = math.hypot(float(world_delta[0]), float(world_delta[1]))
            if move_norm > MIN_HEADING_MOVE:
                heading_yaw = math.degrees(math.atan2(float(world_delta[1]), float(world_delta[0])))
                if current_frame.linear_speed < 0.0:
                    heading_yaw = wrap_angle_deg(heading_yaw + 180.0)
                tf = self.actor.get_transform()
                delta_yaw = wrap_angle_deg(heading_yaw - tf.rotation.yaw)
                current_target_yaw = heading_yaw
            else:
                delta_yaw = wrap_angle_deg(current_target_yaw - previous_target_yaw)
            prev_time = self.reference_frames[self.current_frame_idx - 1].time
            frame_dt = max(self.fixed_delta, float(current_frame.time - prev_time))
            substeps = max(INTERPOLATION_STEPS, int(round(frame_dt * PLAYBACK_FPS)))
            self.pending_motion = PendingMotion(
                world_delta=world_delta.copy(),
                delta_yaw=float(delta_yaw),
                total_steps=substeps,
            )
            self.coast_velocity = world_delta / max(frame_dt, 1e-6)
            self.coast_yaw_rate_deg = delta_yaw / max(frame_dt, 1e-6)

        if self.current_frame_idx == 1 or self.current_frame_idx % 500 == 0:
            target_heading_deg = 0.0
            if self.pending_motion is not None:
                dx = float(self.pending_motion.world_delta[0])
                dy = float(self.pending_motion.world_delta[1])
                if math.hypot(dx, dy) > MIN_HEADING_MOVE:
                    target_heading_deg = math.degrees(math.atan2(dy, dx))
            print(f"[force] frame={self.current_frame_idx:05d}/{total_frames} "
                  f"action=({current_frame.action[0]:+.3f},{current_frame.action[1]:+.3f}) "
                  f"body_delta=({current_frame.body_delta[0]:.3f},{current_frame.body_delta[1]:.3f}) "
                  f"delta_yaw={current_frame.delta_yaw:+.3f} "
                  f"world_delta=({world_delta[0]:+.3f},{world_delta[1]:+.3f}) "
                  f"target_heading={target_heading_deg:+.2f}")
        return True

    def step_once(self, world: Optional["carla.World"] = None) -> bool:
        """执行一步回放（应用相对位移）"""
        if not self.reference_frames:
            return False

        if self.pending_motion is None or self.pending_motion.completed_steps >= self.pending_motion.total_steps:
            self.pending_motion = None
            if not self._prepare_next_motion():
                return False

        if self.pending_motion is not None:
            self._apply_pending_motion_step()

        # 如果需要与世界同步
        if world is not None:
            world.tick()
            if self.actor is not None:
                follow_vehicle_with_spectator(world, self.actor)

        return True

    def next_file(self) -> bool:
        """切换到下一个文件"""
        if self.current_folder_idx == -1:
            print("[error] No folder loaded")
            return False

        if self.current_file_idx + 1 < len(self.current_csv_files):
            self.current_file_idx += 1
            self.loop_current_csv = False
            return self.load_current_file()
        else:
            print("[info] Already at last file")
            return False

    def prev_file(self) -> bool:
        """切换到上一个文件"""
        if self.current_folder_idx == -1:
            print("[error] No folder loaded")
            return False

        if self.current_file_idx > 0:
            self.current_file_idx -= 1
            self.loop_current_csv = False
            return self.load_current_file()
        else:
            print("[info] Already at first file")
            return False

    def toggle_loop(self) -> None:
        """切换循环模式"""
        self.loop_current_csv = not self.loop_current_csv
        status = "ON" if self.loop_current_csv else "OFF"
        print(f"[loop] Loop mode: {status}")

    def start_coasting(self) -> None:
        if self.actor is None:
            return
        self.actor.set_target_velocity(
            carla.Vector3D(
                x=float(self.coast_velocity[0]),
                y=float(self.coast_velocity[1]),
                z=float(self.coast_velocity[2]),
            )
        )
        self.actor.set_target_angular_velocity(carla.Vector3D(z=math.radians(self.coast_yaw_rate_deg)))

    def update(self, world: Optional["carla.World"] = None) -> None:
        if self.actor is None:
            return
        speed = float(np.linalg.norm(self.coast_velocity))
        yaw_rate = abs(self.coast_yaw_rate_deg)
        if speed < 0.02 and yaw_rate < 1.0:
            self.coast_velocity[:] = 0.0
            self.coast_yaw_rate_deg = 0.0
            self.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        else:
            self.coast_velocity *= self.coast_decay
            self.coast_yaw_rate_deg *= self.coast_decay
            self.actor.set_target_velocity(
                carla.Vector3D(
                    x=float(self.coast_velocity[0]),
                    y=float(self.coast_velocity[1]),
                    z=float(self.coast_velocity[2]),
                )
            )
            self.actor.set_target_angular_velocity(carla.Vector3D(z=math.radians(self.coast_yaw_rate_deg)))
        if world is not None:
            world.tick()
            follow_vehicle_with_spectator(world, self.actor)

    def enter_idle(self, stop_immediately: bool = True) -> None:
        """进入空闲状态"""
        self.current_folder_idx = -1
        self.reference_frames = []
        self.current_frame_idx = 0
        self.loop_current_csv = False
        if self.actor is not None and stop_immediately:
            self.coast_velocity[:] = 0.0
            self.coast_yaw_rate_deg = 0.0
            self.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        print("[idle] Entering IDLE state")

    def is_idle(self) -> bool:
        return self.current_folder_idx == -1

    def get_current_info(self) -> str:
        if self.is_idle():
            return "IDLE"
        folder_name = FOLDER_LIST[self.current_folder_idx]
        csv_name = os.path.basename(self.current_csv_files[self.current_file_idx])
        total_frames = len(self.reference_frames)
        loop_marker = " [LOOP]" if self.loop_current_csv else ""
        return f"{folder_name}/{csv_name} frame={self.current_frame_idx}/{total_frames}{loop_marker}"

    def draw_ui(self, screen: Optional["pygame.Surface"], font: Optional["pygame.font.Font"], paused: bool) -> None:
        """绘制 UI"""
        if pygame is None or screen is None or font is None:
            return

        lines = [
            "CSV Standard-Answer Replay (RELATIVE MOTION - NO MODEL)",
            f"Status: {'PAUSED' if paused else 'RUNNING'}",
            f"Loop Mode: {'ON' if self.loop_current_csv else 'OFF'}",
            f"Current: {self.get_current_info()}",
            "",
            "Key mappings:",
            "1-0, -, = : Select folder (can interrupt any playback)",
            "SPACE : Pause/Resume",
            "R     : Reset vehicle to spawn point",
            "Q     : Toggle loop mode (repeat current CSV)",
            "LEFT  : Previous CSV file in folder",
            "RIGHT : Next CSV file in folder",
            "ESC   : Quit",
            "",
            "NOTE: Vehicle moves RELATIVE to current position (no teleport!)",
            (
                f"Spawn point: x={self.spawn_transform.location.x:.3f}, y={self.spawn_transform.location.y:.3f}"
                if self.spawn_transform is not None
                else "Spawn point: unavailable"
            ),
            "",
            "Folder list:",
        ]

        for idx, folder_name in enumerate(FOLDER_LIST[:12]):
            key_name = f"{idx + 1}" if idx < 9 else f"{idx - 8}" if idx == 9 else "-" if idx == 10 else "="
            has_files = idx in self.folder_csv_map and len(self.folder_csv_map[idx]) > 0
            marker = "✓" if has_files else "✗"
            highlight = "→ " if self.current_folder_idx == idx else "  "
            lines.append(f"{highlight}{key_name}: {folder_name} ({marker})")

        screen.fill((20, 20, 20))
        y = 12
        for line in lines:
            img = font.render(line, True, (235, 235, 235))
            screen.blit(img, (12, y))
            y += 26 if "NOTE" in line or "Spawn point" in line else 30
        pygame.display.flip()


def parse_args() -> argparse.Namespace:
    project_root = find_project_root()
    parser = argparse.ArgumentParser(description="CSV standard-answer replay controller - relative motion without model inference")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--fixed-delta", type=float, default=0.05)
    parser.add_argument("--data-root", default="QCarDataSet")
    parser.add_argument("--no-carla", action="store_true", help="Run without CARLA visualization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_pygame_available()
    pygame.init()
    pygame.display.set_caption("CARLA CSV Standard-Answer Replay")
    pygame.display.set_mode((1280, 1000))
    font = pygame.font.SysFont(None, 24)
    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()

    actor = None
    world = None
    original_settings = None
    key_reader = EdgeKeyReader()
    paused = False
    spawn_transform = None

    # 尝试连接 CARLA
    if not args.no_carla:
        try:
            ensure_carla_available()
            client = carla.Client(args.host, args.port)
            client.set_timeout(20.0)
            world = client.get_world()
            print(f"[info] Connected to CARLA at {args.host}:{args.port}")

            original_settings = world.get_settings()
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = args.fixed_delta
            world.apply_settings(settings)

            spawn_transform = choose_spawn_transform(world, args.spawn_index)
            print(f"[info] Spawn point: {spawn_transform.location}")
            actor = spawn_vehicle(world, args.vehicle_filter, spawn_transform)
            print(f"[info] Vehicle spawned: {args.vehicle_filter}")

            actor.set_autopilot(False)
            world.tick()
            world.tick()
            follow_vehicle_with_spectator(world, actor, camera_state=None)
            print("[info] Camera following vehicle")
        except Exception as e:
            print(f"[warning] Failed to connect to CARLA: {e}")
            print("[warning] Running in simulation-only mode (no visualization)")
            actor = None
            world = None

    # 创建控制器
    controller = CSVForcePlayController(
        fixed_delta=args.fixed_delta,
        actor=actor,
        data_root=args.data_root,
        spawn_transform=spawn_transform,
    )

    print("=" * 80)
    print("CSV STANDARD-ANSWER Replay Started (RELATIVE MOTION, NO MODEL)")
    print("Vehicle moves RELATIVE to current position - no teleport!")
    if spawn_transform is not None:
        print(
            f"Vehicle spawns at: x={spawn_transform.location.x:.3f}, y={spawn_transform.location.y:.3f}, z={spawn_transform.location.z:.3f}"
        )
    print("Press 1-0, -, = to select folder (interrupts current playback)")
    print("SPACE: Pause/Resume | R: Reset vehicle to spawn point")
    print("LEFT/RIGHT: Previous/Next CSV file in folder")
    print("ESC: Quit")
    print("=" * 80)

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
                    # R 键：重置车辆到 spawn 点
                    controller.reset_vehicle()
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
                    folder_idx = KEY_MAP[event.key]
                    if folder_idx < len(FOLDER_LIST):
                        print(f"\n[control] Loading folder {FOLDER_LIST[folder_idx]}...")
                        if controller.load_folder(folder_idx):
                            paused = False

        # Windows 原始输入处理
        if key_reader.just_pressed(VK_ESC):
            running = False
        if key_reader.just_pressed(VK_R):
            controller.reset_vehicle()
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

        # 数字键原始输入
        for vk_code, folder_idx in [(0x31, 0), (0x32, 1), (0x33, 2), (0x34, 3), (0x35, 4),
                                    (0x36, 5), (0x37, 6), (0x38, 7), (0x39, 8), (0x30, 9),
                                    (0xBD, 10), (0xBB, 11)]:
            if key_reader.just_pressed(vk_code) and folder_idx < len(FOLDER_LIST):
                print(f"\n[control] Loading folder {FOLDER_LIST[folder_idx]}...")
                if controller.load_folder(folder_idx):
                    paused = False

        # 更新回放
        if not paused:
            if not controller.is_idle():
                controller.step_once(world=world)
            else:
                controller.update(world=world)

        # 绘制 UI
        controller.draw_ui(screen, font, paused)

        # 保持帧率
        clock.tick(60)

    # 清理资源
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
