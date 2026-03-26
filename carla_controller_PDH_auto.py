import argparse
import ctypes
import csv
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
    choose_bundle_for_action,
    extract_state_vector_from_vehicle,
    find_project_root,
    follow_vehicle_with_spectator,
    load_bundle,
    model_position_to_carla_xyz,
    model_quat_to_carla_yaw_deg,
    predict_delta_state,
    predicted_output_to_next_state,
    model_state_to_carla_transform,
    spawn_vehicle,
    choose_spawn_transform,
    wrap_angle_deg,
)

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
    folder_path = Path(data_root) / folder_name
    if not folder_path.exists():
        return []
    return sorted([str(p) for p in folder_path.glob("*.csv")])


def load_reference_frames(path: str) -> List[ReferenceFrame]:
    if not os.path.isabs(path):
        path = os.path.join(find_project_root(), path)
    frames: List[ReferenceFrame] = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"time", "linear_speed", "throttle", "steering", "pos_x", "pos_y", "pos_z", "rot_0", "rot_1", "rot_2", "rot_3"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Reference CSV is missing columns: {sorted(missing)}")

        for raw in reader:
            frames.append(
                ReferenceFrame(
                    time=float(raw["time"]),
                    action=np.array([float(raw["throttle"]), float(raw["steering"])], dtype=np.float32),
                    state=np.array(
                        [float(raw["pos_x"]), float(raw["pos_y"]), float(raw["pos_z"]), float(raw["rot_0"]), float(raw["rot_1"]), float(raw["rot_2"]), float(raw["rot_3"])],
                        dtype=np.float32,
                    ),
                    linear_speed=float(raw["linear_speed"]),
                    source_row={key: float(value) for key, value in raw.items() if value not in {"", None}},
                )
            )
    if not frames:
        raise RuntimeError(f"No frames found in {path}")
    return frames


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
    ) -> None:
        self.forward_bundle = forward_bundle
        self.backward_bundle = backward_bundle
        self.fixed_delta = fixed_delta
        self.actor = actor
        self.data_root = data_root
        self.interpolation_alpha = float(np.clip(interpolation_alpha, 0.0, 1.0))
        self.spawn_transform = spawn_transform
        self.seq_length = forward_bundle.normalizer.seq_length
        self.logs_dir = Path(find_project_root()) / "PDH_auto_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.current_folder_idx = -1
        self.current_file_idx = 0
        self.current_csv_files: List[str] = []
        self.reference_frames: List[ReferenceFrame] = []
        self.controller: Optional["_ReplayInstance"] = None
        self.loop_current_csv = False

        self.folder_csv_map: Dict[int, List[str]] = {}
        for idx, folder_name in enumerate(FOLDER_LIST):
            csv_files = get_csv_files_in_folder(folder_name, data_root)
            if csv_files:
                self.folder_csv_map[idx] = csv_files
                print(f"[init] {folder_name}: {len(csv_files)} CSV files")
            else:
                print(f"[init] {folder_name}: No CSV files found")

    def load_folder(self, folder_idx: int) -> bool:
        if folder_idx not in self.folder_csv_map:
            print(f"[error] Folder index {folder_idx} not available")
            return False
        csv_files = self.folder_csv_map[folder_idx]
        if not csv_files:
            print(f"[error] No CSV files in folder {FOLDER_LIST[folder_idx]}")
            return False

        self.current_folder_idx = folder_idx
        self.current_file_idx = 0
        self.current_csv_files = csv_files
        self.loop_current_csv = False
        return self.load_current_file()

    def load_current_file(self) -> bool:
        if self.current_file_idx >= len(self.current_csv_files):
            return False
        if self.actor is None:
            raise RuntimeError("CARLA actor is required for auto replay")

        csv_path = self.current_csv_files[self.current_file_idx]
        folder_name = FOLDER_LIST[self.current_folder_idx]
        loop_marker = " [LOOP]" if self.loop_current_csv else ""
        print(
            f"\n[load] Model replay: {folder_name}/{os.path.basename(csv_path)} "
            f"({self.current_file_idx + 1}/{len(self.current_csv_files)}){loop_marker}"
        )

        try:
            self.reference_frames = load_reference_frames(csv_path)
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
            )
            print(f"[info] Loaded {len(self.reference_frames)} action frames, ready to replay from current position")
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
        self.actor.set_transform(self.spawn_transform)
        self.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
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
            print(f"\n[complete] All files in {FOLDER_LIST[self.current_folder_idx]} completed")
            self.enter_idle()
            return False

        return True

    def next_file(self) -> bool:
        if self.current_folder_idx == -1:
            print("[error] No folder loaded")
            return False
        if self.current_file_idx + 1 < len(self.current_csv_files):
            self.current_file_idx += 1
            self.loop_current_csv = False
            return self.load_current_file()
        print("[info] Already at last file")
        return False

    def prev_file(self) -> bool:
        if self.current_folder_idx == -1:
            print("[error] No folder loaded")
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
        if self.actor is not None:
            self.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        print("[idle] Entering IDLE state")

    def is_idle(self) -> bool:
        return self.controller is None

    def get_current_info(self) -> str:
        if self.controller is None:
            return "IDLE"
        folder_name = FOLDER_LIST[self.current_folder_idx]
        csv_name = os.path.basename(self.current_csv_files[self.current_file_idx])
        step_info = f"step={self.controller.step_idx}/{self.controller.total_steps()}"
        loop_marker = " [LOOP]" if self.loop_current_csv else ""
        return f"{folder_name}/{csv_name} {step_info}{loop_marker}"

    def draw_ui(self, screen: Optional["pygame.Surface"], font: Optional["pygame.font.Font"], paused: bool) -> None:
        if pygame is None or screen is None or font is None:
            return

        lines = [
            "CSV Model Replay (ACTIONS ONLY, CLOSED LOOP)",
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
            marker = "Y" if has_files else "N"
            highlight = "->" if self.current_folder_idx == idx else "  "
            lines.append(f"{highlight}{key_name}: {folder_name} ({marker})")

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
    ) -> None:
        self.forward_bundle = forward_bundle
        self.backward_bundle = backward_bundle
        self.reference_frames = reference_frames
        self.fixed_delta = fixed_delta
        self.actor = actor
        self.seq_length = seq_length
        self.interpolation_alpha = float(np.clip(interpolation_alpha, 0.0, 1.0))
        self.csv_path = csv_path
        self.step_idx = 0
        self.history: Deque[np.ndarray] = deque(maxlen=seq_length)
        self.model_state = extract_state_vector_from_vehicle(actor)
        self.records: List[Dict[str, float]] = []
        actor_tf = actor.get_transform()
        self.anchor_location = np.array([actor_tf.location.x, actor_tf.location.y, actor_tf.location.z], dtype=np.float32)
        self.anchor_yaw_deg = float(actor_tf.rotation.yaw)
        self.reference_origin = model_position_to_carla_xyz(reference_frames[0].state[:3]).astype(np.float32)
        self.reference_origin_yaw_deg = float(model_quat_to_carla_yaw_deg(reference_frames[0].state[3:7]))
        bootstrap_model_history(
            self.history,
            self.model_state,
            self.reference_frames[0].action,
            self.seq_length,
        )
        actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))

    def total_steps(self) -> int:
        return len(self.reference_frames) - 1

    def _reference_target_pose(self, frame: ReferenceFrame) -> tuple[np.ndarray, float]:
        ref_position = model_position_to_carla_xyz(frame.state[:3]).astype(np.float32)
        relative_position = ref_position - self.reference_origin
        yaw_offset_deg = self.anchor_yaw_deg - self.reference_origin_yaw_deg
        yaw_offset_rad = np.radians(yaw_offset_deg)
        cos_y = float(np.cos(yaw_offset_rad))
        sin_y = float(np.sin(yaw_offset_rad))
        rotated_relative = np.array(
            [
                float(relative_position[0] * cos_y - relative_position[1] * sin_y),
                float(relative_position[0] * sin_y + relative_position[1] * cos_y),
                float(relative_position[2]),
            ],
            dtype=np.float32,
        )
        target_position = self.anchor_location + rotated_relative
        ref_yaw = float(model_quat_to_carla_yaw_deg(frame.state[3:7]))
        target_yaw = wrap_angle_deg(ref_yaw + yaw_offset_deg)
        return target_position, target_yaw

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

    def step_once(self, world: Optional["carla.World"] = None) -> bool:
        if self.step_idx >= self.total_steps():
            return False

        current_ref = self.reference_frames[self.step_idx]
        action = current_ref.action
        bundle = choose_bundle_for_action(action, self.forward_bundle, self.backward_bundle)
        history_np = np.stack(list(self.history), axis=0).astype(np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predicted_output = predict_delta_state(history_np, bundle, device)
        self.model_state = predicted_output_to_next_state(self.model_state, predicted_output)

        self._apply_model_state_to_actor(self.model_state)
        ref_target_pos, ref_target_yaw = self._reference_target_pose(self.reference_frames[self.step_idx + 1])
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
        self.records.append(
            {
                "step": int(self.step_idx),
                "time": float(current_ref.time),
                "action_throttle": float(action[0]),
                "action_steer": float(action[1]),
                "ref_x": float(ref_target_pos[0]),
                "ref_y": float(ref_target_pos[1]),
                "ref_z": float(ref_target_pos[2]),
                "ref_yaw_deg": float(ref_target_yaw),
                "pred_x": float(actor_tf.location.x),
                "pred_y": float(actor_tf.location.y),
                "pred_z": float(actor_tf.location.z),
                "pred_yaw_deg": float(actor_tf.rotation.yaw),
                "error_x": float(pos_error[0]),
                "error_y": float(pos_error[1]),
                "error_z": float(pos_error[2]),
                "error_pos_l2": float(np.linalg.norm(pos_error)),
                "error_yaw_deg": float(yaw_error),
            }
        )

        next_action = self.reference_frames[self.step_idx + 1].action if self.step_idx + 1 < len(self.reference_frames) else action
        self.history.append(np.concatenate([self.model_state, next_action], axis=0).astype(np.float32))
        self.step_idx += 1

        if self.step_idx == 1 or self.step_idx % 50 == 0:
            print(f"[step={self.step_idx:05d}/{self.total_steps()}] model={bundle.name} action=({action[0]:+.3f},{action[1]:+.3f})")

        if world is not None:
            world.tick()
            follow_vehicle_with_spectator(world, self.actor)
        return True

    def _apply_model_state_to_actor(self, next_state: np.ndarray) -> None:
        tf = self.actor.get_transform()
        target_tf = model_state_to_carla_transform(next_state)
        alpha = self.interpolation_alpha
        dx_world = float((target_tf.location.x - tf.location.x) * alpha)
        dy_world = float((target_tf.location.y - tf.location.y) * alpha)
        dz_world = float((target_tf.location.z - tf.location.z) * alpha)
        delta_yaw = ((target_tf.rotation.yaw - tf.rotation.yaw + 180.0) % 360.0 - 180.0) * alpha
        self.actor.set_transform(
            carla.Transform(
                carla.Location(x=tf.location.x + dx_world, y=tf.location.y + dy_world, z=tf.location.z + dz_world),
                carla.Rotation(
                    roll=tf.rotation.roll + (target_tf.rotation.roll - tf.rotation.roll) * alpha,
                    pitch=tf.rotation.pitch + (target_tf.rotation.pitch - tf.rotation.pitch) * alpha,
                    yaw=tf.rotation.yaw + delta_yaw,
                ),
            )
        )
        dt = max(self.fixed_delta, 1e-6)
        self.actor.set_target_velocity(carla.Vector3D(x=dx_world / dt, y=dy_world / dt, z=dz_world / dt))


def parse_args() -> argparse.Namespace:
    project_root = find_project_root()
    parser = argparse.ArgumentParser(description="CSV model replay controller using CSV actions only")
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
    parser.add_argument("--no-carla", action="store_true", help="Run pure model replay without CARLA visualization")
    parser.add_argument("--interpolation-alpha", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forward_bundle = load_bundle("forward", args.forward_model_path, args.forward_norm_path, device)
    backward_bundle = load_bundle("backward", args.backward_model_path, args.backward_norm_path, device)

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
        )

        print("=" * 80)
        print("CSV MODEL Replay Started (CSV ACTIONS ONLY, MODEL CLOSED LOOP)")
        print(f"Interpolation alpha: {args.interpolation_alpha:.2f}")
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

            for vk_code, folder_idx in [(0x31, 0), (0x32, 1), (0x33, 2), (0x34, 3), (0x35, 4),
                                        (0x36, 5), (0x37, 6), (0x38, 7), (0x39, 8), (0x30, 9),
                                        (0xBD, 10), (0xBB, 11)]:
                if key_reader.just_pressed(vk_code) and folder_idx < len(FOLDER_LIST):
                    print(f"\n[control] Loading folder {FOLDER_LIST[folder_idx]}...")
                    if controller.load_folder(folder_idx):
                        paused = False

            if not paused and not controller.is_idle():
                controller.step_once(world=world)

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
