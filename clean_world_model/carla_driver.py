import argparse
import ctypes
import math
import os
import sys
from collections import deque
from typing import Deque, Optional

import numpy as np
import pygame
import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clean_world_model.common import (
    FEATURE_NAMES,
    STATE_NAMES,
    CleanWorldModel,
    Normalizer,
    deg2rad,
    project_root,
    shortest_angle_delta_deg,
    wrap_angle_deg,
)

try:
    import carla
except ImportError as e:
    raise SystemExit("Failed to import CARLA Python API.") from e


VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44
VK_R = 0x52
VK_ESC = 0x1B


def is_key_down(vk_code: int) -> bool:
    return (ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000) != 0


class EdgeKeyReader:
    def __init__(self) -> None:
        self.prev = {key: False for key in [VK_W, VK_A, VK_S, VK_D, VK_R, VK_ESC]}

    def just_pressed(self, vk_code: int) -> bool:
        now = is_key_down(vk_code)
        was = self.prev[vk_code]
        self.prev[vk_code] = now
        return now and not was


def speed_mps(v: "carla.Vector3D") -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def follow_vehicle_with_spectator(world: "carla.World", vehicle: "carla.Vehicle") -> None:
    spectator = world.get_spectator()
    tf = vehicle.get_transform()
    yaw_rad = deg2rad(tf.rotation.yaw)
    spectator.set_transform(
        carla.Transform(
            carla.Location(
                x=tf.location.x - 8.0 * math.cos(yaw_rad),
                y=tf.location.y - 8.0 * math.sin(yaw_rad),
                z=tf.location.z + 3.0,
            ),
            carla.Rotation(pitch=-15.0, yaw=tf.rotation.yaw, roll=0.0),
        )
    )


def extract_state(vehicle: "carla.Vehicle") -> np.ndarray:
    tf = vehicle.get_transform()
    yaw_rad = deg2rad(tf.rotation.yaw)
    return np.array(
        [
            tf.location.x,
            tf.location.y,
            tf.location.z,
            math.sin(yaw_rad),
            math.cos(yaw_rad),
            tf.rotation.roll,
            tf.rotation.pitch,
        ],
        dtype=np.float32,
    )


def yaw_deg_from_state(state: np.ndarray) -> float:
    return wrap_angle_deg(math.degrees(math.atan2(float(state[3]), float(state[4]))))


def bootstrap_history(history: Deque[np.ndarray], state: np.ndarray, action: np.ndarray, seq_length: int) -> None:
    history.clear()
    feat = np.concatenate([state, action], axis=0).astype(np.float32)
    for _ in range(seq_length):
        history.append(feat.copy())


def predict_next_state(
    history_tx9: np.ndarray,
    model: CleanWorldModel,
    normalizer: Normalizer,
    device: torch.device,
) -> np.ndarray:
    x = torch.from_numpy(history_tx9).float().unsqueeze(0).to(device)
    x = normalizer.norm_x(x)
    with torch.no_grad():
        pred = normalizer.denorm_y(model(x)).cpu().numpy()[0]

    yaw_norm = math.hypot(float(pred[3]), float(pred[4]))
    if yaw_norm > 1e-8:
        pred[3] /= yaw_norm
        pred[4] /= yaw_norm
    else:
        pred[3] = 0.0
        pred[4] = 1.0
    return pred.astype(np.float32)


def apply_predicted_state(
    vehicle: "carla.Vehicle",
    current_state: np.ndarray,
    predicted_state: np.ndarray,
    fixed_delta: float,
) -> np.ndarray:
    tf = vehicle.get_transform()
    delta_xy = predicted_state[:2] - current_state[:2]
    step_dist = float(np.linalg.norm(delta_xy))
    if step_dist > 0.20 and step_dist > 1e-8:
        delta_xy = delta_xy * (0.20 / step_dist)

    predicted_yaw = yaw_deg_from_state(predicted_state)
    delta_yaw = shortest_angle_delta_deg(predicted_yaw, tf.rotation.yaw)
    delta_yaw = max(-5.0, min(5.0, delta_yaw))

    new_tf = carla.Transform(
        carla.Location(x=tf.location.x + float(delta_xy[0]), y=tf.location.y + float(delta_xy[1]), z=tf.location.z),
        carla.Rotation(roll=tf.rotation.roll, pitch=tf.rotation.pitch, yaw=tf.rotation.yaw + delta_yaw),
    )
    vehicle.set_transform(new_tf)

    dt = max(fixed_delta, 1e-6)
    vehicle.set_target_velocity(carla.Vector3D(x=float(delta_xy[0]) / dt, y=float(delta_xy[1]) / dt, z=0.0))

    next_state = predicted_state.copy()
    next_state[0] = new_tf.location.x
    next_state[1] = new_tf.location.y
    next_state[2] = new_tf.location.z
    next_state[5] = new_tf.rotation.roll
    next_state[6] = new_tf.rotation.pitch
    yaw_rad = deg2rad(new_tf.rotation.yaw)
    next_state[3] = math.sin(yaw_rad)
    next_state[4] = math.cos(yaw_rad)
    return next_state.astype(np.float32)


def choose_spawn_transform(world: "carla.World", spawn_index: int) -> "carla.Transform":
    points = world.get_map().get_spawn_points()
    if not points:
        raise RuntimeError("No spawn points in current map.")
    return points[max(0, min(spawn_index, len(points) - 1))]


def spawn_vehicle(world: "carla.World", vehicle_filter: str, spawn_transform: "carla.Transform") -> "carla.Vehicle":
    blueprints = world.get_blueprint_library().filter(vehicle_filter)
    if not blueprints:
        raise RuntimeError(f"Vehicle blueprint not found: {vehicle_filter}")
    vehicle = world.try_spawn_actor(blueprints[0], spawn_transform)
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle.")
    return vehicle


def main() -> None:
    root = project_root()

    parser = argparse.ArgumentParser(description="CARLA driver for the clean QCar world model.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--model-dir", default=os.path.join(root, "clean_world_model_artifacts", "models_saved"))
    parser.add_argument("--direction", default="forward", choices=["forward", "backward"])
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--fixed-delta", type=float, default=0.04)
    args = parser.parse_args()

    args.model_dir = os.path.abspath(args.model_dir)

    model_path = os.path.join(args.model_dir, f"{args.direction}_clean_world_model.pth")
    norm_path = os.path.join(args.model_dir, f"{args.direction}_clean_normalization.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = Normalizer.from_file(norm_path)
    model = CleanWorldModel(input_dim=len(FEATURE_NAMES), output_dim=len(STATE_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    pygame.init()
    pygame.display.set_caption("Clean QCar World Model Driver")
    pygame.display.set_mode((900, 230))
    font = pygame.font.SysFont(None, 24)
    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    original_settings = world.get_settings()
    vehicle: Optional["carla.Vehicle"] = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta
        world.apply_settings(settings)

        vehicle = spawn_vehicle(world, args.vehicle_filter, choose_spawn_transform(world, args.spawn_index))
        vehicle.set_autopilot(False)
        world.tick()
        world.tick()

        reset_tf = vehicle.get_transform()
        state = extract_state(vehicle)
        action = np.array([0.0, 0.0], dtype=np.float32)
        history: Deque[np.ndarray] = deque(maxlen=normalizer.seq_length)
        bootstrap_history(history, state, action, normalizer.seq_length)
        auto_mode = False
        frame_count = 0
        key_reader = EdgeKeyReader()

        base_throttle = 0.08 if args.direction == "forward" else -0.08
        steer_mag = 0.20

        while True:
            pygame.event.pump()
            if key_reader.just_pressed(VK_ESC):
                break

            if key_reader.just_pressed(VK_W):
                auto_mode = True
                action = np.array([base_throttle, 0.0], dtype=np.float32)
                state = extract_state(vehicle)
                bootstrap_history(history, state, action, normalizer.seq_length)
            elif key_reader.just_pressed(VK_A):
                auto_mode = True
                action = np.array([base_throttle, -steer_mag], dtype=np.float32)
                state = extract_state(vehicle)
                bootstrap_history(history, state, action, normalizer.seq_length)
            elif key_reader.just_pressed(VK_D):
                auto_mode = True
                action = np.array([base_throttle, steer_mag], dtype=np.float32)
                state = extract_state(vehicle)
                bootstrap_history(history, state, action, normalizer.seq_length)
            elif key_reader.just_pressed(VK_S):
                auto_mode = False
                action = np.array([0.0, 0.0], dtype=np.float32)
                vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            elif key_reader.just_pressed(VK_R):
                auto_mode = False
                action = np.array([0.0, 0.0], dtype=np.float32)
                vehicle.set_transform(reset_tf)
                vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                world.tick()
                state = extract_state(vehicle)
                bootstrap_history(history, state, action, normalizer.seq_length)

            if auto_mode and len(history) == normalizer.seq_length:
                pred_state = predict_next_state(np.stack(list(history), axis=0), model, normalizer, device)
                state = apply_predicted_state(vehicle, state, pred_state, args.fixed_delta)
                history.append(np.concatenate([state, action], axis=0).astype(np.float32))
            else:
                state = extract_state(vehicle)
                history.append(np.concatenate([state, action], axis=0).astype(np.float32))

            world.tick()
            frame_count += 1
            follow_vehicle_with_spectator(world, vehicle)

            tf = vehicle.get_transform()
            speed = speed_mps(vehicle.get_velocity())
            yaw_deg = yaw_deg_from_state(state)
            lines = [
                f"mode={'AUTO' if auto_mode else 'IDLE'} direction={args.direction}",
                f"action throttle={action[0]:.3f} steer={action[1]:.3f}",
                f"vehicle x={tf.location.x:.3f} y={tf.location.y:.3f} z={tf.location.z:.3f}",
                f"yaw={tf.rotation.yaw:.3f} speed={speed:.3f} m/s",
                f"model x={state[0]:.3f} y={state[1]:.3f} yaw={yaw_deg:.3f}",
            ]

            screen.fill((18, 18, 18))
            y = 12
            for line in lines:
                img = font.render(line, True, (235, 235, 235))
                screen.blit(img, (12, y))
                y += 32
            pygame.display.flip()
            clock.tick(60)

            if frame_count % 40 == 0:
                print(
                    f"[frame={frame_count}] mode={'AUTO' if auto_mode else 'IDLE'} "
                    f"action=({action[0]:.3f},{action[1]:.3f}) "
                    f"carla=({tf.location.x:.3f},{tf.location.y:.3f},{tf.rotation.yaw:.3f}) "
                    f"model=({state[0]:.3f},{state[1]:.3f},{yaw_deg:.3f})"
                )
    finally:
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


if __name__ == "__main__":
    main()
