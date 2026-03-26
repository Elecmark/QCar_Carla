import argparse
import ctypes
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
import pygame
import torch
import torch.nn as nn

try:
    import carla
except ImportError as e:
    raise SystemExit(
        "未能导入 carla。请先确认你已经在当前 Python 3.12 虚拟环境中正确安装/链接了 CARLA 0.9.16 的 .whl。"
    ) from e


# ============================================================
# 1) 与训练脚本完全一致的模型结构
#    假设：模型输出的是【下一时刻状态的变化量 Δstate】
#    Δstate = next_state - curr_state
#    其中 state = [pos_x,pos_y,pos_z,rot_0,rot_1,rot_2,rot_3]
# ============================================================
class QCarWorldModel(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=256, num_layers=3, output_dim=7):
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


# ============================================================
# 2) normalization 读取器
# ============================================================
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
            raise ValueError(f"normalization 文件缺少字段: {missing}")

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


# ============================================================
# 3) 自动找项目根目录（要求根目录下存在 models_saved）
# ============================================================
def find_project_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        script_dir,
        os.path.dirname(script_dir),
    ]
    for root in candidates:
        if os.path.exists(os.path.join(root, "models_saved")):
            return root
    raise FileNotFoundError(f"在以下目录都找不到 models_saved: {candidates}")


# ============================================================
# 4) 工具函数
# ============================================================
def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    return rad * 180.0 / math.pi


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


def quat_wxyz_to_euler_deg(qw: float, qx: float, qy: float, qz: float):
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


# ============================================================
# 5) 摄像机跟车
# ============================================================
def follow_vehicle_with_spectator(world: "carla.World", vehicle: "carla.Vehicle") -> None:
    spectator = world.get_spectator()
    tf = vehicle.get_transform()
    yaw_rad = deg2rad(tf.rotation.yaw)

    back_dist = 8.0
    up_dist = 3.0

    cam_x = tf.location.x - back_dist * math.cos(yaw_rad)
    cam_y = tf.location.y - back_dist * math.sin(yaw_rad)
    cam_z = tf.location.z + up_dist

    spectator.set_transform(
        carla.Transform(
            carla.Location(x=cam_x, y=cam_y, z=cam_z),
            carla.Rotation(pitch=-15.0, yaw=tf.rotation.yaw, roll=0.0),
        )
    )


# ============================================================
# 6) “模型状态”定义：与训练严格对齐
#    state = [pos_x, pos_y, pos_z, rot_0, rot_1, rot_2, rot_3]
#    其中 rot_0~rot_3 = quaternion(w, x, y, z)
# ============================================================
def extract_state_vector_from_vehicle(vehicle: "carla.Vehicle") -> np.ndarray:
    tf = vehicle.get_transform()

    pos_x = tf.location.x
    pos_y = tf.location.y
    pos_z = tf.location.z

    quat = euler_deg_to_quat_wxyz(
        roll_deg=tf.rotation.roll,
        pitch_deg=tf.rotation.pitch,
        yaw_deg=tf.rotation.yaw,
    )
    rot_0, rot_1, rot_2, rot_3 = quat.tolist()

    return np.array(
        [pos_x, pos_y, pos_z, rot_0, rot_1, rot_2, rot_3],
        dtype=np.float32,
    )


def apply_model_state_to_vehicle_unlock(
    vehicle: "carla.Vehicle",
    prev_model_state: np.ndarray,
    next_model_state: np.ndarray,
    fixed_delta: float,
) -> None:
    """
    v4.1_unlock 策略：
    - 完全信任模型空间的下一帧绝对状态 next_model_state，
      直接把 CARLA 车辆放到该状态对应的 (x,y,z,roll,pitch,yaw)。
    - z / roll / pitch / yaw 都不再锁定，全部来自模型（解锁所有轴）。
    """
    # 1) 位置直接使用模型的绝对位置
    pos_x = float(next_model_state[0])
    pos_y = float(next_model_state[1])
    pos_z = float(next_model_state[2])

    # 2) 姿态：从下一帧四元数还原 roll/pitch/yaw
    qw, qx, qy, qz = [float(v) for v in next_model_state[3:7]]
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm > 1e-8:
        qw /= norm
        qx /= norm
        qy /= norm
        qz /= norm
        roll_deg, pitch_deg, yaw_deg = quat_wxyz_to_euler_deg(qw, qx, qy, qz)
    else:
        # 四元数异常时，保持当前姿态
        tf_curr = vehicle.get_transform()
        roll_deg = tf_curr.rotation.roll
        pitch_deg = tf_curr.rotation.pitch
        yaw_deg = tf_curr.rotation.yaw

    new_tf = carla.Transform(
        carla.Location(x=pos_x, y=pos_y, z=pos_z),
        carla.Rotation(roll=roll_deg, pitch=pitch_deg, yaw=yaw_deg),
    )
    vehicle.set_transform(new_tf)

    # 3) 用模型空间前后位置差估算速度
    dx = float(next_model_state[0] - prev_model_state[0])
    dy = float(next_model_state[1] - prev_model_state[1])
    dz = float(next_model_state[2] - prev_model_state[2])

    vx = dx / max(fixed_delta, 1e-6)
    vy = dy / max(fixed_delta, 1e-6)
    vz = dz / max(fixed_delta, 1e-6)

    vehicle.set_target_velocity(carla.Vector3D(x=vx, y=vy, z=vz))


# ============================================================
# 7) Windows 键盘边沿检测
# ============================================================
VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44
VK_R = 0x52
VK_ESC = 0x1B


def is_key_down(vk_code: int) -> bool:
    return (ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000) != 0


class EdgeKeyReader:
    def __init__(self):
        self.prev = {
            VK_W: False,
            VK_A: False,
            VK_S: False,
            VK_D: False,
            VK_R: False,
            VK_ESC: False,
        }

    def just_pressed(self, vk_code: int) -> bool:
        now = is_key_down(vk_code)
        was = self.prev[vk_code]
        self.prev[vk_code] = now
        return now and (not was)


# ============================================================
# 8) 模型推理（严格按训练时的特征顺序）
#    history[t] = [pos_x,pos_y,pos_z,rot_0,rot_1,rot_2,rot_3,throttle,steer]
#    输出 Δstate = [Δpos_x,Δpos_y,Δpos_z,Δrot_0,Δrot_1,Δrot_2,Δrot_3]
# ============================================================
def predict_delta_state(
    history_tx9: np.ndarray,
    model: QCarWorldModel,
    normalizer: Normalizer,
    device: torch.device,
) -> np.ndarray:
    x = torch.from_numpy(history_tx9).float().unsqueeze(0).to(device)  # [1, T, 9]
    x = normalizer.norm_x(x)

    with torch.no_grad():
        pred_norm = model(x)  # [1, 7]
        delta_state = normalizer.denorm_y(pred_norm)

    return delta_state.squeeze(0).cpu().numpy().astype(np.float32)


# ============================================================
# 9) 车辆生成
# ============================================================
def choose_spawn_transform(world: "carla.World", spawn_index: int) -> "carla.Transform":
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("当前地图没有可用 spawn points。")
    spawn_index = max(0, min(spawn_index, len(spawn_points) - 1))
    return spawn_points[spawn_index]


def spawn_vehicle(world: "carla.World", vehicle_filter: str, spawn_transform: "carla.Transform") -> "carla.Vehicle":
    blueprints = world.get_blueprint_library().filter(vehicle_filter)
    if not blueprints:
        raise RuntimeError(f"没有找到车辆蓝图: {vehicle_filter}")

    bp = blueprints[0]
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "hero")
    if bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if colors:
            bp.set_attribute("color", colors[0])

    vehicle = world.try_spawn_actor(bp, spawn_transform)
    if vehicle is None:
        raise RuntimeError("生成车辆失败，当前 spawn 点可能被占用。请换一个 spawn_index。")
    return vehicle


# ============================================================
# 10) “模型历史窗口”初始化
# ============================================================
def bootstrap_model_history(
    history: Deque[np.ndarray],
    init_model_state: np.ndarray,
    latched_action: np.ndarray,
    seq_length: int,
) -> None:
    history.clear()
    feat_9 = np.concatenate([init_model_state, latched_action], axis=0).astype(np.float32)
    for _ in range(seq_length):
        history.append(feat_9.copy())


# ============================================================
# 11) 主流程
# ============================================================
def main() -> None:
    project_root = find_project_root()
    default_model_path = os.path.join(project_root, "models_saved", "forward_world_model.pth")
    default_norm_path = os.path.join(project_root, "models_saved", "forward_normalization.pt")

    parser = argparse.ArgumentParser(
        description="CARLA + QCar World Model Auto Driver (v4.1_unlock, full-state mapping)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--model-path", default=default_model_path)
    parser.add_argument("--norm-path", default=default_norm_path)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--fixed-delta", type=float, default=0.04)
    args = parser.parse_args()

    print("project_root =", project_root)
    print("model path   =", args.model_path)
    print("norm path    =", args.norm_path)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"找不到模型文件: {args.model_path}")
    if not os.path.exists(args.norm_path):
        raise FileNotFoundError(f"找不到归一化文件: {args.norm_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = Normalizer.from_file(args.norm_path)

    model = QCarWorldModel(input_dim=9, hidden_dim=256, num_layers=3, output_dim=7)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    pygame.init()
    pygame.display.set_caption("QCar World Model Auto Driver v4.1_unlock")
    pygame.display.set_mode((880, 260))
    font = pygame.font.SysFont(None, 24)
    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    original_settings = world.get_settings()
    vehicle: Optional["carla.Vehicle"] = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta
        world.apply_settings(settings)

        spawn_transform = choose_spawn_transform(world, args.spawn_index)
        vehicle = spawn_vehicle(world, args.vehicle_filter, spawn_transform)
        vehicle.set_autopilot(False)

        world.tick()
        world.tick()

        reset_transform = vehicle.get_transform()

        follow_vehicle_with_spectator(world, vehicle)

        history: Deque[np.ndarray] = deque(maxlen=normalizer.seq_length)
        latched_action = np.array([0.0, 0.0], dtype=np.float32)
        auto_mode = False
        frame_count = 0
        key_reader = EdgeKeyReader()

        model_state = extract_state_vector_from_vehicle(vehicle)
        bootstrap_model_history(
            history=history,
            init_model_state=model_state,
            latched_action=latched_action,
            seq_length=normalizer.seq_length,
        )

        print("=" * 80)
        print("脚本 v4.1_unlock 已启动（完全按模型状态映射到 CARLA）")
        print("模型输入特征：")
        print("[pos_x, pos_y, pos_z, rot_0, rot_1, rot_2, rot_3, throttle, steer]")
        print("单击 W -> 自动前进")
        print("单击 A -> 自动左前")
        print("单击 D -> 自动右前")
        print("单击 S -> 停止")
        print("单击 R -> 重置")
        print("ESC -> 退出")
        print("=" * 80)

        while True:
            pygame.event.pump()

            if key_reader.just_pressed(VK_ESC):
                break

            if key_reader.just_pressed(VK_W):
                auto_mode = True
                latched_action = np.array([0.25, 0.00], dtype=np.float32)
                model_state = extract_state_vector_from_vehicle(vehicle)
                bootstrap_model_history(
                    history, model_state, latched_action, normalizer.seq_length
                )
                print("[输入] 自动前进模式 action=[0.25, 0.00]")

            elif key_reader.just_pressed(VK_A):
                auto_mode = True
                latched_action = np.array([0.25, -0.30], dtype=np.float32)
                model_state = extract_state_vector_from_vehicle(vehicle)
                bootstrap_model_history(
                    history, model_state, latched_action, normalizer.seq_length
                )
                print("[输入] 自动左前模式 action=[0.25, -0.30]")

            elif key_reader.just_pressed(VK_D):
                auto_mode = True
                latched_action = np.array([0.25, 0.30], dtype=np.float32)
                model_state = extract_state_vector_from_vehicle(vehicle)
                bootstrap_model_history(
                    history, model_state, latched_action, normalizer.seq_length
                )
                print("[输入] 自动右前模式 action=[0.25, 0.30]")

            elif key_reader.just_pressed(VK_S):
                auto_mode = False
                latched_action = np.array([0.0, 0.0], dtype=np.float32)
                vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                print("[输入] 停止自动模式")

            elif key_reader.just_pressed(VK_R):
                auto_mode = False
                latched_action = np.array([0.0, 0.0], dtype=np.float32)
                vehicle.set_transform(reset_transform)
                vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                world.tick()
                model_state = extract_state_vector_from_vehicle(vehicle)
                bootstrap_model_history(
                    history, model_state, latched_action, normalizer.seq_length
                )
                follow_vehicle_with_spectator(world, vehicle)
                print("[输入] 已重置车辆与模型历史")

            if auto_mode and len(history) == normalizer.seq_length:
                hist_np = np.stack(list(history), axis=0).astype(np.float32)

                delta_state = predict_delta_state(
                    history_tx9=hist_np,
                    model=model,
                    normalizer=normalizer,
                    device=device,
                )
                next_model_state = model_state + delta_state

                # 四元数再归一化，减少数值漂移
                q = next_model_state[3:7].copy()
                q_norm = np.linalg.norm(q)
                if q_norm > 1e-8:
                    next_model_state[3:7] = q / q_norm

                apply_model_state_to_vehicle_unlock(
                    vehicle=vehicle,
                    prev_model_state=model_state,
                    next_model_state=next_model_state,
                    fixed_delta=args.fixed_delta,
                )

                model_state = next_model_state
                next_feature = np.concatenate(
                    [model_state, latched_action], axis=0
                ).astype(np.float32)
                history.append(next_feature)
            else:
                model_state = extract_state_vector_from_vehicle(vehicle)
                feat_9 = np.concatenate([model_state, latched_action], axis=0).astype(
                    np.float32
                )
                history.append(feat_9)

            world.tick()
            frame_count += 1
            follow_vehicle_with_spectator(world, vehicle)

            tf = vehicle.get_transform()
            vel = vehicle.get_velocity()
            spd = speed_mps(vel)

            mode_text = "AUTO" if auto_mode else "IDLE"
            lines = [
                f"Mode: {mode_text}",
                f"Latched action: throttle={latched_action[0]:.2f}, steer={latched_action[1]:.2f}",
                f"Vehicle pos (CARLA): x={tf.location.x:.3f}, y={tf.location.y:.3f}, z={tf.location.z:.3f}",
                f"Vehicle yaw (CARLA): {tf.rotation.yaw:.3f} deg | speed={spd:.3f} m/s",
                f"Model state pos: x={model_state[0]:.3f}, y={model_state[1]:.3f}, z={model_state[2]:.3f}",
                f"Model quat: ({model_state[3]:.3f},{model_state[4]:.3f},{model_state[5]:.3f},{model_state[6]:.3f})",
                "Tap W=forward, A=left-forward, D=right-forward, S=stop, R=reset, ESC=quit",
            ]

            screen.fill((20, 20, 20))
            y = 12
            for line in lines:
                img = font.render(line, True, (235, 235, 235))
                screen.blit(img, (12, y))
                y += 30
            pygame.display.flip()

            clock.tick(60)

            if frame_count % 20 == 0:
                print(
                    f"[frame={frame_count}] mode={mode_text}, "
                    f"action=({latched_action[0]:.2f},{latched_action[1]:.2f}), "
                    f"CARLA pos=({tf.location.x:.3f},{tf.location.y:.3f},{tf.location.z:.3f}), "
                    f"CARLA yaw={tf.rotation.yaw:.3f}, speed={spd:.3f}, "
                    f"model_pos=({model_state[0]:.3f},{model_state[1]:.3f},{model_state[2]:.3f})"
                )

    finally:
        print("开始清理 CARLA 资源...")
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
        print("已退出。")


if __name__ == "__main__":
    main()

