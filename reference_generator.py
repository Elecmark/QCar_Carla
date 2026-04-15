import argparse
import csv
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from carla_controller_PDH import (
    bootstrap_model_history,
    euler_deg_to_quat_xyzw,
    find_project_root,
    normalize_quaternion_xyzw,
    predict_delta_state,
    predicted_output_to_next_state,
    xyzw_to_raw_quaternion,
)


STATE_COLUMNS = ["pos_x", "pos_y", "pos_z", "rot_0", "rot_1", "rot_2", "rot_3"]
ACTION_COLUMNS = ["throttle", "steering"]
CSV_REQUIRED_COLUMNS = ["time", "pos_x", "pos_y", "pos_z", "rot_0", "rot_1", "rot_2", "rot_3"]


@dataclass
class ReferenceTrajectory:
    name: str
    states: np.ndarray
    actions: np.ndarray
    times: np.ndarray
    metadata: Dict[str, object]


def yaw_deg_to_raw_quaternion(yaw_deg: float) -> np.ndarray:
    quat_xyzw = normalize_quaternion_xyzw(euler_deg_to_quat_xyzw(0.0, 0.0, yaw_deg))
    return xyzw_to_raw_quaternion(quat_xyzw)


def quaternion_yaw_deg(quat_raw: np.ndarray) -> float:
    x, y, z, w = quat_raw[0], quat_raw[2], quat_raw[1], quat_raw[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def state_from_xy_yaw(x: float, y: float, yaw_deg: float, z: float = 0.0) -> np.ndarray:
    quat_raw = yaw_deg_to_raw_quaternion(yaw_deg)
    return np.array([x, y, z, quat_raw[0], quat_raw[1], quat_raw[2], quat_raw[3]], dtype=np.float32)


def finite_difference_actions(states: np.ndarray, throttle_scale: float = 4.0, steer_scale: float = 0.03) -> np.ndarray:
    actions = np.zeros((states.shape[0], 2), dtype=np.float32)
    diffs = np.diff(states[:, :2], axis=0, prepend=states[:1, :2])
    step_norm = np.linalg.norm(diffs, axis=1)
    actions[:, 0] = np.clip(step_norm * throttle_scale, -1.0, 1.0)

    yaws = np.array([quaternion_yaw_deg(state[3:7]) for state in states], dtype=np.float32)
    yaw_delta = np.diff(yaws, prepend=yaws[:1])
    wrapped = ((yaw_delta + 180.0) % 360.0) - 180.0
    actions[:, 1] = np.clip(wrapped * steer_scale, -1.0, 1.0)
    return actions


def generate_reference_trajectory(traj_type: str, length: int, params: Optional[Dict[str, float]] = None) -> ReferenceTrajectory:
    params = dict(params or {})
    dt = float(params.get("dt", 0.1))
    z = float(params.get("z", 0.0))
    t = np.arange(length, dtype=np.float32) * dt

    if traj_type == "straight":
        speed = float(params.get("speed", 0.2))
        x = speed * t
        y = np.zeros_like(t)
    elif traj_type == "circle":
        radius = float(params.get("radius", 5.0))
        omega = float(params.get("omega", 0.12))
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
    elif traj_type == "figure8":
        radius = float(params.get("radius", 4.0))
        omega = float(params.get("omega", 0.12))
        x = radius * np.sin(omega * t)
        y = radius * np.sin(omega * t) * np.cos(omega * t)
    elif traj_type == "sine":
        speed = float(params.get("speed", 0.25))
        amplitude = float(params.get("amplitude", 2.0))
        frequency = float(params.get("frequency", 0.25))
        x = speed * t
        y = amplitude * np.sin(2.0 * math.pi * frequency * t)
    elif traj_type == "s_curve":
        speed = float(params.get("speed", 0.25))
        amplitude = float(params.get("amplitude", 1.5))
        x = speed * t
        y = amplitude * np.tanh(np.sin(2.0 * math.pi * t / max(length * dt, dt)))
    else:
        raise ValueError(f"Unsupported trajectory type: {traj_type}")

    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    yaw_deg = np.degrees(np.arctan2(dy, dx))

    states = np.stack([state_from_xy_yaw(float(px), float(py), float(pyaw), z=z) for px, py, pyaw in zip(x, y, yaw_deg)], axis=0)
    actions = finite_difference_actions(states)
    return ReferenceTrajectory(
        name=traj_type,
        states=states,
        actions=actions,
        times=t,
        metadata={"type": traj_type, "params": params},
    )


def load_reference_trajectory_from_csv(path: Path, max_length: Optional[int] = None) -> ReferenceTrajectory:
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    times: List[float] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [col for col in CSV_REQUIRED_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
        for row in reader:
            states.append(
                np.array(
                    [
                        float(row["pos_x"]),
                        float(row["pos_y"]),
                        float(row["pos_z"]),
                        float(row["rot_0"]),
                        float(row["rot_1"]),
                        float(row["rot_2"]),
                        float(row["rot_3"]),
                    ],
                    dtype=np.float32,
                )
            )
            actions.append(
                np.array(
                    [
                        float(row["throttle"]) if "throttle" in row and row["throttle"] not in {"", None} else 0.0,
                        float(row["steering"]) if "steering" in row and row["steering"] not in {"", None} else 0.0,
                    ],
                    dtype=np.float32,
                )
            )
            times.append(float(row["time"]))
            if max_length is not None and len(states) >= max_length:
                break

    if not states:
        raise RuntimeError(f"No valid rows found in {path}")

    return ReferenceTrajectory(
        name=path.stem,
        states=np.stack(states, axis=0),
        actions=np.stack(actions, axis=0),
        times=np.asarray(times, dtype=np.float32),
        metadata={"type": "follow", "source": str(path)},
    )


def estimate_forward_nominal_speed_from_trajectory(
    forward_bundle,
    device,
    initial_state: np.ndarray,
    control_dt: float,
) -> float:
    throttle_nominal = float(forward_bundle.normalizer.x_mean[7] + 2.0 * forward_bundle.normalizer.x_std[7])
    throttle_nominal = float(np.clip(throttle_nominal, 0.04, 0.12))
    history = deque(maxlen=int(forward_bundle.normalizer.seq_length))
    bootstrap_model_history(
        history,
        np.asarray(initial_state, dtype=np.float32).copy(),
        np.array([throttle_nominal, 0.0], dtype=np.float32),
        int(forward_bundle.normalizer.seq_length),
    )
    predicted_delta = predict_delta_state(np.stack(list(history), axis=0).astype(np.float32), forward_bundle, device)
    planar_step = float(np.linalg.norm(predicted_delta[:2]))
    return max(planar_step / max(control_dt, 1e-6), 1e-3)


def resample_reference_trajectory(
    trajectory: ReferenceTrajectory,
    target_dt: float,
    max_length: Optional[int] = None,
) -> ReferenceTrajectory:
    if trajectory.states.shape[0] < 2:
        return trajectory

    raw_dt = max(float(trajectory.times[1] - trajectory.times[0]), 1e-6)
    if raw_dt >= target_dt * 0.95:
        if max_length is None:
            return trajectory
        return ReferenceTrajectory(
            name=trajectory.name,
            states=trajectory.states[:max_length].copy(),
            actions=trajectory.actions[:max_length].copy(),
            times=trajectory.times[:max_length].copy(),
            metadata=dict(trajectory.metadata),
        )

    target_times = np.arange(float(trajectory.times[0]), float(trajectory.times[-1]) + 1e-9, target_dt, dtype=np.float64)
    indices = np.searchsorted(trajectory.times.astype(np.float64), target_times, side="left")
    indices = np.clip(indices, 0, trajectory.states.shape[0] - 1)
    if max_length is not None:
        indices = indices[:max_length]
        target_times = target_times[:max_length]

    time_scale = float(target_dt / raw_dt)
    states = trajectory.states[indices].astype(np.float32, copy=True)
    actions = trajectory.actions[indices].astype(np.float32, copy=True)
    actions[:, 0] = np.where(actions[:, 0] >= 0.0, np.clip(actions[:, 0] * time_scale, 0.0, 0.20), np.clip(actions[:, 0] * time_scale, -0.20, 0.0))
    actions[:, 1] = np.clip(actions[:, 1] * time_scale, -0.45, 0.45)

    metadata = dict(trajectory.metadata)
    metadata.update(
        {
            "resampled": True,
            "raw_dt": raw_dt,
            "target_dt": float(target_dt),
            "time_scale": time_scale,
            "source_length": int(trajectory.states.shape[0]),
        }
    )
    return ReferenceTrajectory(
        name=trajectory.name,
        states=states,
        actions=actions,
        times=target_times.astype(np.float32),
        metadata=metadata,
    )


def load_reference_trajectory_for_dt(
    path: Path,
    forward_bundle,
    device,
    control_dt: float,
    initial_state: Optional[np.ndarray] = None,
    max_length: Optional[int] = None,
) -> ReferenceTrajectory:
    raw = load_reference_trajectory_from_csv(path)
    if raw.states.shape[0] < 2:
        return raw

    nominal_state = raw.states[0] if initial_state is None else np.asarray(initial_state, dtype=np.float32)
    nominal_speed = estimate_forward_nominal_speed_from_trajectory(forward_bundle, device, nominal_state, control_dt)
    deltas = np.diff(raw.states[:, :3], axis=0)
    dts = np.diff(raw.times)
    valid = dts > 1e-6
    if np.any(valid):
        raw_speeds = np.linalg.norm(deltas[valid], axis=1) / dts[valid]
        reference_speed = float(np.mean(raw_speeds)) if raw_speeds.size > 0 else nominal_speed
    else:
        reference_speed = nominal_speed
    raw_dt = max(float(raw.times[1] - raw.times[0]), 1e-6)
    route_dt = float(np.clip(control_dt * nominal_speed / max(reference_speed, 1e-6), raw_dt, control_dt))

    resampled = resample_reference_trajectory(raw, route_dt, max_length=max_length)
    metadata = dict(resampled.metadata)
    metadata.update(
        {
            "nominal_speed": nominal_speed,
            "reference_speed": reference_speed,
            "route_dt": route_dt,
        }
    )
    return ReferenceTrajectory(
        name=resampled.name,
        states=resampled.states,
        actions=resampled.actions,
        times=resampled.times,
        metadata=metadata,
    )


def list_reference_csvs(data_root: Path) -> List[Path]:
    csvs = list(data_root.glob("*.csv"))
    csvs.extend(data_root.glob("*/*.csv"))
    return sorted({path.resolve(): path for path in csvs}.values())


def sample_follow_trajectory(
    data_root: Path,
    segment_length: int,
    rng: random.Random,
    *,
    forward_bundle=None,
    device=None,
    control_dt: Optional[float] = None,
) -> ReferenceTrajectory:
    csv_files = list_reference_csvs(data_root)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {data_root}")

    chosen = rng.choice(csv_files)
    if forward_bundle is not None and device is not None and control_dt is not None:
        full_traj = load_reference_trajectory_for_dt(chosen, forward_bundle, device, control_dt)
    else:
        full_traj = load_reference_trajectory_from_csv(chosen)
    if full_traj.states.shape[0] <= segment_length:
        return full_traj

    start = rng.randint(0, full_traj.states.shape[0] - segment_length)
    end = start + segment_length
    return ReferenceTrajectory(
        name=f"{full_traj.name}_{start}_{end}",
        states=full_traj.states[start:end].copy(),
        actions=full_traj.actions[start:end].copy(),
        times=full_traj.times[start:end].copy(),
        metadata={"type": "follow", "source": str(chosen), "start": start, "end": end},
    )


def save_reference_trajectory_csv(trajectory: ReferenceTrajectory, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", *STATE_COLUMNS, *ACTION_COLUMNS])
        for time_value, state, action in zip(trajectory.times, trajectory.states, trajectory.actions):
            writer.writerow([float(time_value), *state.tolist(), *action.tolist()])


def parse_kv_params(items: Sequence[str]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for item in items:
        key, value = item.split("=", 1)
        parsed[key.strip()] = float(value.strip())
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate or export reference trajectories for RL controller training.")
    parser.add_argument("--type", required=True, choices=["straight", "circle", "figure8", "sine", "s_curve"])
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--param", action="append", default=[], help="Trajectory parameter in key=value format.")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    params = parse_kv_params(args.param)
    trajectory = generate_reference_trajectory(args.type, args.length, params=params)
    output_path = args.output
    if not output_path.is_absolute():
        output_path = Path(find_project_root()) / output_path
    save_reference_trajectory_csv(trajectory, output_path)
    print(f"Saved {trajectory.name} trajectory with {trajectory.states.shape[0]} steps to {output_path}")


if __name__ == "__main__":
    main()
