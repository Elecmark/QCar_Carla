import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


STATE_COLUMNS = ["pos_x", "pos_y", "pos_z", "rot_0", "rot_1", "rot_2", "rot_3"]
ACTION_COLUMNS = ["throttle", "steering"]
INPUT_COLUMNS = STATE_COLUMNS + ACTION_COLUMNS
ROTATION_ORDER = "xzyw"
SEQ_LENGTH = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
TRAIN_RATIO = 0.8
GRAD_CLIP_NORM = 1.0
EPS = 1e-6
LABEL_SOURCE = "linear_speed_sign_yaw_delta_relative_pose_xzyw_v5"
STATIONARY_STEERING_THRESHOLD = 0.05
MAX_YAW_DELTA_DEG = 30.0
DATASET_DIRS = [
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


@dataclass
class Episode:
    source: str
    inputs: np.ndarray
    targets: np.ndarray


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def normalize_quaternion_xyzw(quat: np.ndarray) -> np.ndarray:
    out = quat.astype(np.float32, copy=True)
    norm = float(np.linalg.norm(out))
    if norm > 1e-8:
        out /= norm
    return out


def raw_quaternion_to_xyzw(quat_raw: np.ndarray) -> np.ndarray:
    quat_raw = quat_raw.astype(np.float32, copy=False)
    return np.array([quat_raw[0], quat_raw[2], quat_raw[1], quat_raw[3]], dtype=np.float32)


def xyzw_to_raw_quaternion(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = quat_xyzw.astype(np.float32, copy=False)
    return np.array([quat_xyzw[0], quat_xyzw[2], quat_xyzw[1], quat_xyzw[3]], dtype=np.float32)


def normalize_quaternion_raw(quat_raw: np.ndarray) -> np.ndarray:
    return xyzw_to_raw_quaternion(normalize_quaternion_xyzw(raw_quaternion_to_xyzw(quat_raw)))


def align_quaternion_xyzw(reference_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    ref = normalize_quaternion_xyzw(reference_quat)
    tgt = normalize_quaternion_xyzw(target_quat)
    if float(np.dot(ref, tgt)) < 0.0:
        tgt = -tgt
    return tgt


def align_quaternion_raw(reference_quat_raw: np.ndarray, target_quat_raw: np.ndarray) -> np.ndarray:
    aligned_xyzw = align_quaternion_xyzw(
        raw_quaternion_to_xyzw(reference_quat_raw),
        raw_quaternion_to_xyzw(target_quat_raw),
    )
    return xyzw_to_raw_quaternion(aligned_xyzw)


def quat_xyzw_to_yaw_deg(quat_xyzw: np.ndarray) -> float:
    qx, qy, qz, qw = normalize_quaternion_xyzw(quat_xyzw)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


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


def canonicalize_position_history(x: np.ndarray) -> np.ndarray:
    out = x.astype(np.float32, copy=True)
    anchor = out[:, -1:, :3]
    out[:, :, :3] = out[:, :, :3] - anchor

    for sample_idx in range(out.shape[0]):
        anchor_quat = raw_quaternion_to_xyzw(out[sample_idx, -1, 3:7])
        anchor_inv = quat_xyzw_inverse(anchor_quat)
        for step_idx in range(out.shape[1]):
            step_quat = raw_quaternion_to_xyzw(out[sample_idx, step_idx, 3:7])
            rel_quat = normalize_quaternion_xyzw(quat_xyzw_multiply(step_quat, anchor_inv))
            out[sample_idx, step_idx, 3:7] = xyzw_to_raw_quaternion(rel_quat)
    return out


def world_delta_to_body_delta(
    prev_state: np.ndarray,
    current_state: np.ndarray,
    next_state: np.ndarray,
    direction: str,
) -> np.ndarray:
    prev_carla = np.array([-prev_state[2], -prev_state[0], prev_state[1]], dtype=np.float32)
    current_carla = np.array([-current_state[2], -current_state[0], current_state[1]], dtype=np.float32)
    next_carla = np.array([-next_state[2], -next_state[0], next_state[1]], dtype=np.float32)

    current_xy = current_carla[:2]
    prev_xy = prev_carla[:2]
    next_xy = next_carla[:2]

    forward_vec = next_xy - prev_xy
    forward_norm = float(np.linalg.norm(forward_vec))

    if forward_norm < 1e-8:
        forward_vec = current_xy - prev_xy
        forward_norm = float(np.linalg.norm(forward_vec))
    if forward_norm < 1e-8:
        forward_vec = next_xy - current_xy
        forward_norm = float(np.linalg.norm(forward_vec))
    if forward_norm < 1e-8:
        sign = 1.0 if direction == "forward" else -1.0
        forward_vec = np.array([sign, 0.0], dtype=np.float32)
        forward_norm = 1.0

    forward_hat = forward_vec / forward_norm
    if direction == "backward":
        forward_hat = -forward_hat
    lateral_hat = np.array([-forward_hat[1], forward_hat[0]], dtype=np.float32)

    world_delta_xy = next_xy - current_xy
    dx_body = float(np.dot(world_delta_xy, forward_hat))
    dy_body = float(np.dot(world_delta_xy, lateral_hat))
    dz_body = float(next_carla[2] - current_carla[2])
    return np.array([dx_body, dy_body, dz_body], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QCar LSTM world models for forward and backward driving.")
    parser.add_argument("--data-root", type=Path, default=Path("QCarDataSet"))
    parser.add_argument("--output-dir", type=Path, default=Path("PDHModel"))
    parser.add_argument("--seq-length", type=int, default=SEQ_LENGTH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--inspect-labels", action="store_true")
    parser.add_argument("--inspect-samples", type=int, default=5)
    parser.add_argument("--directions", nargs="+", choices=["forward", "backward"], default=["forward", "backward"])
    parser.add_argument("--reset-training-state", action="store_true")
    return parser.parse_args()


def list_candidate_dirs(data_root: Path, direction: str) -> List[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    if direction not in {"forward", "backward"}:
        raise ValueError(f"Unsupported direction: {direction}")

    available_dirs = {p.name: p for p in data_root.iterdir() if p.is_dir()}
    missing_dirs = [name for name in DATASET_DIRS if name not in available_dirs]
    if missing_dirs:
        raise FileNotFoundError(f"Dataset directories not found under {data_root}: {missing_dirs}")

    # Read the curated dataset folders, then split rows by measured motion so the
    # forward/backward datasets come from the same source pool.
    return [available_dirs[name] for name in DATASET_DIRS]


def sign_label(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def create_load_stats() -> Dict[str, int | float]:
    return {
        "directories": 0,
        "csv_files": 0,
        "total_rows": 0,
        "forward_rows": 0,
        "backward_rows": 0,
        "stationary_rows": 0,
        "accepted_rows": 0,
        "trainable_files": 0,
        "windows": 0,
        "sign_mismatch_rows": 0,
        "sign_mismatch_nonzero_rows": 0,
        "stationary_turn_rows": 0,
        "accepted_stationary_turn_rows": 0,
    }


def finalize_load_stats(stats: Dict[str, int | float]) -> Dict[str, int | float]:
    total_rows = int(stats["total_rows"])
    nonzero_rows = int(stats["forward_rows"]) + int(stats["backward_rows"])
    mismatch_rows = int(stats["sign_mismatch_rows"])
    mismatch_nonzero_rows = int(stats["sign_mismatch_nonzero_rows"])
    stats["sign_mismatch_ratio"] = (mismatch_rows / total_rows) if total_rows else 0.0
    stats["sign_mismatch_nonzero_ratio"] = (mismatch_nonzero_rows / nonzero_rows) if nonzero_rows else 0.0
    return stats


def print_data_loading_report(direction: str, stats: Dict[str, int | float]) -> None:
    print(
        f"[{direction}] rows total={int(stats['total_rows'])} "
        f"forward={int(stats['forward_rows'])} backward={int(stats['backward_rows'])} "
        f"stationary={int(stats['stationary_rows'])} accepted={int(stats['accepted_rows'])}",
        flush=True,
    )
    print(
        f"[{direction}] throttle/linear_speed sign mismatch: "
        f"all_rows={int(stats['sign_mismatch_rows'])} ({float(stats['sign_mismatch_ratio']) * 100.0:.2f}%) "
        f"nonzero_speed={int(stats['sign_mismatch_nonzero_rows'])} "
        f"({float(stats['sign_mismatch_nonzero_ratio']) * 100.0:.2f}%)",
        flush=True,
    )
    if float(stats["sign_mismatch_nonzero_ratio"]) > 0.10:
        print(
            f"[{direction}] WARNING: throttle and linear_speed disagree on more than 10% of nonzero-speed samples.",
            flush=True,
        )


def read_filtered_rows(csv_path: Path, direction: str, stats: Dict[str, int | float]) -> List[Dict[str, float]]:
    target_sign = 1 if direction == "forward" else -1
    rows: List[Dict[str, float]] = []

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(INPUT_COLUMNS) | {"linear_speed"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        for raw in reader:
            throttle = float(raw["throttle"])
            linear_speed = float(raw["linear_speed"])
            steering = float(raw["steering"])
            throttle_sign = sign_label(throttle)
            speed_sign = sign_label(linear_speed)

            stats["total_rows"] += 1
            if speed_sign > 0:
                stats["forward_rows"] += 1
            elif speed_sign < 0:
                stats["backward_rows"] += 1
            else:
                stats["stationary_rows"] += 1

            if throttle_sign != speed_sign:
                stats["sign_mismatch_rows"] += 1
            if speed_sign != 0 and throttle_sign != speed_sign:
                stats["sign_mismatch_nonzero_rows"] += 1

            is_stationary_turn = speed_sign == 0 and abs(steering) >= STATIONARY_STEERING_THRESHOLD
            if is_stationary_turn:
                stats["stationary_turn_rows"] += 1

            accept_row = False
            if speed_sign == target_sign:
                accept_row = True
            elif is_stationary_turn and throttle_sign == target_sign:
                accept_row = True
                stats["accepted_stationary_turn_rows"] += 1

            if not accept_row:
                continue

            row = {column: float(raw[column]) for column in INPUT_COLUMNS}
            rows.append(row)

    return rows


def build_episode(rows: Sequence[Dict[str, float]], csv_path: Path, seq_length: int, direction: str) -> Episode | None:
    if len(rows) <= seq_length:
        return None

    inputs: List[List[float]] = []
    targets: List[List[float]] = []
    for start in range(len(rows) - seq_length):
        end = start + seq_length
        sequence = rows[start:end]
        prev_index = max(end - 2, 0)
        prev_state = np.array([rows[prev_index][column] for column in STATE_COLUMNS], dtype=np.float32)
        current_state = np.array([rows[end - 1][column] for column in STATE_COLUMNS], dtype=np.float32)
        next_state = np.array([rows[end][column] for column in STATE_COLUMNS], dtype=np.float32)
        body_pos_delta = world_delta_to_body_delta(prev_state, current_state, next_state, direction)
        current_quat_raw = normalize_quaternion_raw(current_state[3:7])
        next_quat_raw = align_quaternion_raw(current_quat_raw, next_state[3:7])
        current_quat = raw_quaternion_to_xyzw(current_quat_raw)
        next_quat = raw_quaternion_to_xyzw(next_quat_raw)
        relative_quat = normalize_quaternion_xyzw(quat_xyzw_multiply(next_quat, quat_xyzw_inverse(current_quat)))
        yaw_delta_deg = wrap_angle_deg(quat_xyzw_to_yaw_deg(relative_quat))
        yaw_delta_deg = float(np.clip(yaw_delta_deg, -MAX_YAW_DELTA_DEG, MAX_YAW_DELTA_DEG))
        delta = np.concatenate([body_pos_delta, np.array([yaw_delta_deg], dtype=np.float32)], axis=0).astype(np.float32)
        window = [[frame[column] for column in INPUT_COLUMNS] for frame in sequence]
        inputs.append(window)
        targets.append(delta.tolist())

    if not inputs:
        return None

    return Episode(
        source=str(csv_path),
        inputs=np.asarray(inputs, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
    )


def load_direction_episodes(data_root: Path, direction: str, seq_length: int) -> Tuple[List[Episode], Dict[str, int]]:
    episodes: List[Episode] = []
    stats = create_load_stats()

    for directory in list_candidate_dirs(data_root, direction):
        stats["directories"] += 1
        for csv_path in sorted(directory.glob("*.csv")):
            stats["csv_files"] += 1
            rows = read_filtered_rows(csv_path, direction, stats)
            stats["accepted_rows"] += len(rows)
            episode = build_episode(rows, csv_path, seq_length, direction)
            if episode is None:
                continue
            stats["trainable_files"] += 1
            stats["windows"] += len(episode.inputs)
            episodes.append(episode)

    if not episodes:
        raise RuntimeError(f"No usable {direction} episodes found under {data_root}")

    finalize_load_stats(stats)
    return episodes, stats


def print_label_report(direction: str, episodes: Sequence[Episode], num_samples: int) -> None:
    _, y = merge_episodes(episodes)
    names = ["body_dx", "body_dy", "body_dz", "yaw_delta_deg"]

    print(f"\n[{direction}] label inspection")
    print(f"  episodes={len(episodes)} windows={len(y)}")
    for index, name in enumerate(names):
        values = y[:, index]
        q10, q50, q90 = np.quantile(values, [0.1, 0.5, 0.9])
        print(
            f"  {name}: mean={values.mean():+.6f} std={values.std():.6f} "
            f"min={values.min():+.6f} q10={q10:+.6f} median={q50:+.6f} q90={q90:+.6f} max={values.max():+.6f}"
        )

    printed = 0
    for episode in episodes:
        for sample_idx in range(len(episode.targets)):
            last_input = episode.inputs[sample_idx, -1]
            target = episode.targets[sample_idx]
            print(
                f"  sample[{printed}] src={episode.source} "
                f"state7={np.array2string(last_input[:7], precision=5, separator=', ')} "
                f"action={np.array2string(last_input[7:], precision=5, separator=', ')} "
                f"target7={np.array2string(target, precision=5, separator=', ')}"
            )
            printed += 1
            if printed >= num_samples:
                return


def print_steering_bucket_report(direction: str, episodes: Sequence[Episode]) -> None:
    x, y = merge_episodes(episodes)
    steering = x[:, -1, 8]
    buckets = [
        ("left", steering < -0.05),
        ("straight", np.abs(steering) <= 0.05),
        ("right", steering > 0.05),
    ]
    print(f"\n[{direction}] steering bucket inspection")
    for name, mask in buckets:
        if not np.any(mask):
            print(f"  {name}: no samples")
            continue
        subset_y = y[mask]
        subset_steer = steering[mask]
        print(
            f"  {name}: n={len(subset_y)} "
            f"steer_mean={subset_steer.mean():+.5f} "
            f"body_dx_mean={subset_y[:, 0].mean():+.6f} "
            f"body_dy_mean={subset_y[:, 1].mean():+.6f} "
            f"yaw_delta_mean={subset_y[:, 3].mean():+.6f}"
        )


def split_episode(episode: Episode, train_ratio: float) -> Tuple[Episode | None, Episode | None]:
    num_windows = len(episode.inputs)
    split_index = int(num_windows * train_ratio)

    if num_windows >= 2:
        split_index = max(1, min(split_index, num_windows - 1))

    train_episode = None
    val_episode = None

    if split_index > 0:
        train_episode = Episode(
            source=episode.source,
            inputs=episode.inputs[:split_index],
            targets=episode.targets[:split_index],
        )
    if split_index < num_windows:
        val_episode = Episode(
            source=episode.source,
            inputs=episode.inputs[split_index:],
            targets=episode.targets[split_index:],
        )

    return train_episode, val_episode


def merge_episodes(episodes: Sequence[Episode]) -> Tuple[np.ndarray, np.ndarray]:
    x_parts = [episode.inputs for episode in episodes if len(episode.inputs) > 0]
    y_parts = [episode.targets for episode in episodes if len(episode.targets) > 0]
    if not x_parts or not y_parts:
        raise RuntimeError("No windows available after merging episodes")
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def compute_normalization(x_train: np.ndarray, y_train: np.ndarray) -> Dict[str, torch.Tensor | int]:
    x_train_canonical = canonicalize_position_history(x_train)
    x_mean = torch.from_numpy(x_train_canonical.mean(axis=(0, 1)).astype(np.float32))
    x_std = torch.from_numpy(x_train_canonical.std(axis=(0, 1)).astype(np.float32)).clamp_min(EPS)
    y_mean = torch.from_numpy(y_train.mean(axis=0).astype(np.float32))
    y_std = torch.from_numpy(y_train.std(axis=0).astype(np.float32)).clamp_min(EPS)
    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def normalize_features(x: np.ndarray, stats: Dict[str, torch.Tensor | int]) -> np.ndarray:
    x = canonicalize_position_history(x)
    x_mean = stats["x_mean"].numpy().reshape(1, 1, -1)
    x_std = stats["x_std"].numpy().reshape(1, 1, -1)
    return ((x - x_mean) / x_std).astype(np.float32)


def normalize_targets(y: np.ndarray, stats: Dict[str, torch.Tensor | int]) -> np.ndarray:
    y_mean = stats["y_mean"].numpy().reshape(1, -1)
    y_std = stats["y_std"].numpy().reshape(1, -1)
    return ((y - y_mean) / y_std).astype(np.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float = GRAD_CLIP_NORM,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def train_direction(
    direction: str,
    episodes: Sequence[Episode],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    training_stats: Dict[str, object],
    stats_path: Path,
) -> Dict[str, object]:
    train_episodes: List[Episode] = []
    val_episodes: List[Episode] = []

    for episode in episodes:
        train_episode, val_episode = split_episode(episode, args.train_ratio)
        if train_episode is not None:
            train_episodes.append(train_episode)
        if val_episode is not None:
            val_episodes.append(val_episode)

    if not train_episodes or not val_episodes:
        raise RuntimeError(f"{direction} data could not be split into non-empty train/val sets")

    x_train, y_train = merge_episodes(train_episodes)
    x_val, y_val = merge_episodes(val_episodes)

    norm_stats = compute_normalization(x_train, y_train)
    norm_stats["seq_length"] = int(args.seq_length)

    x_train_norm = normalize_features(x_train, norm_stats)
    x_val_norm = normalize_features(x_val, norm_stats)
    y_train_norm = normalize_targets(y_train, norm_stats)
    y_val_norm = normalize_targets(y_val, norm_stats)

    train_dataset = SequenceDataset(x_train_norm, y_train_norm)
    val_dataset = SequenceDataset(x_val_norm, y_val_norm)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = QCarWorldModel(input_dim=9, hidden_dim=args.hidden_dim, num_layers=args.num_layers, output_dim=4).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    train_losses: List[float] = []
    val_losses: List[float] = []
    model_path = output_dir / f"{direction}_world_model.pth"
    norm_path = output_dir / f"{direction}_normalization.pt"
    checkpoint_path = output_dir / f"{direction}_training_state.pt"
    start_epoch = 1

    if args.reset_training_state and checkpoint_path.exists():
        checkpoint_path.unlink()

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint_label_source = checkpoint.get("label_source")
        checkpoint_seq_length = int(checkpoint.get("seq_length", -1))
        if checkpoint_label_source == LABEL_SOURCE and checkpoint_seq_length == int(args.seq_length):
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_val_loss = float(checkpoint["best_val_loss"])
            best_epoch = int(checkpoint["best_epoch"])
            train_losses = [float(v) for v in checkpoint["train_loss_history"]]
            val_losses = [float(v) for v in checkpoint["val_loss_history"]]
            start_epoch = int(checkpoint["epoch"]) + 1
            print(f"[{direction}] resuming from epoch {start_epoch}", flush=True)
        else:
            print(
                f"[{direction}] ignoring existing checkpoint because label source changed "
                f"({checkpoint_label_source!r} -> {LABEL_SOURCE!r}) or seq_length differs.",
                flush=True,
            )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        epoch_time_sec = time.time() - epoch_start

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

        avg_epoch_time_sec = sum(
            training_stats[direction].get("epoch_time_history_sec", []) + [epoch_time_sec]
        ) / epoch
        eta_sec = avg_epoch_time_sec * (args.epochs - epoch)
        training_stats[direction] = {
            "direction": direction,
            "model_path": str(model_path),
            "normalization_path": str(norm_path),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_loss_history": train_losses,
            "val_loss_history": val_losses,
            "epoch_time_history_sec": training_stats.get(direction, {}).get("epoch_time_history_sec", []) + [epoch_time_sec],
            "num_train_sequences": int(len(train_dataset)),
            "num_val_sequences": int(len(val_dataset)),
            "num_train_episodes": int(len(train_episodes)),
            "num_val_episodes": int(len(val_episodes)),
        }
        torch.save(
            {
                "epoch": epoch,
                "label_source": LABEL_SOURCE,
                "seq_length": int(args.seq_length),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "train_loss_history": train_losses,
                "val_loss_history": val_losses,
            },
            checkpoint_path,
        )
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(training_stats, handle, indent=2)

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[{direction}] epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"best_val={best_val_loss:.6f}@{best_epoch} "
                f"lr={current_lr:.6g} "
                f"epoch_time={epoch_time_sec:.1f}s eta={eta_sec/60.0:.1f}m",
                flush=True,
            )

    torch.save(norm_stats, norm_path)
    training_stats[direction]["normalization_path"] = str(norm_path)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(training_stats, handle, indent=2)
    return training_stats[direction]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    stats_path = output_dir / "training_stats.json"

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(args.num_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(max(1, min(8, args.num_threads)))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    training_stats: Dict[str, object] = {
        "config": {
            "data_root": str(args.data_root),
            "output_dir": str(output_dir),
            "seq_length": args.seq_length,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "train_ratio": args.train_ratio,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "device": str(device),
            "gradient_clip_max_norm": GRAD_CLIP_NORM,
            "num_threads": args.num_threads,
            "print_every": args.print_every,
            "label_source": LABEL_SOURCE,
            "dataset_dirs": DATASET_DIRS,
        }
    }
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(training_stats, handle, indent=2)

    if args.inspect_labels:
        for direction in args.directions:
            episodes, load_stats = load_direction_episodes(args.data_root, direction, args.seq_length)
            print_data_loading_report(direction, load_stats)
            print(f"\n[{direction}] data_loading={load_stats}")
            print_label_report(direction, episodes, args.inspect_samples)
            print_steering_bucket_report(direction, episodes)
        return

    for direction in args.directions:
        episodes, load_stats = load_direction_episodes(args.data_root, direction, args.seq_length)
        print_data_loading_report(direction, load_stats)
        training_stats[direction] = {"data_loading": load_stats}
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(training_stats, handle, indent=2)
        direction_stats = train_direction(direction, episodes, args, output_dir, device, training_stats, stats_path)
        direction_stats["data_loading"] = load_stats
        training_stats[direction] = direction_stats

    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(training_stats, handle, indent=2)

    print(f"Saved training artifacts to {output_dir}")


if __name__ == "__main__":
    main()
