import argparse
import csv
import json
import random
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from carla_controller_PDH import find_project_root, load_bundle
from dt_model_env import DTModelEnv, EnvConfig
from policy_network import SACAgent, SACConfig
from reference_generator import (
    ReferenceTrajectory,
    generate_reference_trajectory,
    load_reference_trajectory_for_dt,
    sample_follow_trajectory,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        del args, kwargs

    def close(self) -> None:
        return None


def make_summary_writer(log_dir: Path):
    if SummaryWriter is None:
        return NullSummaryWriter()
    return SummaryWriter(log_dir=str(log_dir))


def save_png(image: np.ndarray, output_path: Path) -> None:
    image = np.asarray(image, dtype=np.uint8)
    height, width, channels = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    compressed = zlib.compress(raw, level=9)

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + chunk_type + data + struct.pack("!I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        handle.write(chunk(b"IDAT", compressed))
        handle.write(chunk(b"IEND", b""))


def draw_polyline(canvas: np.ndarray, points: np.ndarray, color: Tuple[int, int, int]) -> None:
    for p0, p1 in zip(points[:-1], points[1:]):
        x0, y0 = p0
        x1, y1 = p1
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, steps).astype(np.int32)
        ys = np.linspace(y0, y1, steps).astype(np.int32)
        valid = (xs >= 0) & (xs < canvas.shape[1]) & (ys >= 0) & (ys < canvas.shape[0])
        canvas[ys[valid], xs[valid]] = color


def create_line_chart(series_dict: Dict[str, Sequence[float]], width: int = 1200, height: int = 800) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    margin = 60
    colors = [(34, 87, 122), (219, 94, 36), (30, 140, 82)]
    all_values = [v for series in series_dict.values() for v in series]
    if not all_values:
        return canvas
    v_min = min(all_values)
    v_max = max(all_values)
    if abs(v_max - v_min) < 1e-6:
        v_max = v_min + 1.0
    for idx, values in enumerate(series_dict.values()):
        if len(values) < 2:
            continue
        points = []
        for i, value in enumerate(values):
            x = margin + i * (width - 2 * margin - 1) / max(1, len(values) - 1)
            y = height - margin - ((value - v_min) / (v_max - v_min)) * (height - 2 * margin - 1)
            points.append((x, y))
        draw_polyline(canvas, np.asarray(points, dtype=np.float32), colors[idx % len(colors)])
    canvas[margin:height - margin, margin] = (0, 0, 0)
    canvas[height - margin, margin:width - margin] = (0, 0, 0)
    return canvas


def sample_generated_trajectory(ref_type: str, length: int, rng: random.Random) -> ReferenceTrajectory:
    if ref_type == "mixed":
        ref_type = rng.choice(["straight", "circle", "figure8", "sine", "s_curve"])
    params = {"dt": 0.1}
    if ref_type == "circle":
        params["radius"] = rng.uniform(2.0, 6.0)
    elif ref_type in {"sine", "s_curve"}:
        params["amplitude"] = rng.uniform(0.8, 2.5)
    return generate_reference_trajectory(ref_type, length, params=params)


def choose_reference_trajectory(
    args: argparse.Namespace,
    rng: random.Random,
    data_root: Path,
    forward_bundle,
    device: torch.device,
) -> ReferenceTrajectory:
    if args.reference_csv is not None:
        return load_reference_trajectory_for_dt(
            args.reference_csv,
            forward_bundle,
            device,
            control_dt=args.fixed_delta,
            max_length=args.max_steps + 1,
        )
    if args.follow_probability > 0.0 and rng.random() < args.follow_probability:
        return sample_follow_trajectory(
            data_root,
            args.max_steps + 1,
            rng,
            forward_bundle=forward_bundle,
            device=device,
            control_dt=args.fixed_delta,
        )
    return sample_generated_trajectory(args.ref_type, args.max_steps + 1, rng)


def collect_supervised_batch(env: DTModelEnv, trajectory: ReferenceTrajectory) -> Tuple[torch.Tensor, torch.Tensor]:
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    obs, _ = env.reset(reference_trajectory=trajectory)
    horizon = min(len(trajectory.actions), env.env_config.max_steps)
    for step in range(horizon):
        expert_action = trajectory.actions[step].astype(np.float32, copy=True)
        action = env.expert_action_to_policy_target(expert_action)
        observations.append(obs.copy())
        actions.append(action.copy())
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return torch.from_numpy(np.asarray(observations, dtype=np.float32)), torch.from_numpy(np.asarray(actions, dtype=np.float32))


def write_metrics_csv(output_path: Path, records: Sequence[Dict[str, float]]) -> None:
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an imitation controller for DT trajectory tracking.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batches-per-epoch", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ref-type", default="mixed", choices=["mixed", "straight", "circle", "figure8", "sine", "s_curve"])
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--follow-probability", type=float, default=1.0)
    parser.add_argument("--fixed-delta", type=float, default=0.05)
    parser.add_argument("--data-root", type=Path, default=Path("QCarDataSet"))
    parser.add_argument("--output-dir", type=Path, default=Path("PDHModel/il_controller_qcardataset"))
    parser.add_argument("--forward-model-path", type=Path, default=Path("PDHModel/forward_world_model.pth"))
    parser.add_argument("--forward-norm-path", type=Path, default=Path("PDHModel/forward_normalization.pt"))
    parser.add_argument("--backward-model-path", type=Path, default=Path("PDHModel/backward_world_model.pth"))
    parser.add_argument("--backward-norm-path", type=Path, default=Path("PDHModel/backward_normalization.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(find_project_root())
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    forward_bundle = load_bundle("forward", str(project_root / args.forward_model_path), str(project_root / args.forward_norm_path), device)
    backward_bundle = load_bundle("backward", str(project_root / args.backward_model_path), str(project_root / args.backward_norm_path), device)
    env = DTModelEnv(forward_bundle, backward_bundle, device=device, env_config=EnvConfig(max_steps=args.max_steps))
    obs_dim = int(env.observation_space.shape[0])
    agent = SACAgent(SACConfig(obs_dim=obs_dim, action_dim=2), device)
    writer = make_summary_writer(output_dir / "tensorboard")

    records: List[Dict[str, float]] = []
    train_losses: List[float] = []

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for batch_idx in range(args.batches_per_epoch):
            trajectory = choose_reference_trajectory(args, rng, project_root / args.data_root, forward_bundle, device)
            observations, actions = collect_supervised_batch(env, trajectory)
            if len(observations) == 0:
                continue
            if len(observations) > args.batch_size:
                indices = torch.randint(0, len(observations), (args.batch_size,))
                batch_obs = observations[indices]
                batch_actions = actions[indices]
            else:
                batch_obs = observations
                batch_actions = actions
            loss = agent.behavior_clone_loss(batch_obs, batch_actions)
            agent.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            agent.actor_optimizer.step()
            epoch_losses.append(float(loss.item()))
            writer.add_scalar("il/batch_loss", float(loss.item()), (epoch - 1) * args.batches_per_epoch + batch_idx)

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_losses.append(mean_loss)
        records.append({"epoch": float(epoch), "train_loss": mean_loss})
        writer.add_scalar("il/epoch_loss", mean_loss, epoch)
        print(f"epoch={epoch:04d} train_loss={mean_loss:.6f}")

    policy_path = output_dir / "policy_controller_il.pth"
    payload = {
        "type": "imitation_policy",
        "config": agent.config.__dict__,
        "actor": agent.actor.state_dict(),
    }
    torch.save(payload, policy_path)

    with (output_dir / "policy_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "policy_path": str(policy_path),
                "obs_dim": obs_dim,
                "action_dim": 2,
                "epochs": args.epochs,
                "batches_per_epoch": args.batches_per_epoch,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "source_data_root": str(project_root / args.data_root),
            },
            handle,
            indent=2,
        )

    chart = create_line_chart({"train_loss": train_losses})
    save_png(chart, output_dir / "training_curves.png")
    write_metrics_csv(output_dir / "training_metrics.csv", records)
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "epochs": args.epochs,
                "final_train_loss": train_losses[-1] if train_losses else 0.0,
                "tensorboard_enabled": SummaryWriter is not None,
            },
            handle,
            indent=2,
        )
    writer.close()
    print(f"saved_il_policy={policy_path}")


if __name__ == "__main__":
    main()
