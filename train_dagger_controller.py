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

from carla_controller_PDH import choose_bundle_for_action, find_project_root, load_bundle, predict_delta_state, predicted_output_to_next_state
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


def write_metrics_csv(output_path: Path, records: Sequence[Dict[str, float]]) -> None:
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


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


def search_expert_action(env: DTModelEnv) -> np.ndarray:
    assert env.reference is not None
    assert env.state is not None
    current_state = env.state.copy()
    next_ref = env.reference.states[min(env.step_idx + 1, len(env.reference.states) - 1)]
    current_ref = env.reference.states[min(env.step_idx, len(env.reference.states) - 1)]
    target_step_distance = float(np.linalg.norm(next_ref[:2] - current_ref[:2]))
    target_yaw_step = float(np.mean(np.square(next_ref[3:7] - current_ref[3:7])))

    throttle_values = np.linspace(0.02, 0.12, 6, dtype=np.float32)
    steer_values = np.linspace(-0.35, 0.35, 15, dtype=np.float32)
    best_action = np.array([0.06, 0.0], dtype=np.float32)
    best_score = float("inf")
    prev_action = env.prev_action.copy() if env.prev_action is not None else np.array([0.06, 0.0], dtype=np.float32)

    for throttle in throttle_values:
        for steer in steer_values:
            candidate = env._project_action(np.array([float(throttle), float(steer)], dtype=np.float32))
            bundle = choose_bundle_for_action(candidate, env.forward_model, env.backward_model)
            history_np = env._history_array(candidate)
            predicted_delta = predict_delta_state(history_np, bundle, env.device)
            predicted_state = predicted_output_to_next_state(current_state, predicted_delta)

            pos_error = float(np.linalg.norm(predicted_state[:3] - next_ref[:3]))
            yaw_error = float(np.mean(np.square(predicted_state[3:7] - next_ref[3:7])))
            predicted_step_distance = float(np.linalg.norm(predicted_state[:2] - current_state[:2]))
            step_distance_error = abs(predicted_step_distance - target_step_distance)
            predicted_yaw_step = float(np.mean(np.square(predicted_state[3:7] - current_state[3:7])))
            yaw_step_error = abs(predicted_yaw_step - target_yaw_step)
            progress_bonus = float(np.linalg.norm(current_ref[:3] - current_state[:3]) - np.linalg.norm(next_ref[:3] - predicted_state[:3]))
            steer_penalty = max(0.0, abs(float(candidate[1])) - 0.28) ** 2
            throttle_penalty = max(0.0, float(candidate[0]) - 0.11) ** 2
            smooth_penalty = float(np.mean(np.square(candidate - prev_action)))
            score = (
                0.8 * pos_error
                + 1.8 * step_distance_error
                + 0.25 * yaw_error
                + 0.8 * yaw_step_error
                - 0.35 * progress_bonus
                + 0.8 * steer_penalty
                + 0.15 * throttle_penalty
                + 0.35 * smooth_penalty
            )
            if score < best_score:
                best_score = score
                best_action = candidate.copy()

    return best_action.astype(np.float32)


def reference_expert_action(env: DTModelEnv) -> np.ndarray:
    assert env.reference is not None
    action_idx = min(env.step_idx, env.reference.actions.shape[0] - 1)
    return env._project_action(env.reference.actions[action_idx].astype(np.float32, copy=True))


def load_initial_policy(agent: SACAgent, policy_path: Path, device: torch.device) -> None:
    payload = torch.load(policy_path, map_location=device)
    agent.load_actor_state_dict(payload)


def collect_dagger_rollout(
    env: DTModelEnv,
    agent: SACAgent,
    trajectory: ReferenceTrajectory,
    deterministic: bool,
    rollout_blend: float,
    expert_source: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    observations: List[np.ndarray] = []
    expert_actions: List[np.ndarray] = []
    observation, _ = env.reset(reference_trajectory=trajectory)
    done = False
    truncated = False
    rewards: List[float] = []
    pos_errors: List[float] = []

    while not (done or truncated):
        expert_action = reference_expert_action(env) if expert_source == "reference" else search_expert_action(env)
        observations.append(observation.copy())
        expert_actions.append(expert_action)

        action = agent.select_action(observation, deterministic=deterministic)
        if rollout_blend > 0.0:
            action = (1.0 - rollout_blend) * np.asarray(action, dtype=np.float32) + rollout_blend * expert_action
        observation, reward, done, truncated, info = env.step(action)
        rewards.append(float(reward))
        pos_errors.append(float(info["pos_error"]))

    stats = {
        "steps": float(len(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_pos_error": float(np.mean(pos_errors)) if pos_errors else 0.0,
        "final_pos_error": float(env.last_info.get("pos_error", 0.0)),
        "terminated": float(env.last_info.get("terminated", 0.0)),
    }
    return np.asarray(observations, dtype=np.float32), np.asarray(expert_actions, dtype=np.float32), stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DAgger controller on DT rollouts.")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--rollouts-per-iteration", type=int, default=8)
    parser.add_argument("--updates-per-iteration", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ref-type", default="mixed", choices=["mixed", "straight", "circle", "figure8", "sine", "s_curve"])
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--follow-probability", type=float, default=1.0)
    parser.add_argument("--fixed-delta", type=float, default=0.05)
    parser.add_argument("--expert-source", choices=["reference", "search"], default="reference")
    parser.add_argument("--data-root", type=Path, default=Path("PDHModel/reference_trajectories"))
    parser.add_argument("--init-policy", type=Path, required=True)
    parser.add_argument("--rollout-blend", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("PDHModel/dagger_controller"))
    parser.add_argument("--forward-model-path", type=Path, default=Path("PDHModel/forward_world_model.pth"))
    parser.add_argument("--forward-norm-path", type=Path, default=Path("PDHModel/forward_normalization.pt"))
    parser.add_argument("--backward-model-path", type=Path, default=Path("PDHModel/backward_world_model.pth"))
    parser.add_argument("--backward-norm-path", type=Path, default=Path("PDHModel/backward_normalization.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reset-position-noise-xy", type=float, default=0.75)
    parser.add_argument("--reset-yaw-noise-deg", type=float, default=20.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
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
    env = DTModelEnv(
        forward_bundle,
        backward_bundle,
        device=device,
        env_config=EnvConfig(
            max_steps=args.max_steps,
            reset_position_noise_xy=args.reset_position_noise_xy,
            reset_yaw_noise_deg=args.reset_yaw_noise_deg,
        ),
    )
    agent = SACAgent(SACConfig(obs_dim=int(env.observation_space.shape[0]), action_dim=2), device)
    load_initial_policy(agent, project_root / args.init_policy, device)

    writer = make_summary_writer(output_dir / "tensorboard")
    all_observations: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    records: List[Dict[str, float]] = []
    train_losses: List[float] = []
    rollout_errors: List[float] = []

    for iteration in range(1, args.iterations + 1):
        iter_stats: List[Dict[str, float]] = []
        for _ in range(args.rollouts_per_iteration):
            trajectory = choose_reference_trajectory(args, rng, project_root / args.data_root, forward_bundle, device)
            observations, actions, stats = collect_dagger_rollout(
                env=env,
                agent=agent,
                trajectory=trajectory,
                deterministic=True,
                rollout_blend=float(np.clip(args.rollout_blend, 0.0, 1.0)),
                expert_source=args.expert_source,
            )
            if len(observations) == 0:
                continue
            all_observations.append(observations)
            all_actions.append(actions)
            iter_stats.append(stats)

        if not all_observations:
            continue

        dataset_obs = torch.from_numpy(np.concatenate(all_observations, axis=0))
        dataset_actions = torch.from_numpy(np.concatenate(all_actions, axis=0))
        losses: List[float] = []
        for update_idx in range(args.updates_per_iteration):
            if len(dataset_obs) > args.batch_size:
                indices = torch.randint(0, len(dataset_obs), (args.batch_size,))
                batch_obs = dataset_obs[indices]
                batch_actions = dataset_actions[indices]
            else:
                batch_obs = dataset_obs
                batch_actions = dataset_actions
            loss = agent.behavior_clone_loss(batch_obs, batch_actions)
            agent.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            agent.actor_optimizer.step()
            loss_value = float(loss.item())
            losses.append(loss_value)
            writer.add_scalar("dagger/batch_loss", loss_value, (iteration - 1) * args.updates_per_iteration + update_idx)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_rollout_error = float(np.mean([stat["mean_pos_error"] for stat in iter_stats])) if iter_stats else 0.0
        train_losses.append(mean_loss)
        rollout_errors.append(mean_rollout_error)
        record = {
            "iteration": float(iteration),
            "dataset_size": float(len(dataset_obs)),
            "train_loss": mean_loss,
            "rollout_mean_pos_error": mean_rollout_error,
            "rollout_final_pos_error": float(np.mean([stat["final_pos_error"] for stat in iter_stats])) if iter_stats else 0.0,
            "termination_rate": float(np.mean([stat["terminated"] for stat in iter_stats])) if iter_stats else 0.0,
        }
        records.append(record)
        writer.add_scalar("dagger/iteration_loss", mean_loss, iteration)
        writer.add_scalar("dagger/rollout_mean_pos_error", mean_rollout_error, iteration)
        print(
            f"iteration={iteration:04d} dataset_size={len(dataset_obs)} "
            f"train_loss={mean_loss:.6f} rollout_mean_pos_error={mean_rollout_error:.4f}"
        )

    policy_path = output_dir / "policy_controller_dagger.pth"
    payload = {
        "type": "dagger_policy",
        "config": agent.config.__dict__,
        "actor": agent.actor.state_dict(),
    }
    torch.save(payload, policy_path)

    with (output_dir / "policy_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "policy_path": str(policy_path),
                "obs_dim": agent.config.obs_dim,
                "action_dim": 2,
                "iterations": args.iterations,
                "rollouts_per_iteration": args.rollouts_per_iteration,
                "updates_per_iteration": args.updates_per_iteration,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "rollout_blend": float(np.clip(args.rollout_blend, 0.0, 1.0)),
                "expert_source": args.expert_source,
                "source_data_root": str(project_root / args.data_root),
                "init_policy": str(project_root / args.init_policy),
            },
            handle,
            indent=2,
        )

    chart = create_line_chart({"train_loss": train_losses, "rollout_pos_error": rollout_errors})
    save_png(chart, output_dir / "training_curves.png")
    write_metrics_csv(output_dir / "training_metrics.csv", records)
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "iterations": args.iterations,
                "final_train_loss": train_losses[-1] if train_losses else 0.0,
                "final_rollout_mean_pos_error": rollout_errors[-1] if rollout_errors else 0.0,
                "expert_source": args.expert_source,
                "tensorboard_enabled": SummaryWriter is not None,
            },
            handle,
            indent=2,
        )
    writer.close()
    print(f"saved_dagger_policy={policy_path}")


if __name__ == "__main__":
    main()
