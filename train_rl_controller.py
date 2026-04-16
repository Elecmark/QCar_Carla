import argparse
import csv
import json
import random
import struct
import zlib
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Sequence, Tuple

import numpy as np
import torch

from carla_controller_PDH import find_project_root, load_bundle
from dt_model_env import DTModelEnv, EnvConfig
from policy_network import SACAgent, SACConfig
from reference_generator import (
    ReferenceTrajectory,
    generate_reference_trajectory,
    list_reference_csvs,
    load_reference_trajectory_for_dt,
    sample_follow_trajectory,
    save_reference_trajectory_csv,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[indices]),
            "actions": torch.from_numpy(self.actions[indices]),
            "rewards": torch.from_numpy(self.rewards[indices]),
            "next_obs": torch.from_numpy(self.next_obs[indices]),
            "dones": torch.from_numpy(self.dones[indices]),
        }


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        del args, kwargs

    def close(self) -> None:
        return None


def make_summary_writer(log_dir: Path):
    if SummaryWriter is None:
        return NullSummaryWriter()
    return SummaryWriter(log_dir=str(log_dir))


def smooth_series(values: Sequence[float], alpha: float = 0.1) -> List[float]:
    smoothed: List[float] = []
    running = 0.0
    for idx, value in enumerate(values):
        running = value if idx == 0 else (1.0 - alpha) * running + alpha * value
        smoothed.append(running)
    return smoothed


def save_png(image: np.ndarray, output_path: Path) -> None:
    image = np.asarray(image, dtype=np.uint8)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("PNG writer expects an RGB image with shape [H, W, 3].")

    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    compressor = zlib.compress(raw, level=9)

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + chunk_type + data + struct.pack("!I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        handle.write(chunk(b"IDAT", compressor))
        handle.write(chunk(b"IEND", b""))


def draw_polyline(canvas: np.ndarray, points: np.ndarray, color: Tuple[int, int, int]) -> None:
    if points.shape[0] < 2:
        return
    for p0, p1 in zip(points[:-1], points[1:]):
        x0, y0 = p0
        x1, y1 = p1
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        if steps <= 0:
            continue
        xs = np.linspace(x0, x1, steps).astype(np.int32)
        ys = np.linspace(y0, y1, steps).astype(np.int32)
        valid = (xs >= 0) & (xs < canvas.shape[1]) & (ys >= 0) & (ys < canvas.shape[0])
        canvas[ys[valid], xs[valid]] = color


def create_line_chart(series_dict: Dict[str, Sequence[float]], width: int = 1200, height: int = 800) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    plot_margin = 60
    colors = [(34, 87, 122), (219, 94, 36), (30, 140, 82), (161, 84, 197)]
    all_values = [value for series in series_dict.values() for value in series]
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
            x = plot_margin + i * (width - 2 * plot_margin - 1) / max(1, len(values) - 1)
            y_norm = (value - v_min) / (v_max - v_min)
            y = height - plot_margin - y_norm * (height - 2 * plot_margin - 1)
            points.append((x, y))
        draw_polyline(canvas, np.asarray(points, dtype=np.float32), colors[idx % len(colors)])

    canvas[plot_margin:height - plot_margin, plot_margin] = (0, 0, 0)
    canvas[height - plot_margin, plot_margin:width - plot_margin] = (0, 0, 0)
    return canvas


def create_trajectory_plot(reference_states: np.ndarray, predicted_states: np.ndarray, width: int = 1200, height: int = 800) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    if len(reference_states) == 0 or len(predicted_states) == 0:
        return canvas

    points = np.concatenate([reference_states[:, :2], predicted_states[:, :2]], axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    margin = 60

    def project(xy: np.ndarray) -> np.ndarray:
        norm = (xy - mins) / span
        px = margin + norm[:, 0] * (width - 2 * margin)
        py = height - (margin + norm[:, 1] * (height - 2 * margin))
        return np.stack([px, py], axis=1)

    draw_polyline(canvas, project(reference_states[:, :2]), (34, 87, 122))
    draw_polyline(canvas, project(predicted_states[:, :2]), (219, 94, 36))
    return canvas


def save_training_curves(output_path: Path, metrics: Dict[str, Sequence[float]]) -> None:
    chart = create_line_chart({key: smooth_series(values) for key, values in metrics.items()})
    save_png(chart, output_path)


def save_trajectory_plot(output_path: Path, reference_states: np.ndarray, predicted_states: np.ndarray) -> None:
    plot = create_trajectory_plot(reference_states, predicted_states)
    save_png(plot, output_path)


def write_metrics_csv(output_path: Path, records: Sequence[Dict[str, float]]) -> None:
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(records[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def sample_generated_trajectory(ref_type: str, length: int, rng: random.Random) -> ReferenceTrajectory:
    if ref_type == "mixed":
        candidates = ["straight", "circle", "figure8", "sine", "s_curve"]
        ref_type = rng.choice(candidates)

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

    if getattr(args, "full_reference_trajectories", False):
        csv_files = list_reference_csvs(data_root)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found under {data_root}")
        chosen = rng.choice(csv_files)
        return load_reference_trajectory_for_dt(
            chosen,
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


def collect_bc_batch(env: DTModelEnv, trajectory: ReferenceTrajectory) -> Tuple[torch.Tensor, torch.Tensor]:
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    obs, _ = env.reset(reference_trajectory=trajectory, initial_state=trajectory.states[0].copy())
    horizon = min(len(trajectory.actions), env.env_config.max_steps)
    for step in range(horizon):
        expert_action = env._project_final_action(trajectory.actions[step])
        policy_target = env.expert_action_to_policy_target(expert_action)
        observations.append(obs.copy())
        actions.append(policy_target)
        obs, _, terminated, truncated, _ = env.step(policy_target)
        if terminated or truncated:
            break
    return torch.from_numpy(np.asarray(observations, dtype=np.float32)), torch.from_numpy(np.asarray(actions, dtype=np.float32))


def behavior_clone_pretrain(
    agent: SACAgent,
    env: DTModelEnv,
    args: argparse.Namespace,
    rng: random.Random,
    data_root: Path,
    writer,
    forward_bundle,
    device: torch.device,
) -> None:
    bc_batch_size = min(args.batch_size, 256)
    last_loss = None
    for bc_idx in range(args.bc_warmup_episodes):
        trajectory = choose_reference_trajectory(args, rng, data_root, forward_bundle, device)
        observations, actions = collect_bc_batch(env, trajectory)
        if len(observations) == 0:
            continue

        for update_idx in range(args.bc_updates):
            if len(observations) > bc_batch_size:
                indices = torch.randint(0, len(observations), (bc_batch_size,))
                batch_obs = observations[indices]
                batch_actions = actions[indices]
            else:
                batch_obs = observations
                batch_actions = actions
            last_loss = agent.behavior_clone_loss(batch_obs, batch_actions)
            agent.actor_optimizer.zero_grad(set_to_none=True)
            last_loss.backward()
            agent.actor_optimizer.step()
            global_step = bc_idx * args.bc_updates + update_idx
            writer.add_scalar("warmup/bc_loss", float(last_loss.item()), global_step)

    if last_loss is not None:
        print(f"bc_pretrain_loss={float(last_loss.item()):.6f}")


def load_initial_policy(agent: SACAgent, init_policy_path: Path, device: torch.device) -> None:
    payload = torch.load(init_policy_path, map_location=device)
    agent.load_actor_state_dict(payload)
    print(f"loaded_init_policy={init_policy_path}")


def evaluate_policy(env: DTModelEnv, agent: SACAgent, trajectory: ReferenceTrajectory) -> Dict[str, object]:
    obs, _ = env.reset(reference_trajectory=trajectory)
    rewards = []
    done = False
    truncated = False
    while not (done or truncated):
        action = agent.select_action(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        del info
        rewards.append(reward)

    predicted_states = np.asarray(env.predicted_states, dtype=np.float32)
    reference_states = trajectory.states[1 : 1 + len(predicted_states)]
    pos_error = float(np.mean(np.linalg.norm(predicted_states[:, :3] - reference_states[:, :3], axis=1))) if len(predicted_states) else 0.0
    return {
        "mean_reward": float(np.mean(rewards) if rewards else 0.0),
        "mean_pos_error": pos_error,
        "predicted_states": predicted_states,
        "reference_states": reference_states,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an SAC RL controller on top of the PDH world model.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=829)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--start-random-steps", type=int, default=1000)
    parser.add_argument("--bc-warmup-episodes", type=int, default=12)
    parser.add_argument("--bc-updates", type=int, default=200)
    parser.add_argument("--ref-type", default="mixed", choices=["mixed", "straight", "circle", "figure8", "sine", "s_curve"])
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--follow-probability", type=float, default=1.0)
    parser.add_argument("--full-reference-trajectories", action="store_true")
    parser.add_argument("--fixed-delta", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=Path("QCarDataSet"))
    parser.add_argument("--output-dir", type=Path, default=Path("PDHModel/rl_qcardataset"))
    parser.add_argument("--forward-model-path", type=Path, default=Path("PDHModel/forward_world_model.pth"))
    parser.add_argument("--forward-norm-path", type=Path, default=Path("PDHModel/forward_normalization.pt"))
    parser.add_argument("--backward-model-path", type=Path, default=Path("PDHModel/backward_world_model.pth"))
    parser.add_argument("--backward-norm-path", type=Path, default=Path("PDHModel/backward_normalization.pt"))
    parser.add_argument("--init-policy", type=Path)
    parser.add_argument("--reset-position-noise-xy", type=float, default=0.75)
    parser.add_argument("--reset-yaw-noise-deg", type=float, default=20.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(find_project_root())
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_output_dir = output_dir / "reference_trajectories"
    ref_output_dir.mkdir(parents=True, exist_ok=True)

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
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    sac_config = SACConfig(obs_dim=obs_dim, action_dim=action_dim)
    agent = SACAgent(sac_config, device)
    if args.init_policy is not None:
        load_initial_policy(agent, project_root / args.init_policy, device)
    replay_buffer = ReplayBuffer(obs_dim, action_dim, args.buffer_size)
    writer = make_summary_writer(output_dir / "tensorboard")

    train_records: List[Dict[str, float]] = []
    eval_records: List[Dict[str, float]] = []
    metric_history = {"episode_reward": [], "episode_pos_error": [], "critic_loss": [], "actor_loss": [], "action_saturation_rate": []}
    total_steps = 0
    recent_losses: Deque[Dict[str, float]] = deque(maxlen=200)

    behavior_clone_pretrain(agent, env, args, rng, project_root / args.data_root, writer, forward_bundle, device)

    for episode in range(1, args.episodes + 1):
        trajectory = choose_reference_trajectory(args, rng, project_root / args.data_root, forward_bundle, device)
        save_reference_trajectory_csv(trajectory, ref_output_dir / f"episode_{episode:04d}_{trajectory.name}.csv")
        observation, reset_info = env.reset(reference_trajectory=trajectory)
        episode_reward = 0.0
        episode_pos_error = []
        episode_saturation = []
        step_count = 0
        done = False
        truncated = False

        while not (done or truncated):
            if total_steps < args.start_random_steps:
                low = env.action_space.low.astype(np.float32)
                high = env.action_space.high.astype(np.float32)
                action = np.random.uniform(low, high).astype(np.float32)
            else:
                action = agent.select_action(observation, deterministic=False).astype(np.float32)

            next_observation, reward, done, truncated, info = env.step(action)
            applied_action = np.array(
                [
                    float(info["applied_action_throttle"]),
                    float(info["applied_action_steering"]),
                ],
                dtype=np.float32,
            )
            replay_buffer.add(observation, applied_action, reward, next_observation, done or truncated)
            observation = next_observation
            episode_reward += reward
            episode_pos_error.append(info["pos_error"])
            saturation = float(abs(info["applied_action_steering"]) > 0.4 or info["applied_action_throttle"] > 0.1)
            episode_saturation.append(saturation)
            step_count += 1
            total_steps += 1

            if replay_buffer.size >= max(args.batch_size, args.warmup_steps):
                for _ in range(args.updates_per_step):
                    loss_dict = agent.update(replay_buffer.sample(args.batch_size))
                    recent_losses.append(loss_dict)

        avg_actor_loss = float(np.mean([item["actor_loss"] for item in recent_losses])) if recent_losses else 0.0
        avg_critic_loss = float(np.mean([item["critic_loss"] for item in recent_losses])) if recent_losses else 0.0
        avg_pos_error = float(np.mean(episode_pos_error)) if episode_pos_error else 0.0
        action_saturation_rate = float(np.mean(episode_saturation)) if episode_saturation else 0.0

        metric_history["episode_reward"].append(episode_reward)
        metric_history["episode_pos_error"].append(avg_pos_error)
        metric_history["critic_loss"].append(avg_critic_loss)
        metric_history["actor_loss"].append(avg_actor_loss)
        metric_history["action_saturation_rate"].append(action_saturation_rate)

        train_record = {
            "episode": float(episode),
            "steps": float(step_count),
            "episode_reward": float(episode_reward),
            "avg_pos_error": avg_pos_error,
            "action_saturation_rate": action_saturation_rate,
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "alpha": float(agent.alpha.item()),
        }
        train_records.append(train_record)

        writer.add_scalar("train/episode_reward", episode_reward, episode)
        writer.add_scalar("train/avg_pos_error", avg_pos_error, episode)
        writer.add_scalar("train/action_saturation_rate", action_saturation_rate, episode)
        writer.add_scalar("train/actor_loss", avg_actor_loss, episode)
        writer.add_scalar("train/critic_loss", avg_critic_loss, episode)
        writer.add_scalar("train/alpha", float(agent.alpha.item()), episode)

        print(
            f"episode={episode:04d} steps={step_count:03d} reward={episode_reward:+.4f} "
            f"pos_error={avg_pos_error:.4f} sat_rate={action_saturation_rate:.3f} "
            f"actor_loss={avg_actor_loss:.4f} critic_loss={avg_critic_loss:.4f} "
            f"ref={reset_info['reference_name']}"
        )

        if episode % args.eval_every == 0 or episode == args.episodes:
            eval_traj = sample_generated_trajectory("sine", args.max_steps + 1, rng)
            eval_result = evaluate_policy(env, agent, eval_traj)
            eval_record = {
                "episode": float(episode),
                "mean_reward": float(eval_result["mean_reward"]),
                "mean_pos_error": float(eval_result["mean_pos_error"]),
            }
            eval_records.append(eval_record)
            writer.add_scalar("eval/mean_reward", eval_record["mean_reward"], episode)
            writer.add_scalar("eval/mean_pos_error", eval_record["mean_pos_error"], episode)
            save_trajectory_plot(output_dir / f"trajectory_eval_{episode:04d}.png", eval_result["reference_states"], eval_result["predicted_states"])

    policy_path = output_dir / "policy_controller.pth"
    torch.save(agent.state_dict(), policy_path)

    config_payload = {
        "policy_path": str(policy_path),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "seed": args.seed,
        "reset_position_noise_xy": args.reset_position_noise_xy,
        "reset_yaw_noise_deg": args.reset_yaw_noise_deg,
        "sac_config": sac_config.__dict__,
        "world_model": {
            "forward_model_path": str(project_root / args.forward_model_path),
            "forward_norm_path": str(project_root / args.forward_norm_path),
            "backward_model_path": str(project_root / args.backward_model_path),
            "backward_norm_path": str(project_root / args.backward_norm_path),
        },
    }
    with (output_dir / "policy_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    save_training_curves(output_dir / "training_curves.png", metric_history)
    write_metrics_csv(output_dir / "training_metrics.csv", train_records)
    write_metrics_csv(output_dir / "evaluation_metrics.csv", eval_records)

    summary = {
        "episodes": args.episodes,
        "total_steps": total_steps,
        "final_reward": metric_history["episode_reward"][-1] if metric_history["episode_reward"] else 0.0,
        "final_pos_error": metric_history["episode_pos_error"][-1] if metric_history["episode_pos_error"] else 0.0,
        "tensorboard_enabled": SummaryWriter is not None,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    writer.close()
    print(f"Saved controller to {policy_path}")


if __name__ == "__main__":
    main()
