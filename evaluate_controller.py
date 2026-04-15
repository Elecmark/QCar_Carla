import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from carla_controller_PDH import find_project_root, load_bundle
from deploy_rl_controller import blend_with_reference_action, load_policy
from dt_model_env import DTModelEnv, EnvConfig
from reference_generator import list_reference_csvs, load_reference_trajectory_for_dt


def evaluate_one_reference(
    env: DTModelEnv,
    agent,
    reference_csv: Path,
    deterministic: bool,
    reference_action_blend: float,
    fixed_delta: float,
) -> Dict[str, float]:
    reference = load_reference_trajectory_for_dt(
        reference_csv,
        env.forward_model,
        env.device,
        control_dt=fixed_delta,
        max_length=env.env_config.max_steps + 1,
    )
    observation, _ = env.reset(reference_trajectory=reference)
    done = False
    truncated = False
    pos_errors: List[float] = []
    yaw_errors: List[float] = []
    rewards: List[float] = []

    while not (done or truncated):
        action = agent.select_action(observation, deterministic=deterministic)
        action = blend_with_reference_action(action, observation, reference_action_blend)
        observation, reward, done, truncated, info = env.step(action)
        rewards.append(float(reward))
        pos_errors.append(float(info["pos_error"]))
        yaw_errors.append(float(info["yaw_error_deg"]))

    final_info = env.last_info
    return {
        "reference_name": reference.name,
        "steps": float(len(pos_errors)),
        "terminated": float(final_info.get("terminated", 0.0)),
        "success": float(final_info.get("success", 0.0)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_pos_error": float(np.mean(pos_errors)) if pos_errors else 0.0,
        "max_pos_error": float(np.max(pos_errors)) if pos_errors else 0.0,
        "final_pos_error": float(final_info.get("pos_error", 0.0)),
        "mean_yaw_error_deg": float(np.mean(yaw_errors)) if yaw_errors else 0.0,
        "final_yaw_error_deg": float(final_info.get("yaw_error_deg", 0.0)),
        "final_along_track_error": float(final_info.get("along_track_error", 0.0)),
        "final_lateral_error": float(final_info.get("lateral_error", 0.0)),
    }


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate a controller on standard reference trajectories.")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("PDHModel/reference_trajectories"))
    parser.add_argument("--forward-model-path", type=Path, default=Path("PDHModel/forward_world_model.pth"))
    parser.add_argument("--forward-norm-path", type=Path, default=Path("PDHModel/forward_normalization.pt"))
    parser.add_argument("--backward-model-path", type=Path, default=Path("PDHModel/backward_world_model.pth"))
    parser.add_argument("--backward-norm-path", type=Path, default=Path("PDHModel/backward_normalization.pt"))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--reference-action-blend", type=float, default=0.0)
    parser.add_argument("--fixed-delta", type=float, default=0.05)
    parser.add_argument("--reset-position-noise-xy", type=float, default=0.75)
    parser.add_argument("--reset-yaw-noise-deg", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=Path("PDHModel/eval_outputs"))
    args = parser.parse_args()

    project_root = Path(find_project_root())
    device = torch.device(args.device)
    blend = float(np.clip(args.reference_action_blend, 0.0, 1.0))

    forward_bundle = load_bundle("forward", str(project_root / args.forward_model_path), str(project_root / args.forward_norm_path), device)
    backward_bundle = load_bundle("backward", str(project_root / args.backward_model_path), str(project_root / args.backward_norm_path), device)
    agent = load_policy(project_root / args.policy, device)
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

    data_root = project_root / args.data_root
    csv_paths = list_reference_csvs(data_root)
    rows: List[Dict[str, float]] = []
    for csv_path in csv_paths:
        row = evaluate_one_reference(env, agent, csv_path, args.deterministic, blend, args.fixed_delta)
        rows.append(row)
        print(
            f"{row['reference_name']}: steps={int(row['steps'])} "
            f"final_pos={row['final_pos_error']:.3f} mean_pos={row['mean_pos_error']:.3f} "
            f"final_yaw={row['final_yaw_error_deg']:.2f}"
        )

    summary = {
        "policy_path": str(project_root / args.policy),
        "reference_action_blend": blend,
        "deterministic": bool(args.deterministic),
        "num_references": len(rows),
        "mean_final_pos_error": float(np.mean([row["final_pos_error"] for row in rows])) if rows else 0.0,
        "mean_mean_pos_error": float(np.mean([row["mean_pos_error"] for row in rows])) if rows else 0.0,
        "mean_final_yaw_error_deg": float(np.mean([row["final_yaw_error_deg"] for row in rows])) if rows else 0.0,
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
        "termination_rate": float(np.mean([row["terminated"] for row in rows])) if rows else 0.0,
    }

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "evaluation_metrics.csv", rows)
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"saved_evaluation={output_dir}")


if __name__ == "__main__":
    main()
