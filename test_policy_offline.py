import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from carla_controller_PDH import find_project_root, load_bundle, model_quat_to_carla_yaw_deg, wrap_angle_deg
from deploy_rl_controller import load_policy
from dt_model_env import DTModelEnv, EnvConfig
from reference_generator import list_reference_csvs, load_reference_trajectory_for_dt


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rollout_reference(env: DTModelEnv, agent, reference_csv: Path, deterministic: bool, fixed_delta: float) -> Dict[str, object]:
    reference = load_reference_trajectory_for_dt(
        reference_csv,
        env.forward_model,
        env.device,
        control_dt=fixed_delta,
        max_length=env.env_config.max_steps + 1,
    )
    observation, reset_info = env.reset(reference_trajectory=reference)

    rows: List[Dict[str, float]] = []
    rewards: List[float] = []
    done = False
    truncated = False

    while not (done or truncated):
        action = agent.select_action(observation, deterministic=deterministic).astype(np.float32)
        observation, reward, done, truncated, info = env.step(action)
        state = env.state.copy()
        ref_state = reference.states[min(env.step_idx, len(reference.states) - 1)]
        yaw_error = float(abs(wrap_angle_deg(model_quat_to_carla_yaw_deg(state[3:7]) - model_quat_to_carla_yaw_deg(ref_state[3:7]))))
        row = {
            "step": float(info["step"]),
            "reward": float(reward),
            "loss": float(info.get("loss", 0.0)),
            "pos_error": float(info["pos_error"]),
            "yaw_error_deg": yaw_error,
            "raw_action_throttle": float(info["raw_action_throttle"]),
            "raw_action_steering": float(info["raw_action_steering"]),
            "applied_action_throttle": float(info["applied_action_throttle"]),
            "applied_action_steering": float(info["applied_action_steering"]),
            "reference_action_throttle": float(info["reference_action_throttle"]),
            "reference_action_steering": float(info["reference_action_steering"]),
            "predicted_delta_x": float(info["predicted_delta_x"]),
            "predicted_delta_y": float(info["predicted_delta_y"]),
            "predicted_delta_z": float(info["predicted_delta_z"]),
            "predicted_delta_yaw": float(info["predicted_delta_yaw"]),
            "state_x": float(state[0]),
            "state_y": float(state[1]),
            "state_z": float(state[2]),
            "ref_x": float(ref_state[0]),
            "ref_y": float(ref_state[1]),
            "ref_z": float(ref_state[2]),
            "terminated": float(done),
            "truncated": float(truncated),
        }
        rows.append(row)
        rewards.append(float(reward))

    final_info = env.last_info
    summary = {
        "reference_name": reference.name,
        "steps": int(len(rows)),
        "terminated": bool(final_info.get("terminated", 0.0)),
        "success": bool(final_info.get("success", 0.0)),
        "initial_pos_error": float(reset_info.get("initial_pos_error", 0.0)) if isinstance(reset_info, dict) else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_pos_error": float(np.mean([row["pos_error"] for row in rows])) if rows else 0.0,
        "max_pos_error": float(np.max([row["pos_error"] for row in rows])) if rows else 0.0,
        "final_pos_error": float(final_info.get("pos_error", 0.0)),
        "mean_yaw_error_deg": float(np.mean([row["yaw_error_deg"] for row in rows])) if rows else 0.0,
        "final_yaw_error_deg": float(final_info.get("yaw_error_deg", 0.0)),
        "mean_loss": float(np.mean([row["loss"] for row in rows])) if rows else 0.0,
        "max_abs_steer": float(np.max(np.abs([row["applied_action_steering"] for row in rows]))) if rows else 0.0,
        "max_throttle": float(np.max([row["applied_action_throttle"] for row in rows])) if rows else 0.0,
        "action_saturation_rate": float(
            np.mean(
                [
                    float(abs(row["applied_action_steering"]) > 0.4 or row["applied_action_throttle"] > 0.1)
                    for row in rows
                ]
            )
        )
        if rows
        else 0.0,
        "rollout_rows": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline DT+RL policy test harness without CARLA.")
    parser.add_argument("--policy", type=Path, default=Path("PDHModel/policy_controller_spec_v1.pth"))
    parser.add_argument("--data-root", type=Path, default=Path("PDHModel/reference_trajectories"))
    parser.add_argument("--forward-model-path", type=Path, default=Path("PDHModel/forward_world_model.pth"))
    parser.add_argument("--forward-norm-path", type=Path, default=Path("PDHModel/forward_normalization.pt"))
    parser.add_argument("--backward-model-path", type=Path, default=Path("PDHModel/backward_world_model.pth"))
    parser.add_argument("--backward-norm-path", type=Path, default=Path("PDHModel/backward_normalization.pt"))
    parser.add_argument("--max-steps", type=int, default=829)
    parser.add_argument("--fixed-delta", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--reset-position-noise-xy", type=float, default=0.5)
    parser.add_argument("--reset-yaw-noise-deg", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=Path("PDHModel/offline_test_outputs"))
    args = parser.parse_args()

    project_root = Path(find_project_root())
    device = torch.device(args.device)

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

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = list_reference_csvs(project_root / args.data_root)

    rows: List[Dict[str, float]] = []
    for csv_path in csv_paths:
        summary = rollout_reference(env, agent, csv_path, args.deterministic, args.fixed_delta)
        rollout_rows = summary.pop("rollout_rows")
        write_csv(output_dir / f"{csv_path.stem}_rollout.csv", rollout_rows)
        rows.append(summary)
        print(
            f"{summary['reference_name']}: steps={summary['steps']} "
            f"final_pos={summary['final_pos_error']:.3f} mean_pos={summary['mean_pos_error']:.3f} "
            f"final_yaw={summary['final_yaw_error_deg']:.2f} sat_rate={summary['action_saturation_rate']:.3f}"
        )

    write_csv(output_dir / "offline_test_metrics.csv", rows)
    aggregate = {
        "policy_path": str(project_root / args.policy),
        "num_references": len(rows),
        "mean_final_pos_error": float(np.mean([row["final_pos_error"] for row in rows])) if rows else 0.0,
        "mean_mean_pos_error": float(np.mean([row["mean_pos_error"] for row in rows])) if rows else 0.0,
        "mean_final_yaw_error_deg": float(np.mean([row["final_yaw_error_deg"] for row in rows])) if rows else 0.0,
        "mean_action_saturation_rate": float(np.mean([row["action_saturation_rate"] for row in rows])) if rows else 0.0,
        "termination_rate": float(np.mean([float(row["terminated"]) for row in rows])) if rows else 0.0,
        "success_rate": float(np.mean([float(row["success"]) for row in rows])) if rows else 0.0,
    }
    with (output_dir / "offline_test_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    print(f"saved_offline_test={output_dir}")


if __name__ == "__main__":
    main()
