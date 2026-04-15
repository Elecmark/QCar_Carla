import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from carla_controller_PDH import find_project_root, load_bundle
from dt_model_env import DTModelEnv, EnvConfig
from policy_network import SACAgent, SACConfig
from reference_generator import (
    ReferenceTrajectory,
    generate_reference_trajectory,
    load_reference_trajectory_from_csv,
    save_reference_trajectory_csv,
)


def load_policy(policy_path: Path, device: torch.device) -> SACAgent:
    payload = torch.load(policy_path, map_location=device)
    config = SACConfig(**payload["config"])
    agent = SACAgent(config, device)
    if "critic" in payload:
        agent.load_state_dict(payload)
    else:
        agent.load_actor_state_dict(payload)
    return agent


def make_reference(args: argparse.Namespace) -> ReferenceTrajectory:
    if args.reference_csv is not None:
        return load_reference_trajectory_from_csv(args.reference_csv, max_length=args.length)
    return generate_reference_trajectory(args.ref_type, args.length, params={"dt": args.dt, "radius": args.radius, "amplitude": args.amplitude})


def write_rollout_csv(output_path: Path, predicted_states: np.ndarray, reference_states: np.ndarray, actions: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "pred_pos_x",
                "pred_pos_y",
                "pred_pos_z",
                "pred_rot_0",
                "pred_rot_1",
                "pred_rot_2",
                "pred_rot_3",
                "ref_pos_x",
                "ref_pos_y",
                "ref_pos_z",
                "ref_rot_0",
                "ref_rot_1",
                "ref_rot_2",
                "ref_rot_3",
                "action_throttle",
                "action_steering",
            ]
        )
        horizon = min(len(predicted_states), len(reference_states), len(actions))
        for idx in range(horizon):
            writer.writerow([idx, *predicted_states[idx].tolist(), *reference_states[idx].tolist(), *actions[idx].tolist()])


def blend_with_reference_action(policy_action: np.ndarray, observation: np.ndarray, blend: float) -> np.ndarray:
    if blend <= 0.0 or observation.shape[0] < 32:
        return np.clip(policy_action, -1.0, 1.0).astype(np.float32)

    ref_action = observation[30:32].astype(np.float32, copy=True)
    mixed = (1.0 - blend) * np.asarray(policy_action, dtype=np.float32) + blend * ref_action
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy a trained SAC controller on the DT world model.")
    parser.add_argument("--policy", type=Path, default=Path("PDHModel/policy_controller.pth"))
    parser.add_argument("--policy-config", type=Path, default=Path("PDHModel/policy_config.json"))
    parser.add_argument("--forward-model-path", type=Path, default=Path("PDHModel/forward_world_model.pth"))
    parser.add_argument("--forward-norm-path", type=Path, default=Path("PDHModel/forward_normalization.pt"))
    parser.add_argument("--backward-model-path", type=Path, default=Path("PDHModel/backward_world_model.pth"))
    parser.add_argument("--backward-norm-path", type=Path, default=Path("PDHModel/backward_normalization.pt"))
    parser.add_argument("--ref-type", default="sine", choices=["straight", "circle", "figure8", "sine", "s_curve"])
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--amplitude", type=float, default=2.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("PDHModel/deploy_outputs"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reference-action-blend", type=float, default=0.25)
    args = parser.parse_args()

    project_root = Path(find_project_root())
    device = torch.device(args.device)

    forward_bundle = load_bundle("forward", str(project_root / args.forward_model_path), str(project_root / args.forward_norm_path), device)
    backward_bundle = load_bundle("backward", str(project_root / args.backward_model_path), str(project_root / args.backward_norm_path), device)
    agent = load_policy(project_root / args.policy, device)
    reference = make_reference(args)

    env = DTModelEnv(forward_bundle, backward_bundle, device=device, env_config=EnvConfig(max_steps=args.length))
    observation, info = env.reset(reference_trajectory=reference)
    del info
    rollout_actions = []

    done = False
    truncated = False
    while not (done or truncated):
        action = agent.select_action(observation, deterministic=args.deterministic)
        action = blend_with_reference_action(action, observation, float(np.clip(args.reference_action_blend, 0.0, 1.0)))
        observation, reward, done, truncated, step_info = env.step(action)
        rollout_actions.append(np.asarray(action, dtype=np.float32))
        print(
            f"step={step_info['step']:04d} reward={reward:+.4f} "
            f"pos_error={step_info['pos_error']:.4f} yaw_error={step_info['yaw_error_deg']:.2f} bundle={step_info['bundle_name']}"
        )

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_actions_np = np.asarray(rollout_actions, dtype=np.float32)
    predicted_states = np.asarray(env.predicted_states, dtype=np.float32)
    write_rollout_csv(output_dir / "deploy_rollout.csv", predicted_states, reference.states[1 : 1 + len(predicted_states)], rollout_actions_np)
    save_reference_trajectory_csv(reference, output_dir / "deploy_reference.csv")

    summary: Dict[str, object] = {
        "reference_name": reference.name,
        "steps": int(len(predicted_states)),
        "final_info": env.last_info,
        "policy_path": str(project_root / args.policy),
        "deterministic": bool(args.deterministic),
        "reference_action_blend": float(np.clip(args.reference_action_blend, 0.0, 1.0)),
    }
    with (output_dir / "deploy_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved deployment outputs to {output_dir}")


if __name__ == "__main__":
    main()
