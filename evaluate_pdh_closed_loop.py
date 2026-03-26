import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from PDH_train_world_model import (
    ACTION_COLUMNS,
    INPUT_COLUMNS,
    STATE_COLUMNS,
    create_load_stats,
    list_candidate_dirs,
    read_filtered_rows,
)
from evaluate_pdh_plots import (
    STATE_NAMES,
    align_predicted_quaternions,
    load_model_bundle,
    save_combined_plot,
    save_comparison_csv,
    save_yaw_plot,
    split_index_for_windows,
)
from carla_controller_PDH import canonicalize_position_history, predicted_output_to_next_state


@dataclass
class ClosedLoopEpisode:
    source: str
    init_history: np.ndarray
    actions: np.ndarray
    true_next_states: np.ndarray


def build_closed_loop_episodes(
    data_root: Path,
    direction: str,
    seq_length: int,
    train_ratio: float,
) -> List[ClosedLoopEpisode]:
    episodes: List[ClosedLoopEpisode] = []
    stats = create_load_stats()

    for directory in list_candidate_dirs(data_root, direction):
        for csv_path in sorted(directory.glob("*.csv")):
            rows = read_filtered_rows(csv_path, direction, stats)
            if len(rows) <= seq_length:
                continue

            num_windows = len(rows) - seq_length
            split_index = split_index_for_windows(num_windows, train_ratio)
            if split_index >= num_windows:
                continue

            init_start = split_index
            init_end = init_start + seq_length
            if init_end >= len(rows):
                continue

            init_history = np.asarray(
                [[rows[idx][column] for column in INPUT_COLUMNS] for idx in range(init_start, init_end)],
                dtype=np.float32,
            )

            actions: List[List[float]] = []
            true_next_states: List[List[float]] = []
            for current_idx in range(init_end - 1, len(rows) - 1):
                actions.append([rows[current_idx][column] for column in ACTION_COLUMNS])
                true_next_states.append([rows[current_idx + 1][column] for column in STATE_COLUMNS])

            if not actions or not true_next_states:
                continue

            episodes.append(
                ClosedLoopEpisode(
                    source=str(csv_path),
                    init_history=init_history,
                    actions=np.asarray(actions, dtype=np.float32),
                    true_next_states=np.asarray(true_next_states, dtype=np.float32),
                )
            )

    if not episodes:
        raise RuntimeError(f"No closed-loop test episodes built for {direction}")
    return episodes


def rollout_closed_loop_episode(
    model: torch.nn.Module,
    normalizer,
    episode: ClosedLoopEpisode,
    device: torch.device,
) -> np.ndarray:
    history: deque[np.ndarray] = deque(
        [frame.astype(np.float32, copy=True) for frame in episode.init_history],
        maxlen=normalizer.seq_length,
    )
    model_state = episode.init_history[-1, :7].astype(np.float32, copy=True)
    predictions: List[np.ndarray] = []

    for step_idx in range(len(episode.actions)):
        history_np = np.stack(list(history), axis=0).astype(np.float32)
        x_canonical = canonicalize_position_history(history_np)
        x_tensor = torch.from_numpy(x_canonical).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = model(normalizer.norm_x(x_tensor))
            pred_output = normalizer.denorm_y(pred_norm).squeeze(0).cpu().numpy().astype(np.float32)

        model_state = predicted_output_to_next_state(model_state, pred_output)
        predictions.append(model_state.copy())

        next_action = episode.actions[step_idx + 1] if step_idx + 1 < len(episode.actions) else episode.actions[step_idx]
        history.append(np.concatenate([model_state, next_action], axis=0).astype(np.float32))

    return np.asarray(predictions, dtype=np.float32)


def evaluate_direction(
    direction: str,
    data_root: Path,
    model_dir: Path,
    output_root: Path,
    train_ratio: float,
    plot_limit: int,
    device: torch.device,
) -> Dict[str, object]:
    model, normalizer = load_model_bundle(direction, model_dir, device)
    episodes = build_closed_loop_episodes(data_root, direction, normalizer.seq_length, train_ratio)

    true_parts: List[np.ndarray] = []
    pred_parts: List[np.ndarray] = []
    for episode in episodes:
        pred_states = rollout_closed_loop_episode(model, normalizer, episode, device)
        pred_states = align_predicted_quaternions(episode.true_next_states, pred_states)
        true_parts.append(episode.true_next_states)
        pred_parts.append(pred_states)

    true_all = np.concatenate(true_parts, axis=0)
    pred_all = np.concatenate(pred_parts, axis=0)

    direction_dir = output_root / f"{direction}_closed_loop_evaluation"
    direction_dir.mkdir(parents=True, exist_ok=True)

    csv_path = direction_dir / f"{direction}_closed_loop_comparison.csv"
    plot_path = direction_dir / f"{direction}_closed_loop_combined_7d_plot.svg"
    yaw_plot_path = direction_dir / f"{direction}_closed_loop_yaw_plot.svg"

    save_comparison_csv(csv_path, true_all, pred_all)
    save_combined_plot(
        plot_path,
        true_all,
        pred_all,
        title=f"PDH {direction.capitalize()} Closed-Loop Test: True vs Predicted State",
        plot_limit=plot_limit,
    )
    save_yaw_plot(
        yaw_plot_path,
        true_all,
        pred_all,
        title=f"PDH {direction.capitalize()} Closed-Loop Test: True vs Predicted Yaw",
        plot_limit=plot_limit,
    )

    mae = np.mean(np.abs(true_all - pred_all), axis=0)
    rmse = np.sqrt(np.mean((true_all - pred_all) ** 2, axis=0))
    return {
        "direction": direction,
        "num_closed_loop_episodes": int(len(episodes)),
        "num_rollout_steps": int(len(true_all)),
        "seq_length": int(normalizer.seq_length),
        "comparison_csv": str(csv_path),
        "combined_plot": str(plot_path),
        "yaw_plot": str(yaw_plot_path),
        "mae_by_dim": {name: float(mae[i]) for i, name in enumerate(STATE_NAMES)},
        "rmse_by_dim": {name: float(rmse[i]) for i, name in enumerate(STATE_NAMES)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PDH closed-loop rollout evaluation plots.")
    parser.add_argument("--data-root", type=Path, default=Path("QCarDataSet"))
    parser.add_argument("--model-dir", type=Path, default=Path("PDHModel"))
    parser.add_argument("--output-dir", type=Path, default=Path("PDH_test_plots_closed_loop"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--plot-limit", type=int, default=500)
    parser.add_argument("--directions", nargs="+", choices=["forward", "backward"], default=["forward", "backward"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "model_dir": str(args.model_dir),
        "data_root": str(args.data_root),
        "plot_limit": int(args.plot_limit),
        "evaluation_type": "closed_loop",
        "directions": {},
    }

    for direction in args.directions:
        summary["directions"][direction] = evaluate_direction(
            direction=direction,
            data_root=args.data_root,
            model_dir=args.model_dir,
            output_root=args.output_dir,
            train_ratio=args.train_ratio,
            plot_limit=args.plot_limit,
            device=device,
        )

    summary_path = args.output_dir / "pdh_closed_loop_evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved PDH closed-loop evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
