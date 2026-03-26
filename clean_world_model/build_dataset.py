import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clean_world_model.common import ACTION_NAMES, STATE_NAMES, ensure_dir, project_root, state_from_row_dict


RAW_REQUIRED_COLUMNS = [
    "time",
    "throttle",
    "steering",
    "pos_x",
    "pos_y",
    "pos_z",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
]


def classify_direction(path: str) -> str:
    folder = os.path.basename(os.path.dirname(path)).lower()
    if folder.endswith("_forward"):
        return "forward"
    if folder.endswith("_backward"):
        return "backward"
    return "skip"


def read_csv_rows(path: str) -> List[Dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        missing = [col for col in RAW_REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"missing columns {missing}")

        rows = []
        for raw in reader:
            try:
                row = {col: float(raw[col]) for col in RAW_REQUIRED_COLUMNS}
            except (KeyError, ValueError):
                continue
            rows.append(row)
    return rows


def split_contiguous_segments(rows: List[Dict[str, float]], gap_factor: float) -> List[List[Dict[str, float]]]:
    if len(rows) < 2:
        return [rows] if rows else []

    times = np.array([row["time"] for row in rows], dtype=np.float64)
    positive_dt = np.diff(times)
    positive_dt = positive_dt[positive_dt > 1e-6]
    if positive_dt.size == 0:
        return []

    median_dt = float(np.median(positive_dt))
    if median_dt <= 0.0:
        return []

    segments: List[List[Dict[str, float]]] = []
    start = 0
    for idx in range(len(rows) - 1):
        dt = rows[idx + 1]["time"] - rows[idx]["time"]
        if dt <= 1e-6 or dt > gap_factor * median_dt:
            segment = rows[start : idx + 1]
            if segment:
                segments.append(segment)
            start = idx + 1
    final_segment = rows[start:]
    if final_segment:
        segments.append(final_segment)
    return segments


def segment_to_tensors(
    segment: List[Dict[str, float]],
    max_abs_steering: float,
    max_abs_throttle: float,
    min_segment_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    states = []
    actions = []

    for row in segment:
        throttle = float(row["throttle"])
        steering = float(row["steering"])
        if abs(steering) > max_abs_steering or abs(throttle) > max_abs_throttle:
            continue
        actions.append(np.array([throttle, steering], dtype=np.float32))
        states.append(state_from_row_dict(row))

    if len(states) < min_segment_length:
        return torch.empty(0, 7), torch.empty(0, 2)

    return torch.tensor(np.stack(states), dtype=torch.float32), torch.tensor(np.stack(actions), dtype=torch.float32)


def build_direction_dataset(
    csv_paths: List[str],
    max_abs_steering: float,
    max_abs_throttle: float,
    min_segment_length: int,
    gap_factor: float,
) -> Dict[str, object]:
    trajectories = []
    skipped_files = 0
    total_segments = 0

    for path in csv_paths:
        try:
            rows = read_csv_rows(path)
        except Exception:
            skipped_files += 1
            continue

        rows.sort(key=lambda item: item["time"])
        segments = split_contiguous_segments(rows, gap_factor=gap_factor)
        for segment_id, segment in enumerate(segments):
            states, actions = segment_to_tensors(
                segment=segment,
                max_abs_steering=max_abs_steering,
                max_abs_throttle=max_abs_throttle,
                min_segment_length=min_segment_length,
            )
            if states.numel() == 0:
                continue
            total_segments += 1
            trajectories.append(
                {
                    "states": states,
                    "actions": actions,
                    "source_csv": path,
                    "segment_id": segment_id,
                    "length": int(states.shape[0]),
                }
            )

    if not trajectories:
        raise RuntimeError("No valid trajectories were extracted.")

    return {
        "trajectories": trajectories,
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "skipped_files": skipped_files,
        "num_segments": total_segments,
    }


def main() -> None:
    root = project_root()

    parser = argparse.ArgumentParser(description="Build clean QCar world-model datasets from raw CSV files.")
    parser.add_argument("--raw-dir", default=os.path.join(root, "QCarDataSet"))
    parser.add_argument("--output-dir", default=os.path.join(root, "clean_world_model_artifacts", "data_processed"))
    parser.add_argument("--max-abs-steering", type=float, default=1.0)
    parser.add_argument("--max-abs-throttle", type=float, default=0.3)
    parser.add_argument("--min-segment-length", type=int, default=25)
    parser.add_argument("--gap-factor", type=float, default=3.0)
    args = parser.parse_args()

    args.raw_dir = os.path.abspath(args.raw_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    ensure_dir(args.output_dir)
    all_csv = sorted(glob.glob(os.path.join(args.raw_dir, "*", "*.csv")))
    forward_csv = [path for path in all_csv if classify_direction(path) == "forward"]
    backward_csv = [path for path in all_csv if classify_direction(path) == "backward"]

    for direction, csv_paths in [("forward", forward_csv), ("backward", backward_csv)]:
        dataset = build_direction_dataset(
            csv_paths=csv_paths,
            max_abs_steering=args.max_abs_steering,
            max_abs_throttle=args.max_abs_throttle,
            min_segment_length=args.min_segment_length,
            gap_factor=args.gap_factor,
        )
        output_path = os.path.join(args.output_dir, f"qcar_{direction}_clean_dataset.pt")
        torch.save(dataset, output_path)
        num_steps = sum(int(item["length"]) for item in dataset["trajectories"])
        print(
            f"[{direction}] trajectories={len(dataset['trajectories'])}, "
            f"steps={num_steps}, skipped_files={dataset['skipped_files']} -> {output_path}"
        )


if __name__ == "__main__":
    main()
