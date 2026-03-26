import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from PDH_train_world_model import (
    INPUT_COLUMNS,
    STATE_COLUMNS,
    create_load_stats,
    list_candidate_dirs,
    read_filtered_rows,
)
from carla_controller_PDH import (
    Normalizer,
    QCarWorldModel,
    align_quaternion_raw,
    canonicalize_position_history,
    predicted_output_to_next_state,
    raw_quaternion_to_xyzw,
)


STATE_NAMES = STATE_COLUMNS


def split_index_for_windows(num_windows: int, train_ratio: float) -> int:
    split_index = int(num_windows * train_ratio)
    if num_windows >= 2:
        split_index = max(1, min(split_index, num_windows - 1))
    return split_index


def build_test_windows(data_root: Path, direction: str, seq_length: int, train_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    x_parts: List[np.ndarray] = []
    next_parts: List[np.ndarray] = []
    stats = create_load_stats()

    for directory in list_candidate_dirs(data_root, direction):
        for csv_path in sorted(directory.glob("*.csv")):
            rows = read_filtered_rows(csv_path, direction, stats)
            if len(rows) <= seq_length:
                continue

            windows: List[List[List[float]]] = []
            next_states: List[List[float]] = []
            for start in range(len(rows) - seq_length):
                end = start + seq_length
                sequence = rows[start:end]
                next_state = [rows[end][column] for column in STATE_COLUMNS]
                window = [[frame[column] for column in INPUT_COLUMNS] for frame in sequence]
                windows.append(window)
                next_states.append(next_state)

            if not windows:
                continue

            windows_np = np.asarray(windows, dtype=np.float32)
            next_states_np = np.asarray(next_states, dtype=np.float32)
            split_index = split_index_for_windows(len(windows_np), train_ratio)
            if split_index < len(windows_np):
                x_parts.append(windows_np[split_index:])
                next_parts.append(next_states_np[split_index:])

    if not x_parts or not next_parts:
        raise RuntimeError(f"No test windows built for {direction}")

    return np.concatenate(x_parts, axis=0), np.concatenate(next_parts, axis=0)


def load_model_bundle(direction: str, model_dir: Path, device: torch.device) -> Tuple[QCarWorldModel, Normalizer]:
    model_path = model_dir / f"{direction}_world_model.pth"
    norm_path = model_dir / f"{direction}_normalization.pt"
    normalizer = Normalizer.from_file(str(norm_path))
    model = QCarWorldModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, normalizer


def predict_outputs(model: QCarWorldModel, normalizer: Normalizer, x_raw: np.ndarray, device: torch.device) -> np.ndarray:
    x_canonical = canonicalize_position_history(x_raw)
    x_tensor = torch.from_numpy(x_canonical).float().to(device)
    with torch.no_grad():
        x_norm = normalizer.norm_x(x_tensor)
        y_norm = model(x_norm)
        y_pred = normalizer.denorm_y(y_norm)
    return y_pred.cpu().numpy().astype(np.float32)


def predict_next_states(x_raw: np.ndarray, pred_outputs: np.ndarray) -> np.ndarray:
    current_states = x_raw[:, -1, :7]
    predictions = [
        predicted_output_to_next_state(current_states[i], pred_outputs[i]).astype(np.float32)
        for i in range(len(current_states))
    ]
    return np.asarray(predictions, dtype=np.float32)


def align_predicted_quaternions(true_next: np.ndarray, pred_next: np.ndarray) -> np.ndarray:
    aligned = pred_next.astype(np.float32, copy=True)
    for index in range(len(aligned)):
        aligned[index, 3:7] = align_quaternion_raw(true_next[index, 3:7], aligned[index, 3:7])
    return aligned


def save_comparison_csv(path: Path, true_next: np.ndarray, pred_next: np.ndarray) -> None:
    rows: List[Dict[str, float]] = []
    for index in range(len(true_next)):
        row: Dict[str, float] = {"step": index}
        for dim, name in enumerate(STATE_NAMES):
            row[f"true_{name}"] = float(true_next[index, dim])
            row[f"pred_{name}"] = float(pred_next[index, dim])
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def quaternion_raw_to_yaw_deg(quat_raw: np.ndarray) -> float:
    qx, qy, qz, qw = raw_quaternion_to_xyzw(quat_raw).astype(np.float64, copy=False)
    norm = float(np.linalg.norm([qx, qy, qz, qw]))
    if norm < 1e-8:
        return 0.0
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.degrees(np.arctan2(siny_cosp, cosy_cosp)))


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _polyline_points(values: np.ndarray, x0: float, y0: float, width: float, height: float, vmin: float, vmax: float) -> str:
    if len(values) == 1:
        xs = np.array([x0 + width / 2.0], dtype=np.float32)
    else:
        xs = np.linspace(x0, x0 + width, len(values), dtype=np.float32)
    denom = max(vmax - vmin, 1e-6)
    ys = y0 + height - ((values - vmin) / denom) * height
    return " ".join(f"{float(x):.2f},{float(y):.2f}" for x, y in zip(xs, ys))


def save_combined_plot(path: Path, true_next: np.ndarray, pred_next: np.ndarray, title: str, plot_limit: int) -> None:
    limit = min(plot_limit, len(true_next))
    true_next = true_next[:limit]
    pred_next = pred_next[:limit]

    pos_block = np.concatenate([true_next[:, :3], pred_next[:, :3]], axis=1)
    rot_block = np.concatenate([true_next[:, 3:], pred_next[:, 3:]], axis=1)
    pos_vmin = float(np.min(pos_block))
    pos_vmax = float(np.max(pos_block))
    rot_vmin = float(np.min(rot_block))
    rot_vmax = float(np.max(rot_block))
    if abs(pos_vmax - pos_vmin) < 1e-6:
        pos_vmax = pos_vmin + 1.0
    if abs(rot_vmax - rot_vmin) < 1e-6:
        rot_vmax = rot_vmin + 1.0

    canvas_w = 1800
    canvas_h = 1400
    margin_x = 60
    margin_y = 90
    cols = 2
    rows = 4
    cell_w = (canvas_w - margin_x * 2) / cols
    cell_h = (canvas_h - margin_y * 2) / rows
    plot_w = cell_w - 70
    plot_h = cell_h - 80

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{canvas_w/2:.0f}" y="42" text-anchor="middle" font-size="26" font-family="Arial">{_svg_escape(title)}</text>',
        '<line x1="80" y1="62" x2="140" y2="62" stroke="#1f77b4" stroke-width="3"/>',
        '<text x="150" y="68" font-size="18" font-family="Arial">True</text>',
        '<line x1="240" y1="62" x2="300" y2="62" stroke="#ff7f0e" stroke-width="3" stroke-dasharray="8 6"/>',
        '<text x="310" y="68" font-size="18" font-family="Arial">Pred</text>',
    ]

    for dim, name in enumerate(STATE_NAMES):
        row = dim // cols
        col = dim % cols
        panel_x = margin_x + col * cell_w
        panel_y = margin_y + row * cell_h
        x0 = panel_x + 50
        y0 = panel_y + 35

        true_vals = true_next[:, dim]
        pred_vals = pred_next[:, dim]
        if dim < 3:
            vmin = pos_vmin
            vmax = pos_vmax
        else:
            vmin = rot_vmin
            vmax = rot_vmax

        parts.append(f'<rect x="{panel_x:.2f}" y="{panel_y:.2f}" width="{cell_w-20:.2f}" height="{cell_h-20:.2f}" fill="none" stroke="#cccccc"/>')
        parts.append(f'<text x="{panel_x + (cell_w-20)/2:.2f}" y="{panel_y + 22:.2f}" text-anchor="middle" font-size="18" font-family="Arial">{_svg_escape(name)}</text>')
        parts.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{plot_w:.2f}" height="{plot_h:.2f}" fill="none" stroke="#999999"/>')
        for frac in [0.25, 0.5, 0.75]:
            gy = y0 + plot_h * frac
            parts.append(f'<line x1="{x0:.2f}" y1="{gy:.2f}" x2="{x0 + plot_w:.2f}" y2="{gy:.2f}" stroke="#eeeeee"/>')

        true_points = _polyline_points(true_vals, x0, y0, plot_w, plot_h, vmin, vmax)
        pred_points = _polyline_points(pred_vals, x0, y0, plot_w, plot_h, vmin, vmax)
        parts.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{true_points}"/>')
        parts.append(f'<polyline fill="none" stroke="#ff7f0e" stroke-width="2" stroke-dasharray="6 4" points="{pred_points}"/>')
        parts.append(f'<text x="{x0:.2f}" y="{y0 + plot_h + 22:.2f}" font-size="12" font-family="Arial">0</text>')
        parts.append(f'<text x="{x0 + plot_w - 40:.2f}" y="{y0 + plot_h + 22:.2f}" font-size="12" font-family="Arial">{limit - 1}</text>')
        parts.append(f'<text x="{x0 - 6:.2f}" y="{y0 + 12:.2f}" text-anchor="end" font-size="12" font-family="Arial">{vmax:.3f}</text>')
        parts.append(f'<text x="{x0 - 6:.2f}" y="{y0 + plot_h:.2f}" text-anchor="end" font-size="12" font-family="Arial">{vmin:.3f}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_yaw_plot(path: Path, true_next: np.ndarray, pred_next: np.ndarray, title: str, plot_limit: int) -> None:
    limit = min(plot_limit, len(true_next))
    true_yaw = np.asarray([quaternion_raw_to_yaw_deg(q) for q in true_next[:limit, 3:7]], dtype=np.float32)
    pred_yaw = np.asarray([quaternion_raw_to_yaw_deg(q) for q in pred_next[:limit, 3:7]], dtype=np.float32)
    vmin = float(min(np.min(true_yaw), np.min(pred_yaw)))
    vmax = float(max(np.max(true_yaw), np.max(pred_yaw)))
    if abs(vmax - vmin) < 1e-6:
        vmax = vmin + 1.0

    canvas_w = 1800
    canvas_h = 520
    x0 = 90.0
    y0 = 90.0
    plot_w = 1620.0
    plot_h = 340.0
    true_points = _polyline_points(true_yaw, x0, y0, plot_w, plot_h, vmin, vmax)
    pred_points = _polyline_points(pred_yaw, x0, y0, plot_w, plot_h, vmin, vmax)

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{canvas_w/2:.0f}" y="42" text-anchor="middle" font-size="26" font-family="Arial">{_svg_escape(title)}</text>',
        '<line x1="90" y1="62" x2="150" y2="62" stroke="#1f77b4" stroke-width="3"/>',
        '<text x="160" y="68" font-size="18" font-family="Arial">True Yaw</text>',
        '<line x1="320" y1="62" x2="380" y2="62" stroke="#ff7f0e" stroke-width="3" stroke-dasharray="8 6"/>',
        '<text x="390" y="68" font-size="18" font-family="Arial">Pred Yaw</text>',
        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{plot_w:.2f}" height="{plot_h:.2f}" fill="none" stroke="#999999"/>',
    ]
    for frac in [0.25, 0.5, 0.75]:
        gy = y0 + plot_h * frac
        parts.append(f'<line x1="{x0:.2f}" y1="{gy:.2f}" x2="{x0 + plot_w:.2f}" y2="{gy:.2f}" stroke="#eeeeee"/>')
    parts.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{true_points}"/>')
    parts.append(f'<polyline fill="none" stroke="#ff7f0e" stroke-width="2" stroke-dasharray="6 4" points="{pred_points}"/>')
    parts.append(f'<text x="{x0:.2f}" y="{y0 + plot_h + 26:.2f}" font-size="12" font-family="Arial">0</text>')
    parts.append(f'<text x="{x0 + plot_w - 40:.2f}" y="{y0 + plot_h + 26:.2f}" font-size="12" font-family="Arial">{limit - 1}</text>')
    parts.append(f'<text x="{x0 - 8:.2f}" y="{y0 + 12:.2f}" text-anchor="end" font-size="12" font-family="Arial">{vmax:.3f}</text>')
    parts.append(f'<text x="{x0 - 8:.2f}" y="{y0 + plot_h:.2f}" text-anchor="end" font-size="12" font-family="Arial">{vmin:.3f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


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
    x_test, s_next_true = build_test_windows(data_root, direction, normalizer.seq_length, train_ratio)
    pred_outputs = predict_outputs(model, normalizer, x_test, device)
    s_next_pred = predict_next_states(x_test, pred_outputs)
    s_next_pred_aligned = align_predicted_quaternions(s_next_true, s_next_pred)

    direction_dir = output_root / f"{direction}_evaluation"
    direction_dir.mkdir(parents=True, exist_ok=True)

    csv_path = direction_dir / f"{direction}_comparison.csv"
    plot_path = direction_dir / f"{direction}_combined_7d_plot.svg"
    yaw_plot_path = direction_dir / f"{direction}_yaw_plot.svg"
    save_comparison_csv(csv_path, s_next_true, s_next_pred_aligned)
    save_combined_plot(
        plot_path,
        s_next_true,
        s_next_pred_aligned,
        title=f"PDH {direction.capitalize()} Test Set: True vs Predicted Next State",
        plot_limit=plot_limit,
    )
    save_yaw_plot(
        yaw_plot_path,
        s_next_true,
        s_next_pred_aligned,
        title=f"PDH {direction.capitalize()} Test Set: True vs Predicted Yaw",
        plot_limit=plot_limit,
    )

    mae = np.mean(np.abs(s_next_true - s_next_pred_aligned), axis=0)
    rmse = np.sqrt(np.mean((s_next_true - s_next_pred_aligned) ** 2, axis=0))
    return {
        "direction": direction,
        "num_test_samples": int(len(x_test)),
        "seq_length": int(normalizer.seq_length),
        "comparison_csv": str(csv_path),
        "combined_plot": str(plot_path),
        "yaw_plot": str(yaw_plot_path),
        "mae_by_dim": {name: float(mae[i]) for i, name in enumerate(STATE_NAMES)},
        "rmse_by_dim": {name: float(rmse[i]) for i, name in enumerate(STATE_NAMES)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PDH model 7D comparison plots on rebuilt test sets.")
    parser.add_argument("--data-root", type=Path, default=Path("QCarDataSet"))
    parser.add_argument("--model-dir", type=Path, default=Path("PDHModel"))
    parser.add_argument("--output-dir", type=Path, default=Path("PDH_test_plots"))
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

    summary_path = args.output_dir / "pdh_evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved PDH evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
