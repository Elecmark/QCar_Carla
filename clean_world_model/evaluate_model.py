import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clean_world_model.common import STATE_NAMES, ensure_dir, project_root


def evaluate_one_direction(direction: str, eval_dir: str, output_dir: str) -> None:
    data = torch.load(os.path.join(eval_dir, f"{direction}_clean_test_data.pt"), map_location="cpu")
    y_true = data["Y_test"]
    y_pred = data["Y_pred"]

    ensure_dir(output_dir)
    direction_dir = os.path.join(output_dir, f"{direction}_clean_evaluation")
    ensure_dir(direction_dir)

    metrics = []
    for idx, feature_name in enumerate(STATE_NAMES):
        true_values = y_true[:, idx].numpy()
        pred_values = y_pred[:, idx].numpy()
        errors = true_values - pred_values

        rmse = float((torch.tensor(errors).pow(2).mean().sqrt()).item())
        mae = float(torch.tensor(errors).abs().mean().item())
        metrics.append({"feature": feature_name, "rmse": rmse, "mae": mae})

        df = pd.DataFrame(
            {
                "time_step": range(len(true_values)),
                f"true_{feature_name}": true_values,
                f"pred_{feature_name}": pred_values,
                "error": errors,
            }
        )
        df.to_csv(os.path.join(direction_dir, f"{direction}_{feature_name}_clean_error.csv"), index=False)

        plt.figure(figsize=(10, 5))
        limit = min(400, len(true_values))
        plt.plot(true_values[:limit], label="true", linewidth=2)
        plt.plot(pred_values[:limit], label="pred", linewidth=2, linestyle="--")
        plt.title(f"{direction.upper()} {feature_name}")
        plt.xlabel("time step")
        plt.ylabel(feature_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(direction_dir, f"{direction}_{feature_name}_clean_plot.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    pd.DataFrame(metrics).to_csv(os.path.join(direction_dir, f"{direction}_clean_metrics.csv"), index=False)
    print(f"[{direction}] evaluation -> {direction_dir}")


def main() -> None:
    root = project_root()

    parser = argparse.ArgumentParser(description="Evaluate clean QCar world models.")
    parser.add_argument("--eval-dir", default=os.path.join(root, "clean_world_model_artifacts", "results_evaluation"))
    parser.add_argument("--output-dir", default=os.path.join(root, "clean_world_model_artifacts", "results_evaluation"))
    args = parser.parse_args()

    args.eval_dir = os.path.abspath(args.eval_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    for direction in ("forward", "backward"):
        evaluate_one_direction(direction=direction, eval_dir=args.eval_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
