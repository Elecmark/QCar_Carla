import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clean_world_model.common import (
    ACTION_NAMES,
    FEATURE_NAMES,
    STATE_NAMES,
    CleanWorldModel,
    build_windows,
    ensure_dir,
    project_root,
    split_by_episode,
)


def load_trajectories(dataset_path: str):
    data = torch.load(dataset_path, map_location="cpu")
    return data["trajectories"]


def train_one_direction(
    direction: str,
    dataset_path: str,
    model_dir: str,
    eval_dir: str,
    seq_length: int,
    train_ratio: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    trajectories = load_trajectories(dataset_path)
    train_episodes, test_episodes = split_by_episode(trajectories, train_ratio=train_ratio)

    x_train, y_train, _ = build_windows(train_episodes, seq_length=seq_length)
    x_test, y_test, meta_test = build_windows(test_episodes, seq_length=seq_length)

    x_mean = x_train.mean(dim=(0, 1), keepdim=True)
    x_std = x_train.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    train_dataset = torch.utils.data.TensorDataset((x_train - x_mean) / x_std, (y_train - y_mean) / y_std)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CleanWorldModel(input_dim=len(FEATURE_NAMES), output_dim=len(STATE_NAMES)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss()

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item()) * batch_x.shape[0]

        scheduler.step()
        epoch_loss = total_loss / len(train_dataset)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"[{direction}] epoch={epoch + 1:03d}/{epochs} train_loss={epoch_loss:.6f} lr={scheduler.get_last_lr()[0]:.6f}")

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        x_test_norm = ((x_test - x_mean) / x_std).to(device)
        y_pred_norm = model(x_test_norm).cpu()
        y_pred = y_pred_norm * y_std + y_mean

    ensure_dir(model_dir)
    ensure_dir(eval_dir)

    model_path = os.path.join(model_dir, f"{direction}_clean_world_model.pth")
    norm_path = os.path.join(model_dir, f"{direction}_clean_normalization.pt")
    test_path = os.path.join(eval_dir, f"{direction}_clean_test_data.pt")

    torch.save(model.state_dict(), model_path)
    torch.save(
        {
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "seq_length": seq_length,
            "state_names": STATE_NAMES,
            "action_names": ACTION_NAMES,
        },
        norm_path,
    )
    torch.save(
        {
            "X_test": x_test,
            "Y_test": y_test,
            "Y_pred": y_pred,
            "meta_test": meta_test,
        },
        test_path,
    )
    print(f"[{direction}] saved model -> {model_path}")


def main() -> None:
    root = project_root()

    parser = argparse.ArgumentParser(description="Train clean QCar world models.")
    parser.add_argument("--data-dir", default=os.path.join(root, "clean_world_model_artifacts", "data_processed"))
    parser.add_argument("--model-dir", default=os.path.join(root, "clean_world_model_artifacts", "models_saved"))
    parser.add_argument("--eval-dir", default=os.path.join(root, "clean_world_model_artifacts", "results_evaluation"))
    parser.add_argument("--seq-length", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir)
    args.model_dir = os.path.abspath(args.model_dir)
    args.eval_dir = os.path.abspath(args.eval_dir)

    for direction in ("forward", "backward"):
        dataset_path = os.path.join(args.data_dir, f"qcar_{direction}_clean_dataset.pt")
        train_one_direction(
            direction=direction,
            dataset_path=dataset_path,
            model_dir=args.model_dir,
            eval_dir=args.eval_dir,
            seq_length=args.seq_length,
            train_ratio=args.train_ratio,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )


if __name__ == "__main__":
    main()
