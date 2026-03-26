import torch
import torch.nn as nn
import torch.optim as optim
import os
import time


# ================= 1. 定义世界模型架构 (🌟 升级大脑容量) =================
class QCarWorldModel(nn.Module):
    # 将隐藏层神经元从 128 提升到 256，层数从 2 提升到 3，大幅增强物理规律的学习能力
    def __init__(self, input_dim=9, hidden_dim=256, num_layers=3, output_dim=7):
        super(QCarWorldModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ================= 2. 数据处理与切分引擎 (🌟 剔除瞬移脏数据) =================
def create_training_data(dataset_path, seq_length=20, train_ratio=0.9):
    # 将记忆长度 seq_length 从 10 提升到了 20 帧 (0.8秒)，让它更懂惯性
    print(f"📖 正在加载数据集: {dataset_path}")
    data = torch.load(dataset_path, weights_only=True)
    actions = data['inputs']
    states = data['labels']

    num_samples = len(states) - seq_length
    x_seqs, y_seqs = [], []

    print("✂️ 正在清理物理时空断层，并提取滑动窗口...")
    dropped_frames = 0
    for i in range(num_samples):
        next_state = states[i + seq_length]
        current_state = states[i + seq_length - 1]
        delta_target = next_state - current_state

        # 🌟 核心补丁 1：消灭文件拼接处的“瞬移尖刺”！
        # 如果一帧(40ms)内 X 或 Y 坐标突然移动超过 0.5 米，绝对是不同文件拼接导致的瞬移
        if abs(delta_target[0].item()) > 0.5 or abs(delta_target[1].item()) > 0.5:
            dropped_frames += 1
            continue  # 扔掉这条脏数据！

        seq_states = states[i: i + seq_length]
        seq_actions = actions[i: i + seq_length]
        joint_input = torch.cat([seq_states, seq_actions], dim=1)

        x_seqs.append(joint_input)
        y_seqs.append(delta_target)

    print(f"🧹 成功清洗掉 {dropped_frames} 条导致尖刺的脏数据！")

    X = torch.stack(x_seqs)
    Y = torch.stack(y_seqs)

    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    # 找回没有被扔掉的对应真实状态，留给后续画图
    valid_indices = []
    for i in range(num_samples):
        dt = states[i + seq_length] - states[i + seq_length - 1]
        if abs(dt[0].item()) <= 0.5 and abs(dt[1].item()) <= 0.5:
            valid_indices.append(i)

    test_indices = valid_indices[split_idx:]
    S_current_test = torch.stack([states[idx + seq_length - 1] for idx in test_indices])
    S_next_real_test = torch.stack([states[idx + seq_length] for idx in test_indices])

    return X_train, Y_train, X_test, Y_test, S_current_test, S_next_real_test


# ================= 3. 主训练流程 (🌟 引入学习率衰减) =================
def train_model(direction="forward"):
    print(f"\n🚀 === 开始训练终极版 {direction.upper()} 世界模型 ===")
    dataset_path = f'./data_processed/qcar_{direction}_dataset.pt'

    X_train, Y_train, X_test, Y_test, S_curr_test, S_next_test = create_training_data(dataset_path)

    x_mean, x_std = X_train.mean(dim=(0, 1), keepdim=True), X_train.std(dim=(0, 1), keepdim=True) + 1e-6
    y_mean, y_std = Y_train.mean(dim=0, keepdim=True), Y_train.std(dim=0, keepdim=True) + 1e-6

    X_train_norm = (X_train - x_mean) / x_std
    Y_train_norm = (Y_train - y_mean) / y_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QCarWorldModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 🌟 核心补丁 2：学习率调度器。每过 10 轮，学习率减半，让模型在谷底进行精密微调
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()

    epochs = 50  # 训练轮数增加到 50 轮，榨干最后一滴性能
    batch_size = 256

    dataset = torch.utils.data.TensorDataset(X_train_norm, Y_train_norm)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 让学习率调度器往前走一步
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"  ▶ Epoch [{epoch + 1}/{epochs}] | 平均 Loss: {epoch_loss / len(loader):.5f} | 当前学习率: {current_lr}")

    os.makedirs('models_saved_v1', exist_ok=True)
    os.makedirs('./results_evaluation', exist_ok=True)

    torch.save(model.state_dict(), f'models_saved_v1/{direction}_world_model.pth')
    torch.save({
        'x_mean': x_mean, 'x_std': x_std,
        'y_mean': y_mean, 'y_std': y_std,
        'seq_length': 20
    }, f'models_saved_v1/{direction}_normalization.pt')

    torch.save({
        'X_test': X_test,
        'Y_test_real_delta': Y_test,
        'S_curr': S_curr_test,
        'S_next_real': S_next_test
    }, f'./results_evaluation/{direction}_test_data.pt')

    print(f"✅ {direction.upper()} 终极模型训练大功告成！")


if __name__ == "__main__":
    train_model(direction="forward")
    train_model(direction="backward")
    print("\n🎉 彻底完工！")