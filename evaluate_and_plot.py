import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import os


# ================= 1. 定义模型架构 (需与训练时完全一致) =================
class QCarWorldModel(nn.Module):
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


# ================= 2. 评估与可视化生成引擎 =================
def evaluate_and_plot(direction="forward"):
    print(f"\n📊 === 开始生成 {direction.upper()} 模型的评估报告 ===")

    # 路径配置
    model_path = f'./models_saved/{direction}_world_model.pth'
    norm_path = f'./models_saved/{direction}_normalization.pt'
    test_data_path = f'./results_evaluation/{direction}_test_data.pt'
    output_dir = f'./results_evaluation/{direction}_evaluation'
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path) or not os.path.exists(test_data_path):
        print(f"⚠️ 找不到 {direction} 的模型或测试数据，跳过评估。")
        return

    # 加载设备与模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QCarWorldModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 加载归一化字典和测试数据
    norm_stats = torch.load(norm_path, map_location=device, weights_only=True)
    x_mean, x_std = norm_stats['x_mean'], norm_stats['x_std']
    y_mean, y_std = norm_stats['y_mean'], norm_stats['y_std']

    test_data = torch.load(test_data_path, map_location=device, weights_only=True)
    X_test = test_data['X_test']
    S_curr = test_data['S_curr']
    S_next_real = test_data['S_next_real']

    # --- 🤖 核心：让 AI 进行预测 ---
    print("🧠 AI 正在对测试集进行预测推理...")
    with torch.no_grad():
        # 1. 归一化输入
        X_test_norm = (X_test - x_mean) / x_std
        # 2. 模型预测输出归一化的 Delta
        pred_delta_norm = model(X_test_norm)
        # 3. 反归一化，还原真实的物理 Delta
        pred_delta_real = pred_delta_norm * y_std + y_mean

    # 4. 计算 AI 预测出的下一帧状态: 预测的下一个状态 = 当前真实状态 + AI预测的变化量
    S_next_pred = S_curr + pred_delta_real

    # --- 📈 生成 7 维表格与绘图 ---
    print("📝 正在生成 CSV 误差表格与对比曲线图...")
    # 老师规定的 7 维状态名称
    state_names = ['pos_x', 'pos_y', 'pos_z', 'rot_0', 'rot_1', 'rot_2', 'rot_3']

    for i, feature_name in enumerate(state_names):
        true_values = S_next_real[:, i].cpu().numpy()
        pred_values = S_next_pred[:, i].cpu().numpy()
        errors = true_values - pred_values

        # 1. 保存为 CSV 表格
        df = pd.DataFrame({
            'Time Step': range(1, len(true_values) + 1),
            f'True {feature_name}': true_values,
            f'Predicted {feature_name}': pred_values,
            'Error': errors
        })
        csv_filename = os.path.join(output_dir, f'{direction}_{feature_name}_error.csv')
        df.to_csv(csv_filename, index=False)

        # 2. 绘制时序对比图
        plt.figure(figsize=(10, 5))
        # 截取前 500 帧来画图，否则数据太密看不清趋势
        plot_limit = min(500, len(true_values))
        plt.plot(true_values[:plot_limit], label='True State (Ground Truth)', color='blue', linewidth=2)
        plt.plot(pred_values[:plot_limit], label='Predicted State (AI Model)', color='orange', linestyle='--',
                 linewidth=2)

        plt.title(f'{direction.upper()} Model: {feature_name} Prediction vs Reality')
        plt.xlabel('Time Step')
        plt.ylabel(f'{feature_name} Value')
        plt.legend()
        plt.grid(True)

        plot_filename = os.path.join(output_dir, f'{direction}_{feature_name}_plot.png')
        plt.savefig(plot_filename, dpi=150)
        plt.close()  # 关掉画板，释放内存

    print(f"✅ 完美！{direction.upper()} 模型的 7 个 CSV 表格和 7 张趋势图已保存在:\n📂 {output_dir}")


if __name__ == "__main__":
    evaluate_and_plot(direction="forward")
    evaluate_and_plot(direction="backward")
    print("\n🎉 终极任务完成！请去 results_evaluation 文件夹验收成果并发送给队友/导师！")