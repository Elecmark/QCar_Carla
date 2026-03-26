import os
import pandas as pd
import torch
import numpy as np


def process_dataset(parent_dir, output_file):
    """
    核心功能：遍历指定的父文件夹，把里面所有的 CSV 文件合并，
    并只提取老师要求的 2维控制量(UT) 和 7维车辆状态(ST)，最后打包成 .pt 文件。
    """
    print(f"\n🚀 开始处理数据集: {parent_dir}")

    # 老师规定的 2 维控制输入 (Actuation: u)
    action_cols = ['throttle', 'steering']

    # 老师规定的 7 维核心车辆状态 (State: S)
    # 包含 3维位置坐标 (x, y, z) 和 4维朝向角/四元数 (rot_0 到 rot_3)
    state_cols = ['pos_x', 'pos_y', 'pos_z', 'rot_0', 'rot_1', 'rot_2', 'rot_3']

    all_actions = []
    all_states = []
    total_files_processed = 0

    # 检查你设置的 raw_data 路径存不存在
    if not os.path.exists(parent_dir):
        print(f"❌ 找不到文件夹: '{parent_dir}'，请检查你的 data_raw 目录结构！")
        return False

    # 1. 遍历父文件夹下所有的子文件夹 (例如 linefollow_constant 等)
    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)

        # 如果不是文件夹就跳过
        if not os.path.isdir(folder_path):
            continue

        print(f"  📂 正在合并子轨迹数据: {folder_name}")

        # 2. 遍历该轨迹文件夹下的所有 CSV 文件
        for file_name in os.listdir(folder_path):
            if not file_name.endswith('.csv'):
                continue

            file_path = os.path.join(folder_path, file_name)

            try:
                # 使用 pandas 读取 CSV 表格
                df = pd.read_csv(file_path)

                # 检查这个表格里有没有缺失老师要求的核心列
                missing_cols = [col for col in (action_cols + state_cols) if col not in df.columns]
                if missing_cols:
                    print(f"    ⚠️ 跳过文件 {file_name}: 缺失核心列 {missing_cols}")
                    continue

                # 清理掉包含空值 (NaN) 的脏数据行，防止毒害 AI
                df = df.dropna(subset=action_cols + state_cols)

                # 提取出纯净的数据，并转换为 numpy 浮点数数组
                actions_np = df[action_cols].values.astype(np.float32)
                states_np = df[state_cols].values.astype(np.float32)

                # 把提取出来的数据塞进我们的大列表里
                all_actions.append(actions_np)
                all_states.append(states_np)
                total_files_processed += 1

            except Exception as e:
                print(f"    ❌ 读取 {file_name} 时发生错误: {e}")

    # 3. 汇总并打包成模型所需的格式
    if len(all_actions) == 0:
        print(f"❌ {parent_dir} 中没有提取到任何有效数据！")
        return False

    # 将成百上千个小表格的数据，垂直堆叠成一个巨大的矩阵
    final_actions = np.vstack(all_actions)
    final_states = np.vstack(all_states)

    print(f"✅ 处理完成！共读取并合并了 {total_files_processed} 个 CSV 文件。")
    print(f"📊 提炼出有效数据帧 (Timesteps): {len(final_actions)} 行")

    # 保存为 PyTorch 专属的张量字典 (.pt 文件)
    # inputs 对应动作 U，labels 对应状态 S
    torch.save({
        'inputs': torch.tensor(final_actions),
        'labels': torch.tensor(final_states)
    }, output_file)

    print(f"💾 数据已成功打包至: {output_file}")
    return True


if __name__ == "__main__":
    # ======== 🌟 核心路径配置 ========
    # 根据你刚才新建的目录结构，指定原始数据的根目录
    base_dir = './data_raw/QCarRawData'

    # 严格按照老师的要求，区分向前开和向后倒车
    forward_dir = os.path.join(base_dir, 'Forward_Dataset')
    backward_dir = os.path.join(base_dir, 'Backward_Dataset')

    # 确保保存处理后数据的文件夹存在，如果没有就自动创建一个
    os.makedirs('./data_processed', exist_ok=True)

    # 分别运行数据处理流水线，生成两个纯净的数据包
    process_dataset(forward_dir, './data_processed/qcar_forward_dataset.pt')
    process_dataset(backward_dir, './data_processed/qcar_backward_dataset.pt')

    print("\n🎉 第一阶段：全部数据清洗与分类任务圆满完成！")