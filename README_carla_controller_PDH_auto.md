# `carla_controller_PDH_auto.py` 说明

这份文档只针对当前项目里的 `carla_controller_PDH_auto.py`。

目标是把下面几件事说明清楚：

- 这个自动控制器当前到底是什么原理
- DT world model 的输入输出是什么
- RL controller 的输入输出是什么
- 训练链路是怎么串起来的
- 根目录和 `PDHModel/` 下哪些模型文件在起作用
- 当前实现里已经存在的关键问题

---

## 1. 当前 `auto` 控制器的整体原理

`carla_controller_PDH_auto.py` 的工作方式不是“RL 直接开车”，而是：

1. 读取一条参考轨迹 CSV
2. 把参考轨迹对齐到当前 CARLA 车辆的起始位置和朝向
3. 根据当前车辆状态和下一帧参考状态，先构造一个 **baseline action**
4. 如果启用了 RL correction，就让 RL 输出一个 **residual action**
5. `final_action = baseline_action + residual_action`
6. 把 `final_action` 转成 CARLA `VehicleControl`
7. 把控制量直接发给 CARLA 真车体

所以当前 `auto` 的本质是：

- **baseline 跟踪器** 是主控制器
- **RL** 只是修正器
- **CARLA 真车体** 是最终闭环对象
- **DT world model** 不是最终执行器，而是辅助模块

---

## 2. 运行时数据流

当前 `auto` 一步的主要流程在 `_ReplayInstance.step_once()` 里：

1. 从 CARLA 取当前车辆真实状态
2. 转到参考轨迹坐标系下，得到 `current_route_state`
3. 用 `current_route_state + current_ref + next_ref` 构造 RL observation
4. 用 baseline 控制律算出 `baseline_action`
5. RL 输出 residual，叠加后得到 `action`
6. 用 `action` 选 forward/backward world model，并调用 DT 预测 `predicted_delta`
7. 但真正执行时，代码调用的是：
   - `self.actor.apply_control(control)`
8. `world.tick()` 之后再读取 CARLA 真实结果，计算误差并记录日志

结论：

- **CARLA 真实状态** 决定下一步控制
- **RL** 是残差修正
- **DT** 当前主要用于：
  - 文件加载时估计 nominal speed
  - 根据 `action` 预测一步 delta，做日志和诊断
  - 选择 forward/backward bundle

当前 `auto` 不是“DT rollout 驱动 CARLA”，而是“CARLA 真闭环 + DT 辅助预测/分析”。

---

## 3. DT world model

### 3.1 模型定义

DT world model 定义在 `carla_controller_PDH.py` 里的 `QCarWorldModel`：

- 结构：LSTM + MLP
- 输入维度：`9`
- 输出维度：`4`

### 3.2 DT 输入

每个时间步的输入特征是：

- `state[7]`
  - `pos_x`
  - `pos_y`
  - `pos_z`
  - `rot_0`
  - `rot_1`
  - `rot_2`
  - `rot_3`
- `action[2]`
  - `throttle`
  - `steer`

所以单帧输入是 `7 + 2 = 9` 维。

实际送入模型的是一段历史序列：

- shape: `T x 9`
- `T = normalizer.seq_length`

并且在送入前会做：

- 历史位置以最后一帧位置为锚点做平移归一
- 四元数做相对姿态归一
- 用 `forward_normalization.pt / backward_normalization.pt` 标准化

### 3.3 DT 输出

模型输出 4 维：

- `delta_x_body`
- `delta_y_body`
- `delta_z_body`
- `delta_yaw_deg`

含义是“在当前车体坐标系下的一步位姿变化”。

然后通过 `predicted_output_to_next_state()` 变成下一时刻状态：

- 位置：把 body frame 位移旋转到世界/模型坐标系
- 姿态：在当前 yaw 上叠加 `delta_yaw_deg`

### 3.4 forward / backward 两套模型

当前项目有两套 DT：

- `PDHModel/forward_world_model.pth`
- `PDHModel/backward_world_model.pth`

对应归一化文件：

- `PDHModel/forward_normalization.pt`
- `PDHModel/backward_normalization.pt`

运行时通过动作油门符号选择：

- `throttle >= 0` -> forward model
- `throttle < 0` -> backward model

---

## 4. RL controller

### 4.1 模型定义

RL policy 定义在 `policy_network.py`：

- 算法：SAC
- actor/critic 隐层：`(256, 256)`
- actor 输出：2 维 residual action

默认 `auto` 使用的策略文件：

- `PDHModel/spec_rl_resampled_fix1/policy_controller.pth`

对应配置：

- `PDHModel/spec_rl_resampled_fix1/policy_config.json`

其中关键配置：

- `obs_dim = 26`
- `action_dim = 2`
- `episodes = 240`
- `max_steps = 829`
- `batch_size = 256`
- `buffer_size = 1000000`
- `reset_position_noise_xy = 0.75`
- `reset_yaw_noise_deg = 20.0`

### 4.2 RL observation

RL observation 在 `carla_controller_PDH_auto.py::_build_policy_observation()` 中构造，为 `26` 维：

1. 当前状态归一化 `current_state[7]`
2. 当前参考状态归一化 `current_ref.state[7]`
3. 跟踪误差归一化 `tracking_error[7]`
4. 上一步动作归一化 `prev_action[2]`
5. 上下文统计量 `3`
   - `cumulative_mean_loss`
   - `current_loss`
   - `yaw_error_deg`

总维度：

- `7 + 7 + 7 + 2 + 3 = 26`

归一尺度：

- `state_scale = [20, 20, 2, 1, 1, 1, 1]`
- `error_scale = [5, 5, 1, 1, 1, 1, 1]`

动作归一化：

- throttle: `(a[0] / 0.12) * 2 - 1`
- steer: `a[1] / 0.45`

### 4.3 RL action

RL 输出 2 维 residual：

- `residual_throttle`
- `residual_steer`

`policy_network.py` 里的理论约束是：

- throttle residual: `[-0.03, 0.03]`
- steer residual: `[-0.12, 0.12]`

但 `carla_controller_PDH_auto.py` 运行时又做了一层更宽的 clip：

- throttle residual clip: `[-0.06, 0.06]`
- steer residual clip: `[-0.25, 0.25]`

然后运行时叠加方式是：

```python
applied_action = clip_action_for_dt(baseline_action + residual_action)
```

也就是说 RL 不直接输出最终控制，而是输出 baseline 上的修正项。

---

## 5. Baseline controller

当前 `auto` 的 baseline 是一个手工控制律，在 `_build_baseline_action()`：

- 利用当前状态和下一帧参考状态计算：
  - 前向误差 `forward_err`
  - 横向误差 `lateral_err`
  - 航向误差 `yaw_err_deg`
  - 目标步长/目标速度

然后生成：

- throttle
- steer

核心特征：

- throttle 只做正向前进控制
- steer 由横向误差和航向误差线性组合得到

再经过 `_clip_action_for_dt()` 做裁剪。

所以从控制结构上看：

- baseline 决定主要趋势
- RL 只负责修边

---

## 6. 训练过程

### 6.1 DT world model 训练

世界模型训练脚本主要是：

- `PDH_train_world_model.py`

产物：

- `forward_world_model.pth`
- `backward_world_model.pth`
- 对应 normalization 文件

world model 的训练目标是：

- 给定历史 `state + action` 序列
- 预测下一步的位移和偏航变化

### 6.2 RL controller 训练

RL 训练主脚本：

- `train_rl_controller.py`

训练环境：

- `dt_model_env.py::DTModelEnv`

流程：

1. 加载 forward/backward world model
2. 构造参考轨迹
   - 可以来自生成轨迹
   - 也可以来自 `QCarDataSet`
   - 也可以来自完整 reference trajectory CSV
3. 在 `DTModelEnv` 中训练 residual policy

#### 环境中的动作空间

`DTModelEnv.action_space`：

- shape: `(2,)`
- throttle residual: `[-0.03, 0.03]`
- steer residual: `[-0.12, 0.12]`

#### 环境中的 observation 空间

`DTModelEnv.observation_space`：

- shape: `(26,)`

#### 环境中的一步更新

在 `DTModelEnv.step()`：

1. 算 `baseline_action`
2. 对策略输出 residual 做裁剪
3. `clipped_action = baseline_action + clipped_residual`
4. 把 `clipped_action` 输入 DT world model
5. 得到 `next_state`
6. 和 reference next state 比较，算 reward

#### Reward 组成

主要在 `rewarder.py::compute_reward()`：

- 当前状态误差
- 累计状态误差
- yaw penalty
- action saturation penalty
- action smooth penalty
- residual penalty（在 `DTModelEnv.step()` 里额外加）

所以训练目标是：

- 在 DT world model 环境里，学会对 baseline 做 residual 修正，使参考跟踪误差更小

### 6.3 RL 训练前的 BC 预热

`train_rl_controller.py` 在 SAC 正式训练前还会做一段行为克隆预热：

- `behavior_clone_pretrain()`

方式：

1. 采样参考轨迹
2. 把 expert/final action 转成 residual target
3. 用 actor 做 supervised warmup

这一步的目的是：

- 不让 SAC 从完全随机 residual 开始
- 先把 policy 拉到一个“近似 baseline 修正器”的初始状态

---

## 7. 运行时涉及的主要模型与数据文件

### 7.1 `auto` 默认直接使用

- DT:
  - `PDHModel/forward_world_model.pth`
  - `PDHModel/backward_world_model.pth`
  - `PDHModel/forward_normalization.pt`
  - `PDHModel/backward_normalization.pt`
- RL:
  - `PDHModel/spec_rl_resampled_fix1/policy_controller.pth`
  - `PDHModel/spec_rl_resampled_fix1/policy_config.json`
- 参考轨迹目录:
  - `PDHModel/reference_trajectories/`

### 7.2 运行日志

自动回放日志目录：

- `PDH_auto_logs/`

当前已经能看到的日志文件示例：

- `circle_radius5_dt0.004_auto_error.csv`
- `figure8_size10_dt0.004_auto_error.csv`
- `s_curve_length20_dt0.004_auto_error.csv`

### 7.3 项目里存在的其他 controller/policy 产物

`PDHModel/` 下还存在很多实验目录，例如：

- `il_controller_*`
- `dagger_*`
- `rl_*`
- `spec_rl_*`
- `recovery_rl_*`

但 `carla_controller_PDH_auto.py` 默认用的是：

- `spec_rl_resampled_fix1`

---

## 8. 当前 `carla_controller_PDH_auto.py` 的已知问题

这里只写当前实现层面的实际问题，不写泛泛而谈。

### 8.1 运行时结构和 RL 训练环境不完全一致

训练时的 RL 环境是 `DTModelEnv`：

- baseline 控制律参数是一套
- residual 限幅是一套
- 状态演化由 DT rollout 决定

运行时的 `carla_controller_PDH_auto.py`：

- baseline 控制律参数不是完全同一套
- residual clip 范围更宽
- 真正闭环对象是 CARLA，而不是 DT rollout

这意味着：

- 训练分布和部署分布存在偏差
- policy 在 `DTModelEnv` 里有效，不代表在 `auto` 里同样稳定

### 8.2 baseline 控制律与训练环境版本不一致

`DTModelEnv._baseline_action()` 和 `carla_controller_PDH_auto.py::_build_baseline_action()` 的系数不同。

这很关键，因为 residual policy 学的是“修正某一个 baseline”，不是修正任意 baseline。

结果就是：

- baseline 一换，residual 的意义就变了

### 8.3 residual 运行时 clip 与训练约束不一致

训练环境动作空间：

- throttle residual: `[-0.03, 0.03]`
- steer residual: `[-0.12, 0.12]`

运行时 `auto` 又裁到：

- throttle residual: `[-0.06, 0.06]`
- steer residual: `[-0.25, 0.25]`

虽然 policy 本身通常输出不到这么大，但代码层面这是不一致的。

### 8.4 当前 `auto` 实际上几乎只在前进

`_build_baseline_action()` 的 throttle 是正值区间：

- 大致 `0.035 ~ 0.12`

运行时还调用：

```python
control = build_vehicle_control(float(max(0.0, action[0])), float(action[1]))
```

这进一步把负 throttle 截掉了。

所以对当前 `auto` 而言：

- backward world model 基本不会真正参与闭环控制
- 自动回放本质上是一个前进跟踪器

### 8.5 DT 在 `auto` 里更多是辅助预测，不是主 rollout 引擎

当前代码里确实计算了：

- `predicted_delta`
- `predicted_route_state`

但真正执行不是把 DT 预测状态写回 CARLA，而是：

- 直接把控制量发给 CARLA
- tick 完再读取 CARLA 真实状态

因此当前 `auto` 中的 DT 更像：

- 调试分析器
- 辅助选择 bundle 的预测器

而不是完整的闭环状态推进器

### 8.6 policy observation 用的是 `current_ref`，不是 `next_ref`

`_build_policy_observation()` 里直接把：

- 当前状态
- 当前参考状态
- 当前误差

拼成 observation，并没有把 `next_ref` 本体直接送进去，只是外部 baseline/action 计算用到了 `next_ref`。

这会让 RL 更偏向“纠正当前误差”，而不是显式看未来一步目标。

### 8.7 当前 route 坐标系只保留 yaw，不保留完整 3D 姿态

在 `_current_actor_route_state()` 中：

- route quaternion 是由 yaw 重新构造的
- 没有保留 roll / pitch

如果轨迹或 CARLA 场景里存在明显坡度或车体姿态变化，这里会有信息损失。

### 8.8 launch assist 会覆盖动作分布

`_apply_launch_assist()` 在低速/卡住时会强行把：

- throttle 拉高
- steer 收窄

这对“让车动起来”有帮助，但也会改变 policy 实际落地动作分布。

所以日志里看到的 `policy_action` 不一定等于最终施加到 CARLA 的 `action`。

### 8.9 `policy_action` 的日志语义不总是“RL 输出”

在 `_choose_corrected_action()` 中：

- RL 开启时，`policy_action` 表示 residual
- RL 关闭时，`policy_action` 被直接设成 baseline

所以日志字段 `policy_throttle/policy_steer` 的语义其实不是恒定的。

### 8.10 `use_rl_correction` 默认是开

代码里：

- `USE_RL_CORRECTION_DEFAULT = True`

这意味着默认启动就是 baseline + RL residual，而不是先看 baseline 裸跑效果。

从调试角度看，这并不理想。

---

## 9. 当前 `auto` 应该如何理解

一句话总结：

`carla_controller_PDH_auto.py` 目前更像是一个：

- **参考轨迹跟踪 baseline controller**
- 加一个 **在 DT 环境里训练出的 residual RL policy**
- 最终在 **CARLA 真实车辆闭环** 中执行的系统

它不是：

- 纯 RL controller
- 纯 DT rollout controller
- 完全训练部署一致的 controller

---

## 10. 如果后续要继续改，优先级建议

如果继续收敛这个 `auto` 控制器，建议优先做下面几件事：

1. 统一训练环境和运行时 baseline 控制律
2. 统一 residual action clip 范围
3. 明确 `auto` 是否真的需要 backward 跟踪能力
4. 决定 DT 在运行时到底是：
   - 只做分析
   - 还是参与闭环 rollout
5. 统一日志字段语义，区分：
   - baseline action
   - residual action
   - final applied action
6. 把 `use_rl_correction` 默认改成可对比模式
   - 先 baseline
   - 再 baseline + RL

---

## 11. 对当前文件最准确的简短结论

`carla_controller_PDH_auto.py` 当前是：

- **参考驱动**
- **baseline 主导**
- **RL residual 修正**
- **CARLA 真闭环**
- **DT 辅助预测与诊断**

这也是理解当前效果和问题的正确前提。
