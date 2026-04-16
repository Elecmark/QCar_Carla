from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np
import torch

from carla_controller_PDH import (
    ModelBundle,
    apply_carla_yaw_delta_to_model_quat,
    bootstrap_model_history,
    choose_bundle_for_action,
    model_quat_to_carla_yaw_deg,
    predict_delta_state,
    predicted_output_to_next_state,
    wrap_angle_deg,
)
from policy_network import STEER_RESIDUAL_LIMIT, THROTTLE_RESIDUAL_LIMIT
from reference_generator import ReferenceTrajectory
from rewarder import compute_reward

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except ImportError:
        class _EnvBase:
            metadata = {}

        class _Box:
            def __init__(self, low, high, shape, dtype=np.float32):
                self.low = np.full(shape, low, dtype=dtype)
                self.high = np.full(shape, high, dtype=dtype)
                self.shape = shape
                self.dtype = dtype

        class _Spaces:
            Box = _Box

        class _GymModule:
            Env = _EnvBase

        gym = _GymModule()
        spaces = _Spaces()


@dataclass
class EnvConfig:
    max_steps: int = 829
    terminate_on_success: bool = False
    success_position_threshold: float = 0.20
    failure_position_threshold: float = 2.0
    failure_yaw_threshold_deg: float = 30.0
    reset_position_noise_xy: float = 0.5
    reset_yaw_noise_deg: float = 15.0
    reward_alpha: float = 0.5


class DTModelEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        forward_model: ModelBundle,
        backward_model: ModelBundle,
        device: torch.device,
        carla_client=None,
        env_config: Optional[EnvConfig] = None,
    ) -> None:
        super().__init__()
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.device = device
        self.carla_client = carla_client
        self.env_config = env_config or EnvConfig()
        self.seq_length = int(forward_model.normalizer.seq_length)

        self.action_space = spaces.Box(
            low=np.array([-THROTTLE_RESIDUAL_LIMIT, -STEER_RESIDUAL_LIMIT], dtype=np.float32),
            high=np.array([THROTTLE_RESIDUAL_LIMIT, STEER_RESIDUAL_LIMIT], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(26,), dtype=np.float32)

        self.history: Deque[np.ndarray] = deque(maxlen=self.seq_length)
        self.reference: Optional[ReferenceTrajectory] = None
        self.state: Optional[np.ndarray] = None
        self.prev_action: Optional[np.ndarray] = None
        self.step_idx = 0
        self.last_info: Dict[str, float] = {}
        self.predicted_states = []
        self.cumulative_state_error_sum = np.zeros(7, dtype=np.float32)
        self.forward_action_low = np.array(
            [
                max(0.0, float(self.forward_model.normalizer.x_mean[7] - 3.0 * self.forward_model.normalizer.x_std[7])),
                -0.45,
            ],
            dtype=np.float32,
        )
        self.forward_action_high = np.array(
            [
                min(0.20, float(self.forward_model.normalizer.x_mean[7] + 3.0 * self.forward_model.normalizer.x_std[7])),
                0.45,
            ],
            dtype=np.float32,
        )
        self.state_scale = np.array([20.0, 20.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.error_scale = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.residual_penalty_weight = 0.20

    def _normalize_vector(self, values: np.ndarray, scales: np.ndarray) -> np.ndarray:
        normalized = np.asarray(values, dtype=np.float32) / np.asarray(scales, dtype=np.float32)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        throttle_norm = np.clip((action[0] / 0.12) * 2.0 - 1.0, -1.0, 1.0)
        steer_norm = np.clip(action[1] / 0.45, -1.0, 1.0)
        return np.array([throttle_norm, steer_norm], dtype=np.float32)

    def _sample_initial_state(self, base_state: np.ndarray) -> np.ndarray:
        state = np.asarray(base_state, dtype=np.float32).copy()
        noise_xy = float(max(0.0, self.env_config.reset_position_noise_xy))
        noise_yaw_deg = float(max(0.0, self.env_config.reset_yaw_noise_deg))
        if noise_xy > 0.0:
            state[0] += float(np.random.uniform(-noise_xy, noise_xy))
            state[1] += float(np.random.uniform(-noise_xy, noise_xy))
        if noise_yaw_deg > 0.0:
            yaw_delta = float(np.random.uniform(-noise_yaw_deg, noise_yaw_deg))
            state[3:7] = apply_carla_yaw_delta_to_model_quat(state[3:7], yaw_delta)
        return state

    def _reference_action(self) -> np.ndarray:
        assert self.reference is not None
        action_idx = min(self.step_idx, self.reference.actions.shape[0] - 1)
        return self.reference.actions[action_idx].astype(np.float32, copy=True)

    def _project_residual_action(self, action: np.ndarray) -> np.ndarray:
        projected = np.asarray(action, dtype=np.float32).copy()
        projected[0] = np.clip(projected[0], -THROTTLE_RESIDUAL_LIMIT, THROTTLE_RESIDUAL_LIMIT)
        projected[1] = np.clip(projected[1], -STEER_RESIDUAL_LIMIT, STEER_RESIDUAL_LIMIT)
        return projected.astype(np.float32)

    def _project_final_action(self, action: np.ndarray) -> np.ndarray:
        projected = np.asarray(action, dtype=np.float32).copy()
        projected[0] = np.clip(projected[0], 0.0, 0.12)
        projected[1] = np.clip(projected[1], -0.45, 0.45)
        projected = np.clip(projected, self.forward_action_low, self.forward_action_high)
        projected[0] = np.clip(projected[0], 0.0, 0.12)
        return projected.astype(np.float32)

    def _baseline_action(self) -> np.ndarray:
        assert self.reference is not None
        assert self.state is not None
        current_ref = self.reference.states[min(self.step_idx, len(self.reference.states) - 1)]
        next_ref = self.reference.states[min(self.step_idx + 1, len(self.reference.states) - 1)]
        dt = 0.05
        if self.reference.times.shape[0] > 1:
            idx = min(self.step_idx, self.reference.times.shape[0] - 2)
            dt = max(float(self.reference.times[idx + 1] - self.reference.times[idx]), 1e-6)

        yaw_deg = float(model_quat_to_carla_yaw_deg(self.state[3:7]))
        yaw_rad = np.radians(yaw_deg)
        cos_y = float(np.cos(yaw_rad))
        sin_y = float(np.sin(yaw_rad))
        target_vec = next_ref[:2] - self.state[:2]
        forward_err = float(target_vec[0] * cos_y + target_vec[1] * sin_y)
        lateral_err = float(-target_vec[0] * sin_y + target_vec[1] * cos_y)
        yaw_target_deg = float(model_quat_to_carla_yaw_deg(next_ref[3:7]))
        yaw_err_deg = float(wrap_angle_deg(yaw_target_deg - yaw_deg))
        target_step = float(np.linalg.norm(next_ref[:2] - current_ref[:2]))
        target_speed = target_step / dt

        throttle = 0.04 + 0.42 * max(0.0, forward_err) + 0.02 * target_speed
        if forward_err < -0.05:
            throttle = 0.035
        steer = 0.55 * lateral_err + 0.012 * yaw_err_deg
        baseline = np.array([throttle, steer], dtype=np.float32)
        baseline[0] = np.clip(baseline[0], 0.035, 0.10)
        baseline[1] = np.clip(baseline[1], -0.30, 0.30)
        return self._project_final_action(baseline)

    def expert_action_to_policy_target(self, expert_action: np.ndarray) -> np.ndarray:
        baseline = self._baseline_action()
        residual = np.asarray(expert_action, dtype=np.float32) - baseline
        return self._project_residual_action(residual)

    def _build_observation(self) -> np.ndarray:
        assert self.state is not None
        assert self.reference is not None
        ref_now = self.reference.states[min(self.step_idx, len(self.reference.states) - 1)]
        tracking_error = self.state - ref_now
        prev_action = self.prev_action if self.prev_action is not None else np.zeros(2, dtype=np.float32)
        current_loss = float(np.mean(np.square(self.state - ref_now)))
        history_steps = max(1, int(self.step_idx))
        cumulative_mean_loss = float(np.mean(self.cumulative_state_error_sum / float(history_steps))) if self.step_idx > 0 else current_loss
        yaw_error_deg = float(wrap_angle_deg(model_quat_to_carla_yaw_deg(self.state[3:7]) - model_quat_to_carla_yaw_deg(ref_now[3:7])))
        return np.concatenate(
            [
                self._normalize_vector(self.state, self.state_scale),
                self._normalize_vector(ref_now, self.state_scale),
                self._normalize_vector(tracking_error, self.error_scale),
                self._normalize_action(prev_action),
                np.array(
                    [
                        np.clip(cumulative_mean_loss / 4.0, -1.0, 1.0),
                        np.clip(current_loss / 4.0, -1.0, 1.0),
                        np.clip(yaw_error_deg / 30.0, -1.0, 1.0),
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        ).astype(np.float32)

    def _history_array(self, action: np.ndarray) -> np.ndarray:
        assert self.state is not None
        feature = np.concatenate([self.state, action], axis=0).astype(np.float32)
        temp_history = deque(self.history, maxlen=self.seq_length)
        if not temp_history:
            for _ in range(self.seq_length):
                temp_history.append(feature.copy())
        else:
            temp_history.append(feature.copy())
            while len(temp_history) < self.seq_length:
                temp_history.appendleft(temp_history[0].copy())
        return np.stack(list(temp_history), axis=0).astype(np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, object]] = None,
        initial_state: Optional[np.ndarray] = None,
        reference_trajectory: Optional[ReferenceTrajectory] = None,
    ):
        del seed
        options = dict(options or {})
        if reference_trajectory is None:
            reference_trajectory = options.get("reference_trajectory")
        if reference_trajectory is None:
            raise ValueError("reference_trajectory is required for DTModelEnv.reset()")

        self.reference = reference_trajectory
        self.step_idx = 0
        self.prev_action = None
        self.predicted_states = []
        self.last_info = {}
        self.cumulative_state_error_sum = np.zeros(7, dtype=np.float32)

        if initial_state is not None:
            self.state = np.asarray(initial_state, dtype=np.float32).copy()
        else:
            self.state = self._sample_initial_state(reference_trajectory.states[0])
        bootstrap_action = reference_trajectory.actions[0] if reference_trajectory.actions.shape[0] > 0 else np.zeros(2, dtype=np.float32)
        bootstrap_action = self._project_final_action(bootstrap_action)
        bootstrap_model_history(self.history, self.state, bootstrap_action, self.seq_length)

        observation = self._build_observation()
        initial_error = float(np.linalg.norm(self.state[:3] - reference_trajectory.states[0][:3]))
        return observation, {
            "reference_name": reference_trajectory.name,
            "reference_type": reference_trajectory.metadata.get("type", "unknown"),
            "initial_pos_error": initial_error,
        }

    def step(self, action: np.ndarray):
        if self.reference is None or self.state is None:
            raise RuntimeError("Environment must be reset before calling step().")

        raw_action = np.asarray(action, dtype=np.float32).copy()
        ref_action = self._reference_action()
        baseline_action = self._baseline_action()
        clipped_residual = self._project_residual_action(raw_action)
        clipped_action = self._project_final_action(baseline_action + clipped_residual)
        bundle = choose_bundle_for_action(clipped_action, self.forward_model, self.backward_model)
        history_np = self._history_array(clipped_action)
        predicted_delta = predict_delta_state(history_np, bundle, self.device)
        next_state = predicted_output_to_next_state(self.state, predicted_delta)
        ref_next = self.reference.states[min(self.step_idx + 1, len(self.reference.states) - 1)]

        reward_info = compute_reward(
            predicted_state=next_state,
            reference_current_state=self.reference.states[min(self.step_idx, len(self.reference.states) - 1)],
            reference_state=ref_next,
            prev_action=self.prev_action,
            current_action=clipped_action,
            cumulative_state_error_sum=self.cumulative_state_error_sum,
            step_index=self.step_idx + 1,
            alpha=self.env_config.reward_alpha,
            success_position_threshold=self.env_config.success_position_threshold,
            failure_position_threshold=self.env_config.failure_position_threshold,
            failure_yaw_threshold_deg=self.env_config.failure_yaw_threshold_deg,
        )
        residual_ratio = np.array(
            [
                clipped_residual[0] / max(THROTTLE_RESIDUAL_LIMIT, 1e-6),
                clipped_residual[1] / max(STEER_RESIDUAL_LIMIT, 1e-6),
            ],
            dtype=np.float32,
        )
        residual_penalty = float(self.residual_penalty_weight * np.mean(np.square(residual_ratio)))
        reward_info["reward"] = float(reward_info["reward"] - residual_penalty)
        reward_info["loss"] = float(reward_info["loss"] + residual_penalty)
        reward_info["residual_penalty"] = residual_penalty
        self.cumulative_state_error_sum = reward_info["updated_cumulative_state_error_sum"].astype(np.float32)

        self.history.append(np.concatenate([self.state, clipped_action], axis=0).astype(np.float32))
        self.prev_action = clipped_action
        self.state = next_state
        self.predicted_states.append(next_state.copy())
        self.step_idx += 1

        terminated = bool(reward_info["terminated"] > 0.5)
        truncated = self.step_idx >= min(self.env_config.max_steps, self.reference.states.shape[0] - 1)
        if self.env_config.terminate_on_success and reward_info["success"] > 0.5:
            terminated = True

        observation = self._build_observation()
        info = {
            "bundle_name": bundle.name,
            "predicted_delta_x": float(predicted_delta[0]),
            "predicted_delta_y": float(predicted_delta[1]),
            "predicted_delta_z": float(predicted_delta[2]),
            "predicted_delta_yaw": float(predicted_delta[3]),
            "step": self.step_idx,
            "reference_name": self.reference.name,
            "reference_type": self.reference.metadata.get("type", "unknown"),
            "raw_action_throttle": float(raw_action[0]),
            "raw_action_steering": float(raw_action[1]),
            "baseline_action_throttle": float(baseline_action[0]),
            "baseline_action_steering": float(baseline_action[1]),
            "residual_action_throttle": float(clipped_residual[0]),
            "residual_action_steering": float(clipped_residual[1]),
            "applied_action_throttle": float(clipped_action[0]),
            "applied_action_steering": float(clipped_action[1]),
            "reference_action_throttle": float(ref_action[0]),
            "reference_action_steering": float(ref_action[1]),
            **reward_info,
        }
        info.pop("updated_cumulative_state_error_sum", None)
        self.last_info = info
        return observation, float(reward_info["reward"]), terminated, truncated, info

    def render(self):
        if self.reference is None:
            return None
        return {
            "reference_states": self.reference.states.copy(),
            "predicted_states": np.asarray(self.predicted_states, dtype=np.float32),
            "last_info": dict(self.last_info),
        }

    def current_reference(self) -> ReferenceTrajectory:
        if self.reference is None:
            raise RuntimeError("Environment has no reference trajectory loaded.")
        return self.reference
