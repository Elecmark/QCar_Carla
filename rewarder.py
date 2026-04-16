from typing import Dict, Optional

import numpy as np

from carla_controller_PDH import model_quat_to_carla_yaw_deg, wrap_angle_deg


def _state_squared_error(predicted_state: np.ndarray, reference_state: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted_state, dtype=np.float32)
    reference = np.asarray(reference_state, dtype=np.float32)
    return np.square(predicted - reference).astype(np.float32)


def compute_reward(
    predicted_state: np.ndarray,
    reference_current_state: np.ndarray,
    reference_state: np.ndarray,
    prev_action: Optional[np.ndarray],
    current_action: np.ndarray,
    cumulative_state_error_sum: np.ndarray,
    step_index: int,
    alpha: float = 0.5,
    yaw_weight: float = 0.35,
    action_saturation_weight: float = 0.75,
    action_smooth_weight: float = 0.25,
    success_position_threshold: float = 0.20,
    failure_position_threshold: float = 2.0,
    failure_yaw_threshold_deg: float = 30.0,
) -> Dict[str, float]:
    predicted = np.asarray(predicted_state, dtype=np.float32)
    reference = np.asarray(reference_state, dtype=np.float32)
    current_action = np.asarray(current_action, dtype=np.float32)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    state_sq_error = _state_squared_error(predicted, reference)
    current_loss = float(np.mean(state_sq_error))

    cumulative_error_sum = np.asarray(cumulative_state_error_sum, dtype=np.float32) + state_sq_error
    history_steps = max(1, int(step_index) + 1)
    cumulative_mean_loss = float(np.mean(cumulative_error_sum / float(history_steps)))

    yaw_pred = float(model_quat_to_carla_yaw_deg(predicted[3:7]))
    yaw_ref = float(model_quat_to_carla_yaw_deg(reference[3:7]))
    yaw_error_deg = float(abs(wrap_angle_deg(yaw_pred - yaw_ref)))
    yaw_penalty = yaw_weight * (yaw_error_deg / 30.0) ** 2

    throttle = float(current_action[0])
    steer = float(current_action[1])
    throttle_excess = max(0.0, throttle - 0.115)
    steer_excess = max(0.0, abs(steer) - 0.40)
    action_saturation_penalty = action_saturation_weight * (throttle_excess * throttle_excess + steer_excess * steer_excess)

    smooth_penalty = 0.0
    if prev_action is not None:
        prev = np.asarray(prev_action, dtype=np.float32)
        smooth_penalty = action_smooth_weight * float(np.mean(np.square(current_action - prev)))

    pos_error = float(np.linalg.norm(predicted[:3] - reference[:3]))
    base_loss = alpha * cumulative_mean_loss + (1.0 - alpha) * current_loss
    total_loss = float(base_loss + yaw_penalty + action_saturation_penalty + smooth_penalty)
    reward = float(-total_loss)
    success = pos_error <= success_position_threshold and yaw_error_deg <= 10.0
    terminated = pos_error >= failure_position_threshold or yaw_error_deg >= failure_yaw_threshold_deg

    return {
        "reward": float(reward),
        "loss": total_loss,
        "base_loss": float(base_loss),
        "current_loss": current_loss,
        "cumulative_mean_loss": cumulative_mean_loss,
        "pos_error": pos_error,
        "yaw_error_deg": yaw_error_deg,
        "yaw_penalty": float(yaw_penalty),
        "action_saturation_penalty": float(action_saturation_penalty),
        "smooth_penalty": float(smooth_penalty),
        "terminated": float(terminated),
        "success": float(success),
        "updated_cumulative_state_error_sum": cumulative_error_sum.astype(np.float32),
    }
