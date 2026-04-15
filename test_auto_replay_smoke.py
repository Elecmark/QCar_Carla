from types import SimpleNamespace
from pathlib import Path

import torch

import carla_controller_PDH_auto as auto
from carla_controller_PDH import find_project_root, load_bundle


class _Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _Rotation:
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw


class _Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or _Location()
        self.rotation = rotation or _Rotation()


class _Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _FakeActor:
    def __init__(self, tf: _Transform):
        self._tf = tf

    def get_transform(self):
        return _Transform(
            _Location(self._tf.location.x, self._tf.location.y, self._tf.location.z),
            _Rotation(self._tf.rotation.roll, self._tf.rotation.pitch, self._tf.rotation.yaw),
        )

    def set_transform(self, tf):
        self._tf = tf

    def set_target_velocity(self, _):
        return None

    def set_target_angular_velocity(self, _):
        return None


def main() -> None:
    auto.carla = SimpleNamespace(Location=_Location, Rotation=_Rotation, Transform=_Transform, Vector3D=_Vector3D)
    project_root = Path(find_project_root())
    device = torch.device("cpu")
    forward_bundle = load_bundle(
        "forward",
        str(project_root / "PDHModel" / "forward_world_model.pth"),
        str(project_root / "PDHModel" / "forward_normalization.pt"),
        device,
    )
    backward_bundle = load_bundle(
        "backward",
        str(project_root / "PDHModel" / "backward_world_model.pth"),
        str(project_root / "PDHModel" / "backward_normalization.pt"),
        device,
    )
    reference_path = project_root / "PDHModel" / "reference_trajectories" / "circle_radius5_dt0.004.csv"
    raw_frames = auto.load_reference_frames(str(reference_path))
    reference_speed = sum(frame.linear_speed for frame in raw_frames[1:100]) / 99.0
    nominal_speed = auto.estimate_forward_nominal_speed(forward_bundle, raw_frames[0].state, 0.05)
    route_dt = max(float(raw_frames[1].time - raw_frames[0].time), min(0.05, 0.05 * nominal_speed / max(reference_speed, 1e-6)))
    frames = auto.resample_reference_frames(raw_frames, route_dt)

    actor = _FakeActor(_Transform(_Location(0.0, 0.0, 0.0), _Rotation(0.0, 0.0, 0.0)))
    replay = auto._ReplayInstance(
        forward_bundle=forward_bundle,
        backward_bundle=backward_bundle,
        reference_frames=frames,
        fixed_delta=0.05,
        actor=actor,
        seq_length=int(forward_bundle.normalizer.seq_length),
        interpolation_alpha=1.0,
        csv_path=str(reference_path),
        policy_agent=None,
        policy_deterministic=True,
        reference_action_blend=0.25,
        reference_steer_gain=1.0,
        yaw_step_steer_gain=0.28,
        debug_draw_enabled=False,
    )
    replay.step_once(world=None)
    record = replay.records[-1]
    print(
        f"smoke total_loss={record['total_loss']:.6f} "
        f"pos_l2={record['error_pos_l2']:.6f} yaw={record['error_yaw_deg']:.6f}"
    )
    if record["total_loss"] > 1.0:
        raise SystemExit("auto replay smoke test failed: total_loss too large")


if __name__ == "__main__":
    main()
