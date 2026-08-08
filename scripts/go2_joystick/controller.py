import os
import threading

import numpy as np
import onnxruntime as rt

from consts import (
    Kd,
    Kp,
    action_scale,
    cmd_max_vx,
    cmd_max_vy,
    cmd_max_yaw,
    ctrl_dt,
    default_qpos,
    idx_map,
    policy_action_dim,
    policy_obs_dim,
    sim_dt,
    stand_down_joint_pos,
    stand_kd,
    stand_kp_down,
    stand_kp_up,
)

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, WirelessController_
from unitree_sdk2py.utils.crc import CRC



def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    r, i, j, k = quaternion
    two_s = 2.0 / np.sum(quaternion * quaternion)
    return np.array(
        [
            1.0 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1.0 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1.0 - two_s * (i * i + j * j),
        ],
        dtype=np.float32,
    ).reshape(3, 3)


def rotate_inv(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    return quaternion_to_matrix(quaternion).T @ np.asarray(vector, dtype=np.float32)


class Go2JoystickOnnxController:
    """Single-policy Go2Joystick controller for sim2sim and sim2real."""

    def __init__(self, policy_path: str):
        self._policy_path = os.path.abspath(policy_path)
        self._policy = rt.InferenceSession(
            self._policy_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._policy.get_inputs()[0].name
        input_shape = self._policy.get_inputs()[0].shape
        input_dim = input_shape[-1]
        if isinstance(input_dim, int) and input_dim != policy_obs_dim:
            raise ValueError(
                f"Expected a {policy_obs_dim}-D Go2Joystick policy, got {input_shape}"
            )
        output_names = {output.name for output in self._policy.get_outputs()}
        if "actions" not in output_names:
            raise ValueError(f"ONNX policy has no actions output: {output_names}")

        self._default_angles = default_qpos[7:].astype(np.float32)
        self._command = np.zeros(3, dtype=np.float32)
        # Go2Joystick stores the normalized action, not the joint target.
        self._last_action = np.zeros(policy_action_dim, dtype=np.float32)

        self._counter = 0
        self._n_substeps = int(round(ctrl_dt / sim_dt))
        self._last_missing_state_log_t = 0.0

        self.lock = threading.Lock()
        self.latest_low_state = None
        self.latest_wireless = None
        self._wireless_keys = 0
        self._last_start_select = False
        self._shutdown_active = False
        self._shutdown_t = 0.0
        self._standup_initialized = False
        self._standup_t = 0.0
        self._standup_start_qpos = np.zeros(12, dtype=np.float32)

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0

        self.crc = CRC()

        low_state_suber = ChannelSubscriber("rt/lowstate", LowState_)
        wireless_suber = ChannelSubscriber(
            "rt/wirelesscontroller", WirelessController_
        )
        low_state_suber.Init(self.LowStateHandler, 10)
        wireless_suber.Init(self.WirelessHandler, 10)

        self._key_bits = {
            "start": 2,
            "select": 3,
        }

    def LowStateHandler(self, msg: LowState_):
        with self.lock:
            self.latest_low_state = {
                "imu_quat": np.array(msg.imu_state.quaternion, dtype=np.float32),
                "imu_gyro": np.array(msg.imu_state.gyroscope, dtype=np.float32),
                "motor_q": np.array(
                    [motor.q for motor in msg.motor_state[:12]], dtype=np.float32
                ),
                "motor_dq": np.array(
                    [motor.dq for motor in msg.motor_state[:12]], dtype=np.float32
                ),
            }

    def WirelessHandler(self, msg: WirelessController_):
        with self.lock:
            self.latest_wireless = {
                "keys": int(msg.keys),
                "lx": float(msg.lx),
                "ly": float(msg.ly),
                "rx": float(msg.rx),
                "ry": float(msg.ry),
            }
            self._wireless_keys = int(msg.keys)

    def _get_low_state(self):
        with self.lock:
            if self.latest_low_state is None:
                return None
            return self.latest_low_state.copy()

    def _update_command_from_wireless(self):
        with self.lock:
            wireless = None if self.latest_wireless is None else self.latest_wireless.copy()
        if wireless is None:
            return

        # Match the training command order [vx, vy, yaw_rate]. Do not convert
        # the right stick to a target heading: Go2Joystick was trained on yaw rate.
        self._command = np.array(
            [
                np.clip(wireless["ly"], -1.0, 1.0) * cmd_max_vx,
                np.clip(wireless["lx"], -1.0, 1.0) * cmd_max_vy,
                np.clip(wireless["rx"], -1.0, 1.0) * cmd_max_yaw,
            ],
            dtype=np.float32,
        )

    def _key_pressed(self, key_name: str) -> bool:
        bit = self._key_bits[key_name]
        return (self._wireless_keys & (1 << bit)) != 0

    def _shutdown_requested(self) -> bool:
        start_select = self._key_pressed("start") and self._key_pressed("select")
        requested = start_select and not self._last_start_select
        self._last_start_select = start_select
        return requested

    def _joint_angles_mujoco_order(self, low_state: dict) -> np.ndarray:
        return np.asarray([low_state["motor_q"][i] for i in idx_map], dtype=np.float32)

    def _joint_vel_mujoco_order(self, low_state: dict) -> np.ndarray:
        return np.asarray([low_state["motor_dq"][i] for i in idx_map], dtype=np.float32)

    def _gravity_local(self, quaternion: np.ndarray) -> np.ndarray:
        return rotate_inv(np.array([0.0, 0.0, -1.0], dtype=np.float32), quaternion)

    def _build_obs(self, low_state: dict) -> np.ndarray:
        """Build the exact 45-D Go2Joystick policy state, before ONNX norm."""
        gyro = np.asarray(low_state["imu_gyro"], dtype=np.float32)
        gravity = self._gravity_local(low_state["imu_quat"])
        joint_pos = self._joint_angles_mujoco_order(low_state)
        joint_vel = self._joint_vel_mujoco_order(low_state)
        obs = np.concatenate(
            [
                gyro,
                gravity,
                joint_pos - self._default_angles,
                joint_vel,
                self._last_action,
                self._command,
            ]
        ).astype(np.float32)
        if obs.shape != (policy_obs_dim,):
            raise RuntimeError(f"Expected obs shape ({policy_obs_dim},), got {obs.shape}")
        if not np.isfinite(obs).all():
            raise FloatingPointError("Non-finite Go2Joystick observation")
        return obs

    def _infer_action(self, obs: np.ndarray) -> np.ndarray:
        policy_obs = np.ascontiguousarray(obs.reshape(1, -1), dtype=np.float32)
        actions = self._policy.run(["actions"], {self._input_name: policy_obs})[0]
        actions = np.asarray(actions[0], dtype=np.float32)
        if actions.shape != (policy_action_dim,):
            raise RuntimeError(
                f"Expected action shape ({policy_action_dim},), got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise FloatingPointError("Non-finite ONNX action")
        return np.clip(actions, -1.0, 1.0)

    def _write_ctrl(self, ctrl: np.ndarray, kp: float, kd: float):
        # ctrl is in MuJoCo policy order; LowCmd is in Unitree order.
        for unitree_index in range(policy_action_dim):
            mujoco_index = int(idx_map[unitree_index])
            self.cmd.motor_cmd[unitree_index].q = float(ctrl[mujoco_index])
            self.cmd.motor_cmd[unitree_index].dq = 0.0
            self.cmd.motor_cmd[unitree_index].kp = float(kp)
            self.cmd.motor_cmd[unitree_index].kd = float(kd)
            self.cmd.motor_cmd[unitree_index].tau = 0.0

    def _publish(self):
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

    def standup_to_default_step(self, dt: float, duration: float) -> bool:
        low_state = self._get_low_state()
        if low_state is None:
            return False

        if not self._standup_initialized:
            self._standup_start_qpos = self._joint_angles_mujoco_order(low_state)
            self._standup_t = 0.0
            self._standup_initialized = True

        self._standup_t = min(self._standup_t + dt, duration)
        alpha = 1.0 if duration <= 1e-6 else self._standup_t / duration
        target_qpos = self._default_angles
        ctrl = (1.0 - alpha) * self._standup_start_qpos + alpha * target_qpos
        self._write_ctrl(ctrl, kp=stand_kp_up, kd=stand_kd)
        self._publish()
        return alpha >= 1.0

    def _hold_joint_pos(self, target_qpos: np.ndarray, kp: float, kd: float):
        self._write_ctrl(target_qpos, kp=kp, kd=kd)
        self._publish()

    def shutdown_step(self, dt: float) -> bool:
        if self._shutdown_requested():
            self._shutdown_active = True
            self._shutdown_t = 0.0
        if not self._shutdown_active:
            return False

        self._shutdown_t += dt
        stand_up_duration = 2.0
        hold_duration = 0.6
        stand_down_duration = 2.0
        if self._shutdown_t < stand_up_duration:
            self._hold_joint_pos(self._default_angles, stand_kp_up, stand_kd)
        elif self._shutdown_t < stand_up_duration + hold_duration:
            self._hold_joint_pos(self._default_angles, stand_kp_up, stand_kd)
        elif self._shutdown_t < stand_up_duration + hold_duration + stand_down_duration:
            self._hold_joint_pos(stand_down_joint_pos[idx_map], stand_kp_down, stand_kd)
        else:
            # stand_down_joint_pos is specified in Unitree motor order.
            self._hold_joint_pos(stand_down_joint_pos[idx_map], stand_kp_down, stand_kd)
        return True

    def joystick_control(self):
        if self._counter % self._n_substeps == 0:
            low_state = self._get_low_state()
            if low_state is None:
                return
            self._update_command_from_wireless()
            obs = self._build_obs(low_state)
            action = self._infer_action(obs)
            ctrl = self._default_angles + action * action_scale
            self._write_ctrl(ctrl, kp=Kp, kd=Kd)
            self._last_action = action
        self._publish()
        self._counter += 1
