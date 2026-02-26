import threading
from typing import Optional

import numpy as np
import onnxruntime as rt

from TrotUtil import cos_wave, make_kinematic_ref, rotate_inv
from consts import (
    default_qpos,
    stand_up_joint_pos,
    stand_down_joint_pos,
    idx_map,
    sim_dt,
    ctrl_dt,
    action_scale,
    command,
    velocity_is_world_frame,
    Kp,
    Kd,
    stand_kp_up,
    stand_kp_down,
    stand_kd,
)

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    LowCmd_,
    LowState_,
    SportModeState_,
)
from unitree_sdk2py.utils.crc import CRC


class Go2Joystick2OnnxController:
    """Anchor + residual ONNX controller via Unitree SDK2 low-level API."""

    def __init__(self, anchor_policy_path: str, residual_policy_path: str):
        self._anchor_policy = rt.InferenceSession(
            anchor_policy_path, providers=["CPUExecutionProvider"]
        )
        self._residual_policy = rt.InferenceSession(
            residual_policy_path, providers=["CPUExecutionProvider"]
        )

        self._default_angles = default_qpos[7:].astype(np.float32)
        self._action_scale = action_scale.astype(np.float32)
        self._command = command.astype(np.float32)

        self._last_action = np.zeros(12, dtype=np.float32)
        self._anchor_action = np.zeros(12, dtype=np.float32)

        self._counter = 0
        self._n_substeps = int(ctrl_dt / sim_dt)

        step_k = 13
        kin_q = make_kinematic_ref(cos_wave, step_k, scale=0.3, dt=ctrl_dt)
        self._kinematic_ref_qpos = (np.array(kin_q) + self._default_angles).astype(
            np.float32
        )
        self._step_idx = 0
        self._l_cycle = self._kinematic_ref_qpos.shape[0]

        self.lock = threading.Lock()
        self.latest_low_state = None
        self.latest_high_state = None

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
        high_state_suber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        low_state_suber.Init(self.LowStateHandler, 10)
        high_state_suber.Init(self.HighStateHandler, 10)

    def LowStateHandler(self, msg: LowState_):
        with self.lock:
            self.latest_low_state = {
                "imu_quat": np.array(msg.imu_state.quaternion, dtype=np.float32),
                "imu_gyro": np.array(msg.imu_state.gyroscope, dtype=np.float32),
                "motor_q": np.array([m.q for m in msg.motor_state[:12]], dtype=np.float32),
                "motor_dq": np.array([m.dq for m in msg.motor_state[:12]], dtype=np.float32),
            }

    def HighStateHandler(self, msg: SportModeState_):
        with self.lock:
            self.latest_high_state = {
                "velocity": np.array(msg.velocity, dtype=np.float32),
            }

    def _get_states(self):
        with self.lock:
            if self.latest_low_state is None:
                return None, None
            low = self.latest_low_state.copy()
            high = None if self.latest_high_state is None else self.latest_high_state.copy()
        return low, high

    def _joint_angles_mujoco_order(self, low_state: dict) -> np.ndarray:
        return np.array([low_state["motor_q"][n] for n in idx_map], dtype=np.float32)

    def _joint_vel_mujoco_order(self, low_state: dict) -> np.ndarray:
        return np.array([low_state["motor_dq"][n] for n in idx_map], dtype=np.float32)

    def _g_local(self, quat: np.ndarray) -> np.ndarray:
        return rotate_inv(np.array([0.0, 0.0, -1.0], dtype=np.float32), quat).astype(
            np.float32
        )

    def _build_anchor_obs(self, low_state: dict) -> np.ndarray:
        yaw_rate = np.float32(low_state["imu_gyro"][2] * 0.25)
        g_local = self._g_local(low_state["imu_quat"])
        angles = self._joint_angles_mujoco_order(low_state)
        kin_ref = self._kinematic_ref_qpos[self._step_idx % self._l_cycle]

        obs = np.concatenate(
            [
                np.array([yaw_rate], dtype=np.float32),
                g_local,
                angles - self._default_angles,
                self._last_action,
                kin_ref,
            ]
        )
        return np.clip(obs, -100.0, 100.0).astype(np.float32)

    def _build_residual_obs(self, low_state: dict, high_state: dict) -> np.ndarray:
        if high_state is None:
            v_local = np.zeros(3, dtype=np.float32)
        else:
            # In simulator bridge, sportmodestate.velocity is sourced from frame_vel sensor.
            v_raw = np.array(high_state["velocity"], dtype=np.float32)
            if velocity_is_world_frame:
                v_local = rotate_inv(v_raw, low_state["imu_quat"]).astype(np.float32)
            else:
                v_local = v_raw

        w_local = np.array(low_state["imu_gyro"], dtype=np.float32)
        g_local = self._g_local(low_state["imu_quat"])
        angles = self._joint_angles_mujoco_order(low_state)
        joint_vel = self._joint_vel_mujoco_order(low_state)

        obs = np.concatenate(
            [
                v_local,
                w_local,
                g_local,
                self._command,
                angles - self._default_angles,
                joint_vel,
                self._anchor_action,
            ]
        )
        return np.clip(obs, -100.0, 100.0).astype(np.float32)

    def _infer_anchor(self, anchor_obs: np.ndarray) -> np.ndarray:
        actions, _ = self._anchor_policy.run(
            ["actions", "std"], {"obs": anchor_obs.reshape(1, -1)}
        )
        return np.clip(actions[0], -1.0, 1.0).astype(np.float32)

    def _infer_residual(self, residual_obs: np.ndarray) -> np.ndarray:
        actions, _ = self._residual_policy.run(
            ["actions", "std"], {"obs": residual_obs.reshape(1, -1)}
        )
        return np.clip(actions[0], -1.0, 1.0).astype(np.float32)

    def _write_ctrl(self, ctrl: np.ndarray, kp: float, kd: float):
        for i in range(12):
            self.cmd.motor_cmd[i].q = float(ctrl[idx_map[i]])
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kp = float(kp)
            self.cmd.motor_cmd[i].kd = float(kd)
            self.cmd.motor_cmd[i].tau = 0.0

    def stand_control(self, phase: float, target_qpos: Optional[np.ndarray] = None):
        phase = float(np.clip(phase, 0.0, 1.0))
        if target_qpos is None:
            target_qpos = stand_up_joint_pos
        for i in range(12):
            self.cmd.motor_cmd[i].q = float(
                phase * target_qpos[i]
                + (1.0 - phase) * stand_down_joint_pos[i]
            )
            self.cmd.motor_cmd[i].kp = float(
                phase * stand_kp_up + (1.0 - phase) * stand_kp_down
            )
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = float(stand_kd)
            self.cmd.motor_cmd[i].tau = 0.0

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

    def hold_joint_pos(self, target_qpos: np.ndarray, kp: float, kd: float):
        for i in range(12):
            self.cmd.motor_cmd[i].q = float(target_qpos[i])
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kp = float(kp)
            self.cmd.motor_cmd[i].kd = float(kd)
            self.cmd.motor_cmd[i].tau = 0.0

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

    def joystick_control(self):
        if self._counter % self._n_substeps == 0:
            low_state, high_state = self._get_states()
            if low_state is None:
                return

            anchor_obs = self._build_anchor_obs(low_state)
            self._anchor_action = self._infer_anchor(anchor_obs)

            residual_obs = self._build_residual_obs(low_state, high_state)
            residual_action = self._infer_residual(residual_obs)

            mixed_action = self._anchor_action + residual_action
            ctrl = self._default_angles + mixed_action * self._action_scale

            self._write_ctrl(ctrl, kp=Kp, kd=Kd)
            self._last_action = ctrl.astype(np.float32)
            self._step_idx = (self._step_idx + 1) % self._l_cycle

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)
        self._counter += 1
