import threading
from typing import Optional

import numpy as np
import onnxruntime as rt

from TrotUtil import cos_wave, make_kinematic_ref, rotate_inv, quaternion_to_matrix
from consts import (
    default_qpos,
    stand_up_joint_pos,
    stand_down_joint_pos,
    idx_map,
    sim_dt,
    ctrl_dt,
    anchor_action_scale,
    residual_action_scale,
    velocity_is_world_frame,
    imu_gyro_is_body_frame,
    cmd_max_vx,
    cmd_max_vy,
    cmd_max_w,
    cmd_max_yaw,
    yaw_kp,
    yaw_kd,
    yaw_w_clip,
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
    WirelessController_,
)
from unitree_sdk2py.utils.crc import CRC


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def yaw_from_quat(q: np.ndarray) -> float:
    R = quaternion_to_matrix(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


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
        self._anchor_action_scale = anchor_action_scale.astype(np.float32)
        self._residual_action_scale = residual_action_scale.astype(np.float32)
        self._command = np.zeros(3, dtype=np.float32)
        self._target_yaw = None

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
        self.latest_wireless = None
        self._wireless_keys = 0
        self._last_start_select = False
        self._shutdown_active = False
        self._shutdown_t = 0.0

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
        wireless_suber = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        low_state_suber.Init(self.LowStateHandler, 10)
        high_state_suber.Init(self.HighStateHandler, 10)
        wireless_suber.Init(self.WirelessHandler, 10)

        self._key_bits = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }

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

    def _get_states(self):
        with self.lock:
            if self.latest_low_state is None:
                return None, None
            low = self.latest_low_state.copy()
            high = None if self.latest_high_state is None else self.latest_high_state.copy()
        return low, high

    def _yaw_pd_command(self, low_state: dict, vx: float, vy: float, target_yaw: float) -> np.ndarray:
        yaw = yaw_from_quat(low_state["imu_quat"])
        yaw_err = wrap_to_pi(target_yaw - yaw)
        yaw_rate = float(low_state["imu_gyro"][2])

        w_cmd = yaw_kp * yaw_err - yaw_kd * yaw_rate
        w_cmd = float(np.clip(w_cmd, -yaw_w_clip, yaw_w_clip))
        return np.array([vx, vy, w_cmd], dtype=np.float32)

    def _update_command_from_wireless(self, low_state: dict):
        with self.lock:
            if self.latest_wireless is None:
                return
            lx = float(self.latest_wireless["lx"])
            ly = float(self.latest_wireless["ly"])
            rx = float(self.latest_wireless["rx"])

        vx = float(np.clip(ly, -1.0, 1.0)) * cmd_max_vx
        vy = float(np.clip(lx, -1.0, 1.0)) * cmd_max_vy
        target_yaw_cmd = float(np.clip(rx, -1.0, 1.0)) * cmd_max_yaw

        if self._target_yaw is None:
            self._target_yaw = yaw_from_quat(low_state["imu_quat"])

        # Reset target yaw to current heading when R1 is held (替代右摇杆按下)
        if self._key_pressed("R1"):
            self._target_yaw = yaw_from_quat(low_state["imu_quat"])
        else:
            self._target_yaw = target_yaw_cmd

        self._command = self._yaw_pd_command(low_state, vx, vy, self._target_yaw)

    def _key_pressed(self, key_name: str) -> bool:
        bit = self._key_bits[key_name]
        return (self._wireless_keys & (1 << bit)) != 0

    def _shutdown_requested(self) -> bool:
        start_select = self._key_pressed("start") and self._key_pressed("select")
        requested = start_select and not self._last_start_select
        self._last_start_select = start_select
        return requested

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

        target_qpos = default_qpos[7:].astype(np.float32)
        if self._shutdown_t < stand_up_duration:
            phase = float(np.tanh(self._shutdown_t / 1.0))
            self.stand_control(phase=phase, target_qpos=target_qpos)
        elif self._shutdown_t < stand_up_duration + hold_duration:
            self.hold_joint_pos(target_qpos, kp=stand_kp_up, kd=stand_kd)
        elif self._shutdown_t < stand_up_duration + hold_duration + stand_down_duration:
            t = self._shutdown_t - stand_up_duration - hold_duration
            phase = float(1.0 - np.tanh(t / 1.0))
            self.stand_control(phase=phase, target_qpos=target_qpos)
        else:
            self.hold_joint_pos(stand_down_joint_pos, kp=stand_kp_down, kd=stand_kd)

        return True

    def _joint_angles_mujoco_order(self, low_state: dict) -> np.ndarray:
        return np.array([low_state["motor_q"][n] for n in idx_map], dtype=np.float32)

    def _joint_vel_mujoco_order(self, low_state: dict) -> np.ndarray:
        return np.array([low_state["motor_dq"][n] for n in idx_map], dtype=np.float32)

    def _g_local(self, quat: np.ndarray) -> np.ndarray:
        return rotate_inv(np.array([0.0, 0.0, -1.0], dtype=np.float32), quat).astype(
            np.float32
        )

    def _build_obs(self, low_state: dict, high_state: dict) -> np.ndarray:
        if high_state is None:
            v_local = np.zeros(3, dtype=np.float32)
        else:
            # In simulator bridge, sportmodestate.velocity is sourced from frame_vel sensor.
            v_raw = np.array(high_state["velocity"], dtype=np.float32)
            if velocity_is_world_frame:
                v_local = rotate_inv(v_raw, low_state["imu_quat"]).astype(np.float32)
            else:
                v_local = v_raw

        if imu_gyro_is_body_frame:
            w_local = np.array(low_state["imu_gyro"], dtype=np.float32)
        else:
            w_local = rotate_inv(np.array(low_state["imu_gyro"], dtype=np.float32), low_state["imu_quat"]).astype(np.float32)
        g_local = self._g_local(low_state["imu_quat"])
        angles = self._joint_angles_mujoco_order(low_state)
        joint_vel = self._joint_vel_mujoco_order(low_state)
        kin_ref = self._kinematic_ref_qpos[self._step_idx % self._l_cycle]

        obs = np.concatenate(
            [
                v_local,
                w_local,
                g_local,
                self._command,
                angles - self._default_angles,
                joint_vel,
                self._last_action,
                kin_ref,
                self._anchor_action,
            ]
        )
        return np.clip(obs, -100.0, 100.0).astype(np.float32)

    def _build_anchor_obs(self, low_state: dict, high_state: dict) -> np.ndarray:
        obs = self._build_obs(low_state, high_state)
        obs = obs.copy()
        # Mask command (indices 9-11) and anchor_action (indices 60-71).
        obs[9:12] = 0.0
        obs[60:72] = 0.0
        return obs

    def _build_residual_obs(self, low_state: dict, high_state: dict) -> np.ndarray:
        return self._build_obs(low_state, high_state)

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

            self._update_command_from_wireless(low_state)

            anchor_obs = self._build_anchor_obs(low_state, high_state)
            self._anchor_action = self._infer_anchor(anchor_obs)

            residual_obs = self._build_residual_obs(low_state, high_state)
            residual_action = self._infer_residual(residual_obs)

            mixed_action = (
                self._anchor_action * self._anchor_action_scale
                + residual_action * self._residual_action_scale
            )
            ctrl = self._default_angles + mixed_action

            self._write_ctrl(ctrl, kp=Kp, kd=Kd)
            self._last_action = ctrl.astype(np.float32)
            self._step_idx = (self._step_idx + 1) % self._l_cycle

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)
        self._counter += 1
