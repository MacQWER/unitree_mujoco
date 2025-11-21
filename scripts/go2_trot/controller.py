import onnxruntime as rt
import numpy as np
import threading
from TrotUtil import cos_wave, make_kinematic_ref, rotate_inv
from consts import (default_qpos, Kp, Kd, idx_map, 
                    sim_dt, ctrl_dt, action_scale, stand_down_joint_pos)

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC

# ---------------------------------------------------------------------
# ONNX Controller (corrected)
# ---------------------------------------------------------------------
class Go2OnnxController:

    def __init__(self, policy_path):
        # Load ONNX policy
        self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

        self._default_angles = default_qpos[7:]
        self._default_qpos = default_qpos
        self._action_scale = action_scale

        # last_action should be raw action ∈ [-1,1], not the ctrl
        self._last_action = self._default_angles.copy()

        self._counter = 0
        self._n_substeps = int(ctrl_dt / sim_dt)

        # gait reference (same as JAX)
        step_k = 13
        kin_q = make_kinematic_ref(cos_wave, step_k, scale=0.3, dt=ctrl_dt)
        kin_q = np.array(kin_q) + np.array(self._default_angles)

        self._kinematic_ref_qpos = kin_q
        self._step_idx = 0
        self._l_cycle = kin_q.shape[0]

        # unitree api: state subscriber
        self.latest_state = None
        self.lock = threading.Lock()

        # unitree api: pub
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0
        
        self.crc = CRC()

        # unitree api: sub
        low_state_suber = ChannelSubscriber("rt/lowstate", LowState_)
        low_state_suber.Init(self.LowStateHandler, 10)

    # ---------------------------------------------------------------------
    # Unitree LowState handler
    # ---------------------------------------------------------------------
    def LowStateHandler(self, msg: LowState_):
        with self.lock:
            self.latest_state = {
                "imu_quat": np.array(msg.imu_state.quaternion),
                "imu_gyro": np.array(msg.imu_state.gyroscope),
                "motor_q": np.array([m.q for m in msg.motor_state[:12]]),
                "motor_dq": np.array([m.dq for m in msg.motor_state[:12]])
            }
    # ---------------------------------------------------------------------
    # Build observation (STRICTLY same as play_go2_trot_jax)
    # ---------------------------------------------------------------------
    def get_obs(self) -> np.ndarray:
        with self.lock:
            if self.latest_state is None:
                return None
            state = self.latest_state.copy()

            # yaw_rate: sensor gyro z-axis * 0.25
            gyro = state["imu_gyro"]
            yaw_rate = gyro[2] * 0.25

            # orientation: quaternion → g_local
            quat = state["imu_quat"]
            g_world = np.array([0.0, 0.0, -1.0])
            g_local = rotate_inv(g_world, quat)

            # joint angles: MUST follow the same order as JAX
            angles = np.array([state["motor_q"][n] for n in idx_map])

            # reference qpos
            kin_ref = self._kinematic_ref_qpos[self._step_idx]

            # obs structure SAME AS JAX
            obs = np.concatenate([
                [yaw_rate],
                g_local,
                angles - self._default_angles,
                self._last_action,
                kin_ref,
            ])

        return np.clip(obs, -100, 100).astype(np.float32)

    # ---------------------------------------------------------------------
    # Control step
    # ---------------------------------------------------------------------
    def trot_control(self):
        if self._counter % self._n_substeps == 0:
            # collect obs
            obs = self.get_obs()
            if obs is None:
                return  # wait for valid obs
            obs = obs.reshape(1, -1)

            # ONNX forward
            actions, std = self._policy.run(["actions", "std"], {"obs": obs})
            act = actions[0]
            act = np.clip(act, -1.0, 1.0)

            # PD target & update last_action
            ctrl = self._default_angles + act * self._action_scale
            self._last_action = ctrl.copy()
            # for mujoco_idx, unitree_idx in enumerate(idx_map):
            #     self.cmd.motor_cmd[unitree_idx].q = ctrl[mujoco_idx]
            #     self.cmd.motor_cmd[unitree_idx].kp = Kp
            #     self.cmd.motor_cmd[unitree_idx].dq = 0.0
            #     self.cmd.motor_cmd[unitree_idx].kd = Kd
            #     self.cmd.motor_cmd[unitree_idx].tau = 0.0
            for i in range(12):
                self.cmd.motor_cmd[i].q = ctrl[idx_map[i]]
                self.cmd.motor_cmd[i].dq = 0.0 
                self.cmd.motor_cmd[i].kp = Kp
                self.cmd.motor_cmd[i].kd = Kd
                self.cmd.motor_cmd[i].tau = 0.0

            # phase update
            self._step_idx = (self._step_idx + 1) % self._l_cycle

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)
        self._counter += 1

    def stand_control(self, phase):

        for i in range(12):
            self.cmd.motor_cmd[i].q = phase * default_qpos[i+7] + (
                1 - phase) * stand_down_joint_pos[i]
            self.cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 3.5
            self.cmd.motor_cmd[i].tau = 0.0

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)