import numpy as np

# MuJoCo home keyframe qpos in unitree_robots/go2/go2.xml.
default_qpos = np.array([
    0.0, 0.0, 0.27, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.9, -1.8, 0.0, 0.9, -1.8,
    0.0, 0.9, -1.8, 0.0, 0.9, -1.8,
], dtype=np.float32)

stand_up_joint_pos = np.array([
    0.00571868, 0.608813, -1.21763,
    -0.00571868, 0.608813, -1.21763,
    0.00571868, 0.608813, -1.21763,
    -0.00571868, 0.608813, -1.21763,
], dtype=np.float32)

stand_down_joint_pos = np.array([
    0.0473455, 1.22187, -2.44375,
    -0.0473455, 1.22187, -2.44375,
    0.0473455, 1.22187, -2.44375,
    -0.0473455, 1.22187, -2.44375,
], dtype=np.float32)

# unitree_idx = idx_map[mujoco_idx], where mujoco order is FR, FL, RR, RL.
idx_map = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int32)

sim_dt = 0.002
ctrl_dt = 0.02

# Observation layout / scaling aligned with
# mujoco_playground._src.locomotion.go2.JoystickGo2.
obs_w_local_scale = 0.25
obs_joint_vel_scale = 0.05

obs_w_local_slice = slice(0, 3)
obs_g_local_slice = slice(3, 6)
obs_command_slice = slice(6, 9)
obs_angles_slice = slice(9, 21)
obs_joint_vel_slice = slice(21, 33)
obs_last_action_slice = slice(33, 45)
obs_kin_ref_slice = slice(45, 57)
obs_anchor_action_slice = slice(57, 69)

# Yaw PID (angle -> yaw-rate command)
yaw_kp = 1.0
yaw_kd = 0.5
yaw_w_clip = 0.5
yaw_err_threshold = 0.2  # rad, yaw误差死区

# Joystick2 policy scales aligned with JoystickGo2.default_config().
anchor_action_scale = np.array([0.3, 0.5, 0.5] * 4, dtype=np.float32)
residual_action_scale = np.array([0.5, 0.8, 0.8] * 4, dtype=np.float32)
command = np.array([1.0, 0.0, 0.5], dtype=np.float32)
imu_gyro_is_body_frame = True
cmd_max_vx = 0.5
cmd_max_vy = 0.2
cmd_max_yaw = 3.141592653589793

# Policy-time PD gains used for lowcmd.
Kp = 35.0
Kd = 0.5

stand_kp_up = 60.0
stand_kp_down = 20.0
stand_kd = 3.5

# Gait phase 相关（与 JoystickGo2 环境对齐）
obs_gait_phase_slice = slice(69, 71)  # gait_phase [sin(θ), cos(θ)]

# 静止判断阈值
stationary_cmd_threshold = 0.01
stationary_w_cmd_threshold = 0.05
