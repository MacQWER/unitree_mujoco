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

# unitree_idx = idx_map[mujoco_idx].
# The policy uses the MuJoCo joint order; LowCmd uses Unitree order.
idx_map = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int32)

# Go2Joystick was trained at ctrl_dt=0.02 and sim_dt=0.004.
# The Unitree deployment loop runs at 0.002 s and evaluates the policy every
# 10 low-level ticks, preserving the 20 ms policy period.
sim_dt = 0.002
ctrl_dt = 0.02

# Go2Joystick command amplitudes from joystick.py command_config.a.
cmd_max_vx = 1.5
cmd_max_vy = 0.8
cmd_max_yaw = 1.2

# Go2Joystick action and low-level PD settings.
action_scale = 0.5
Kp = 35.0
Kd = 0.5
stand_kp_up = 60.0
stand_kp_down = 20.0
stand_kd = 3.5

# Policy observation dimensions and order:
# [gyro(3), gravity(3), joint_pos-default(12), joint_vel(12),
#  last_action(12), command(vx, vy, yaw_rate)(3)] = 45.
policy_obs_dim = 45
policy_action_dim = 12
