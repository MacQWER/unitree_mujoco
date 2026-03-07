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

# Joystick2 policy scales from your verified play_go2_joystick2.py.
action_scale = np.array([0.5, 0.5, 0.5] * 4, dtype=np.float32)
command = np.array([1.0, 0.0, 0.5], dtype=np.float32)
velocity_is_world_frame = True
cmd_max_vx = 1.5
cmd_max_vy = 0.80
cmd_max_w = 1.2

# Policy-time PD gains used for lowcmd.
Kp = 35.0
Kd = 0.5

stand_kp_up = 60.0
stand_kp_down = 20.0
stand_kd = 3.5
