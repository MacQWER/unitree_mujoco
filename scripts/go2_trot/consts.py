import numpy as np

default_qpos = np.array([   0., 0., 0.27, 1., 0., 0., 0.,
                            0., 0.9, -1.8, 0., 0.9, -1.8, 
                            0., 0.9, -1.8, 0., 0.9, -1.8   ], dtype=float)

stand_down_joint_pos = np.array([
    0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 
    0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
], dtype=float)

Kp = 50.0
Kd = 2.8
sim_dt = 0.002
ctrl_dt = 0.02

# unitree_idx = idx_map[mujoco_idx]
idx_map = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int32)

action_scale=np.array([0.2, 0.8, 0.8] * 4)