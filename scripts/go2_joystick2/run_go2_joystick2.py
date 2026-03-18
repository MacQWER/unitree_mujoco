import sys
import time

import numpy as np

from etils import epath

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from consts import sim_dt, default_qpos, stand_kp_up, stand_kd
from controller import Go2Joystick2OnnxController

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / ".." / ".." / "onnx"


def main():
    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    controller = Go2Joystick2OnnxController(
        anchor_policy_path=(_ONNX_DIR / "go2_apg2_anchor_policy_new.onnx").as_posix(),
        residual_policy_path=(_ONNX_DIR / "go2_apg2_residual_policy_new.onnx").as_posix(),
    )

    input("Press enter to start")

    running_time = 0.0
    stand_duration = 3.0
    hold_duration = 1.0
    target_qpos = default_qpos[7:]
    while True:
        step_start = time.perf_counter()
        running_time += sim_dt

        if running_time < stand_duration:
            phase = np.tanh(running_time / 1.2)
            controller.stand_control(phase=phase, target_qpos=target_qpos)
        elif running_time < stand_duration + hold_duration:
            controller.hold_joint_pos(target_qpos=target_qpos, kp=stand_kp_up, kd=stand_kd)
        else:
            if not controller.shutdown_step(sim_dt):
                controller.joystick_control()

        time_until_next_step = sim_dt - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
