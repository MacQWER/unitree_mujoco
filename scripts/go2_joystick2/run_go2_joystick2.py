import sys
import time

from etils import epath

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from consts import sim_dt
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

    standup_done = False
    standup_duration = 3.0
    while True:
        step_start = time.perf_counter()

        if not standup_done:
            standup_done = controller.standup_to_default_step(
                dt=sim_dt, duration=standup_duration
            )
        else:
            if not controller.shutdown_step(sim_dt):
                controller.joystick_control()

        time_until_next_step = sim_dt - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
