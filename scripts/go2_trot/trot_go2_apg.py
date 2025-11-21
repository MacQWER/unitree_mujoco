import time
import sys
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.utils.crc import CRC

from consts import sim_dt
from controller import Go2OnnxController

from etils import epath
_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / ".." / ".." / "onnx"

# env parameters
runing_time = 0.0
crc = CRC()

input("Press enter to start")

if __name__ == '__main__':

    if len(sys.argv) <2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    controller = Go2OnnxController((_ONNX_DIR / "go2_apg_policy.onnx").as_posix())

    while True:
        step_start = time.perf_counter()

        runing_time += sim_dt

        if (runing_time < 3.0):
            # Stand up in first 3 second
            # Total time for standing up or standing down is about 1.2s
            phase = np.tanh(runing_time / 1.2)
            controller.stand_control(phase=phase)
        else:
            # Then trot
            controller.trot_control()

        time_until_next_step = sim_dt - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)