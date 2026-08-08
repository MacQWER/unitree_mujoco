import argparse
import time

from etils import epath
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from consts import sim_dt
from controller import Go2JoystickOnnxController

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / ".." / ".." / "onnx"
_POLICY_DEFAULT = (_ONNX_DIR / "go2_joystick_ppo_intermedia.onnx").as_posix()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the single-policy Go2Joystick ONNX controller."
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "real"],
        default="sim",
        help="sim: DDS domain 1 on lo; real: DDS domain 0 on a robot NIC.",
    )
    parser.add_argument(
        "--interface",
        default=None,
        help="DDS network interface. Defaults to lo in sim mode.",
    )
    parser.add_argument(
        "--domain-id",
        type=int,
        default=None,
        help="DDS domain id. Defaults to 1 in sim mode and 0 in real mode.",
    )
    parser.add_argument(
        "--policy",
        default=_POLICY_DEFAULT,
        help="Path to the 45-D Go2Joystick ONNX policy.",
    )
    parser.add_argument(
        "--standup-duration",
        type=float,
        default=3.0,
        help="Duration of interpolation from current pose to the policy default pose.",
    )
    parser.add_argument(
        "--no-standup",
        action="store_true",
        help="Skip stand-up interpolation and enter policy control immediately.",
    )
    return parser.parse_args()


def _init_channel(mode: str, interface: str | None, domain_id: int | None):
    if domain_id is None:
        domain_id = 1 if mode == "sim" else 0
    if interface is None and mode == "sim":
        interface = "lo"
    if interface is None:
        ChannelFactoryInitialize(domain_id)
    else:
        ChannelFactoryInitialize(domain_id, interface)
    return domain_id, interface


def main():
    args = _parse_args()
    domain_id, interface = _init_channel(args.mode, args.interface, args.domain_id)
    controller = Go2JoystickOnnxController(args.policy)

    print(
        f"Starting go2_joystick mode={args.mode} domain_id={domain_id} "
        f"interface={interface or 'default'}"
    )
    print(f"Policy: {args.policy}")
    print("Policy observation: [gyro, gravity, joint_pos-default, joint_vel, last_action, command] (45)")
    print("Command mapping: ly->vx (1.5 m/s), lx->vy (0.8 m/s), rx->yaw_rate (1.2 rad/s)")
    if args.mode == "real":
        print("REAL ROBOT MODE: verify the robot is lifted/clear and the emergency stop is ready.")
    input("Press enter to start")

    standup_done = bool(args.no_standup)
    while True:
        step_start = time.perf_counter()
        if not standup_done:
            standup_done = controller.standup_to_default_step(
                dt=sim_dt, duration=args.standup_duration
            )
        elif not controller.shutdown_step(sim_dt):
            controller.joystick_control()
        remaining = sim_dt - (time.perf_counter() - step_start)
        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    main()
