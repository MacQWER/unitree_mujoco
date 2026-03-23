import argparse
import time

from etils import epath

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from consts import sim_dt
from controller import Go2Joystick2OnnxController

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / ".." / ".." / "onnx"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run Go2 joystick2 ONNX controller for sim2sim or sim2real."
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "real"],
        default="sim",
        help="sim: DDS domain 1 on lo by default, real: DDS domain 0 on a NIC.",
    )
    parser.add_argument(
        "--interface",
        default=None,
        help="Network interface for DDS. Defaults to lo in sim mode.",
    )
    parser.add_argument(
        "--domain-id",
        type=int,
        default=None,
        help="DDS domain id. Defaults to 1 in sim mode and 0 in real mode.",
    )
    parser.add_argument(
        "--anchor-policy",
        default=(_ONNX_DIR / "go2_apg2_anchor_policy_newnew.onnx").as_posix(),
        help="Path to anchor policy ONNX.",
    )
    parser.add_argument(
        "--residual-policy",
        default=(_ONNX_DIR / "go2_apg2_residual_policy_newnew.onnx").as_posix(),
        help="Path to residual policy ONNX.",
    )
    parser.add_argument(
        "--standup-duration",
        type=float,
        default=3.0,
        help="Stand-up interpolation duration in seconds.",
    )
    parser.add_argument(
        "--no-standup",
        action="store_true",
        help="Skip stand-up interpolation and enter control immediately.",
    )
    return parser.parse_args()


def _init_channel(mode: str, interface: str | None, domain_id: int | None):
    if domain_id is None:
        domain_id = 1 if mode == "sim" else 0

    if interface is None:
        if mode == "sim":
            interface = "lo"
            ChannelFactoryInitialize(domain_id, interface)
        else:
            ChannelFactoryInitialize(domain_id)
    else:
        ChannelFactoryInitialize(domain_id, interface)

    return domain_id, interface


def main():
    args = _parse_args()
    domain_id, interface = _init_channel(args.mode, args.interface, args.domain_id)

    controller = Go2Joystick2OnnxController(
        anchor_policy_path=args.anchor_policy,
        residual_policy_path=args.residual_policy,
    )

    print(
        f"Starting go2_joystick2 mode={args.mode} domain_id={domain_id} interface={interface or 'default'}"
    )
    print(f"Anchor policy: {args.anchor_policy}")
    print(f"Residual policy: {args.residual_policy}")
    input("Press enter to start")

    standup_done = bool(args.no_standup)
    while True:
        step_start = time.perf_counter()

        if not standup_done:
            standup_done = controller.standup_to_default_step(
                dt=sim_dt, duration=args.standup_duration
            )
        else:
            if not controller.shutdown_step(sim_dt):
                controller.joystick_control()

        time_until_next_step = sim_dt - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
