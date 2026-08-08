"""Publish keyboard commands on the Unitree wireless-controller DDS topic.

This publisher uses the same key layout as ``go2_joystick2``.  The command
scales are specific to the Go2Joystick PPO policy and must stay in sync with
the policy's command distribution.
"""

import argparse
import time

import pygame

from consts import cmd_max_vx, cmd_max_vy, cmd_max_yaw
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_


KEY_BITS = {
    "R1": 0,
    "L1": 1,
    "start": 2,
    "select": 3,
    "R2": 4,
    "L2": 5,
    "F1": 6,
    "F2": 7,
    "A": 8,
    "B": 9,
    "X": 10,
    "Y": 11,
    "up": 12,
    "right": 13,
    "down": 14,
    "left": 15,
}

KEYBOARD_CMD_STEP_VX = 0.5
KEYBOARD_CMD_STEP_VY = 0.4
KEYBOARD_CMD_STEP_YAW = cmd_max_yaw / 4.0


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Publish Go2Joystick keyboard commands to rt/wirelesscontroller."
    )
    parser.add_argument(
        "--mode",
        choices=("sim", "real"),
        default="sim",
        help="sim uses DDS domain 1/lo; real uses domain 0 and the robot NIC.",
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
    return parser.parse_args()


def _init_channel(mode, interface, domain_id):
    if domain_id is None:
        domain_id = 1 if mode == "sim" else 0
    if interface is None and mode == "sim":
        interface = "lo"
    if interface is None:
        ChannelFactoryInitialize(domain_id)
    else:
        ChannelFactoryInitialize(domain_id, interface)
    return domain_id, interface


def _clamp(value, low, high):
    return max(low, min(high, value))


def _safe_div(value, denom):
    if denom <= 0:
        return 0.0
    return _clamp(value / denom, -1.0, 1.0)


def _build_key_value(keys):
    key_state = [0] * 16
    key_state[KEY_BITS["A"]] = int(keys[pygame.K_j])
    key_state[KEY_BITS["B"]] = int(keys[pygame.K_h])
    key_state[KEY_BITS["X"]] = int(keys[pygame.K_u])
    key_state[KEY_BITS["Y"]] = int(keys[pygame.K_o])
    key_state[KEY_BITS["start"]] = int(keys[pygame.K_RETURN])
    key_state[KEY_BITS["select"]] = int(keys[pygame.K_BACKSPACE])
    key_state[KEY_BITS["up"]] = int(keys[pygame.K_UP])
    key_state[KEY_BITS["right"]] = int(keys[pygame.K_RIGHT])
    key_state[KEY_BITS["down"]] = int(keys[pygame.K_DOWN])
    key_state[KEY_BITS["left"]] = int(keys[pygame.K_LEFT])
    key_state[KEY_BITS["L1"]] = int(keys[pygame.K_z])
    key_state[KEY_BITS["R1"]] = int(keys[pygame.K_c])
    key_state[KEY_BITS["L2"]] = int(keys[pygame.K_x])
    key_state[KEY_BITS["R2"]] = int(keys[pygame.K_v])

    key_value = 0
    for index, value in enumerate(key_state):
        key_value += value << index
    return key_value


def _draw_text(screen, font, text, y, color=(240, 240, 240)):
    screen.blit(font.render(text, True, color), (12, y))


def main():
    args = _parse_args()
    domain_id, interface = _init_channel(
        args.mode, args.interface, args.domain_id
    )

    publisher = ChannelPublisher("rt/wirelesscontroller", WirelessController_)
    publisher.Init()
    msg = unitree_go_msg_dds__WirelessController_()

    pygame.init()
    pygame.display.set_caption("Go2Joystick Keyboard Wireless Controller")
    screen = pygame.display.set_mode((900, 190))
    font = pygame.font.Font(None, 24)
    title_font = pygame.font.Font(None, 30)
    clock = pygame.time.Clock()

    vx = 0.0
    vy = 0.0
    yaw_rate = 0.0
    running = True
    try:
        print(
            f"Publishing rt/wirelesscontroller mode={args.mode} "
            f"domain_id={domain_id} interface={interface or 'default'}"
        )
        print(
            f"Command limits: vx={cmd_max_vx:.2f}, vy={cmd_max_vy:.2f}, "
            f"yaw_rate={cmd_max_yaw:.2f}"
        )
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        vx = 0.0
                        vy = 0.0
                        yaw_rate = 0.0
                    elif event.key == pygame.K_w:
                        vx += KEYBOARD_CMD_STEP_VX
                    elif event.key == pygame.K_s:
                        vx -= KEYBOARD_CMD_STEP_VX
                    elif event.key == pygame.K_a:
                        vy += KEYBOARD_CMD_STEP_VY
                    elif event.key == pygame.K_d:
                        vy -= KEYBOARD_CMD_STEP_VY
                    elif event.key == pygame.K_q:
                        yaw_rate += KEYBOARD_CMD_STEP_YAW
                    elif event.key == pygame.K_e:
                        yaw_rate -= KEYBOARD_CMD_STEP_YAW

            vx = _clamp(vx, -cmd_max_vx, cmd_max_vx)
            vy = _clamp(vy, -cmd_max_vy, cmd_max_vy)
            yaw_rate = _clamp(yaw_rate, -cmd_max_yaw, cmd_max_yaw)

            keys = pygame.key.get_pressed()
            msg.keys = _build_key_value(keys)
            # Controller decodes ly -> vx, lx -> vy, rx -> yaw rate.
            msg.ly = _safe_div(vx, cmd_max_vx)
            msg.lx = _safe_div(vy, cmd_max_vy)
            msg.rx = _safe_div(yaw_rate, cmd_max_yaw)
            msg.ry = float(keys[pygame.K_i]) - float(keys[pygame.K_k])
            publisher.Write(msg)

            screen.fill((10, 10, 10))
            _draw_text(
                screen,
                title_font,
                "Keyboard -> rt/wirelesscontroller (Go2Joystick PPO)",
                12,
                color=(120, 220, 180),
            )
            _draw_text(
                screen,
                font,
                f"vx={vx:+.2f}  vy={vy:+.2f}  yaw_rate={yaw_rate:+.2f}  "
                f"ly={msg.ly:+.2f}  lx={msg.lx:+.2f}  rx={msg.rx:+.2f}",
                48,
            )
            _draw_text(
                screen,
                font,
                "W/S: vx   A/D: vy   Q/E: yaw rate   Space: zero   Esc: quit",
                80,
            )
            _draw_text(
                screen,
                font,
                "Enter+Backspace: emergency shutdown   J/H/U/O: A/B/X/Y   "
                "Arrows: D-pad   Z/C: L1/R1   X/V: L2/R2",
                112,
            )
            _draw_text(
                screen,
                font,
                f"limits: vx +/-{cmd_max_vx:.2f} m/s, vy +/-{cmd_max_vy:.2f} m/s, "
                f"yaw +/-{cmd_max_yaw:.2f} rad/s",
                144,
                color=(170, 170, 170),
            )
            pygame.display.flip()
            clock.tick(50)
    finally:
        msg.keys = 0
        msg.lx = 0.0
        msg.ly = 0.0
        msg.rx = 0.0
        msg.ry = 0.0
        publisher.Write(msg)
        pygame.quit()


if __name__ == "__main__":
    main()
